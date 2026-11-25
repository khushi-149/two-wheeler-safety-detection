"""
Two-wheeler safety pipeline:
  motorcycle tracking -> rider counting -> helmet check -> triple riding
  -> number plate OCR -> evidence images -> SQLite log.

Riders and helmets are analysed on a crop around EACH tracked motorcycle
(``settings.USE_CROP_ANALYSIS``), so small / distant riders in wide traffic
scenes are detected. Set it to False for the original full-frame method.

Used by detect_video_full.py / detect_webcam.py (command line) and
dashboard.py (Streamlit).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

import settings as cfg
from assignment import (
    assign_heads_to_bikes,
    assign_people_to_bikes,
    bike_region,
    dedupe_detections,
    hip_from_keypoints,
    dedupe_nested_bikes,
    drop_headless_people,
    match_no_helmet_to_motorcycle,
    merge_duplicate_people,
    merge_nested_heads,
    person_seat_point,
    smoothed_count,
    unresolved_rider_heads,
)
from capture import IMAGE_EXTENSIONS, ImageCapture, ThreadedCapture
from geometry import iou
from head_check import head_crop_detections, verify_decision, verify_with_decision
from drawing import (BLUE, GREEN, RED, WHITE, Style, draw_label, draw_tag,
                     draw_warning, make_header)
from plate_reader import AsyncPlateReader, PlateReader, PlateTracker
from rider_association import (
    head_point,
    associate_riders,
    count_riders,
    dedupe_people,
    detect_triple_riding,
    head_from_keypoints,
)
from rider_tracking import MotorcycleTracker, PhotoMotorcycleDetector
from settings import model_path, resolve_path
from violation_db import ViolationDB

log = logging.getLogger(__name__)

# class id of the neutral "helmet unknown" marker (not a detector class)
UNKNOWN_HELMET = -1

try:
    import torch
    USE_GPU = torch.cuda.is_available()
except ImportError:  # torch normally comes with ultralytics
    USE_GPU = False

PERSON_CLASS = cfg.COCO_PERSON_CLASS


def redact_source(source) -> str:
    """Describe a source for messages WITHOUT leaking a stream's user:password."""
    if isinstance(source, int):
        return f"camera {source}"

    text = str(source)

    if "://" in text:
        scheme, _, rest = text.partition("://")
        host = rest.rsplit("@", 1)[-1].split("/")[0]
        return f"{scheme}://{host}"

    return text


def resolve_helmet_classes(names) -> tuple[int, int]:
    """
    Work out (helmet_class_id, no_helmet_class_id) from the model's own class
    names, so a retrained model with a different class order still works.
    Falls back to the defaults in settings when the names are not recognisable.
    """
    helmet_id = no_helmet_id = None

    items = names.items() if isinstance(names, dict) else enumerate(names or [])

    for class_id, name in items:
        label = str(name).lower().replace("_", " ").replace("-", " ")

        if "without" in label or label.startswith("no ") or "no helmet" in label:
            no_helmet_id = int(class_id)
        elif "helmet" in label:
            helmet_id = int(class_id)

    if helmet_id is None or no_helmet_id is None or helmet_id == no_helmet_id:
        return cfg.HELMET_CLASS, cfg.NO_HELMET_CLASS

    return helmet_id, no_helmet_id


# -------------------- Pipeline --------------------

class ViolationPipeline:

    def __init__(self,
                 pose_path=cfg.POSE_MODEL,
                 helmet_path=cfg.HELMET_MODEL,
                 plate_model_path=cfg.PLATE_MODEL,
                 traffic_path=cfg.TRAFFIC_MODEL,
                 use_plates=True,
                 db_path=cfg.DB_PATH,
                 evidence_folder=cfg.EVIDENCE_DIR,
                 video_folder=cfg.OUTPUT_VIDEO_DIR,
                 plate_folder=cfg.PLATE_DIR,
                 image_folder=cfg.OUTPUT_IMAGE_DIR):

        helmet_file = model_path(helmet_path)

        if not os.path.exists(helmet_file):
            raise FileNotFoundError(
                f"Helmet model not found: {helmet_file}\n"
                "Download helmet_detector.pt from the project's GitHub "
                "Releases (or train it with src/train_helmet.py) and place it "
                "in the models/ folder."
            )

        self.traffic_path = model_path(traffic_path)
        self.pose_model = YOLO(model_path(pose_path))
        self.helmet_model = YOLO(helmet_file)
        self.helmet_class, self.no_helmet_class = resolve_helmet_classes(
            getattr(self.helmet_model, "names", None)
        )

        self.db = ViolationDB(db_path)

        self.evidence_folder = resolve_path(evidence_folder)
        self.video_folder = resolve_path(video_folder)
        self.plate_folder = resolve_path(plate_folder)
        self.image_folder = resolve_path(image_folder)

        for folder in (self.evidence_folder, self.video_folder,
                       self.plate_folder, self.image_folder):
            os.makedirs(folder, exist_ok=True)

        self.plate_reader = None
        self.photo_mode = False      # set by run(): photos use a larger analysis size
        self._photo_traffic = None   # photo motorcycle model, loaded on first photo
        self.last_debug = {}         # raw detections of the last analysis (for explain_photo)
        self.debug_refine = False    # explain_photo turns this on to log the re-checks
        self.refine_log = []

        log.info("Compute device: %s", "GPU (CUDA)" if USE_GPU else "CPU only")

        if use_plates:
            plate_file = model_path(plate_model_path)

            if not (os.path.exists(plate_file) or cfg.ALLOW_OCR_FALLBACK):
                log.warning(
                    "%s not found -> plate reading is OFF (it would be slow "
                    "and unreliable without a plate detector). Helmet and "
                    "triple-riding checks still run.", plate_file)
            else:
                try:
                    self.plate_reader = PlateReader(plate_file)
                except ImportError:
                    log.warning("easyocr not installed - plate reading "
                                "disabled. Run: pip install easyocr")

    # ---------------------------------------------------------------
    # Person / helmet detection

    def _person_from_result(self, result, index, box, ox=0, oy=0):
        """Build a person dict; shifts crop coordinates into frame coordinates."""
        keypoints = result.keypoints.xy[index].cpu().tolist()
        confidences = None

        if result.keypoints.conf is not None:
            confidences = result.keypoints.conf[index].cpu().tolist()

        if ox or oy:
            # keep undetected keypoints at (0, 0)
            keypoints = [
                [x + ox, y + oy] if (x or y) else [x, y]
                for x, y in keypoints
            ]

        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

        return {
            "box": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
            "head": head_from_keypoints(keypoints, confidences),
            "hip": hip_from_keypoints(keypoints, confidences),
            "conf": float(box.conf[0]),
            # how many of nose / left eye / right eye the pose model sees clearly
            "face": sum(
                1 for k in (0, 1, 2)
                if confidences is not None and k < len(confidences)
                and confidences[k] >= 0.5
            ),
        }

    def _detect_people(self, frame):
        """Original method: pose on the whole frame."""
        people = []
        pose = self.pose_model(frame, conf=cfg.POSE_CONF, verbose=False)[0]

        if pose.boxes is None or pose.keypoints is None:
            return people

        for index, box in enumerate(pose.boxes):
            if int(box.cls[0]) == PERSON_CLASS:
                people.append(self._person_from_result(pose, index, box))

        return dedupe_people(people)

    def _detect_helmets_fullframe(self, frame):
        result = self.helmet_model(frame, conf=cfg.HELMET_CONF, verbose=False)[0]

        return [
            {
                "box": b.xyxy[0].cpu().tolist(),
                "conf": float(b.conf[0]),
                "cls": int(b.cls[0]),
            }
            for b in result.boxes
        ]

    def _analyze_crops(self, frame, motorcycles):
        """
        Run pose + helmet models on a crop around every motorcycle
        (batched). Returns (people, helmet_detections) in frame coordinates.
        """
        h, w = frame.shape[:2]

        crops, offsets = [], []

        for moto in motorcycles:
            rx1, ry1, rx2, ry2 = bike_region(moto["box"], w, h)

            if rx2 - rx1 < 16 or ry2 - ry1 < 16:
                continue

            crops.append(frame[ry1:ry2, rx1:rx2])
            offsets.append((rx1, ry1))

        if not crops:
            return [], []

        people, helmets = [], []

        imgsz = self._crop_imgsz(crops)

        pose_results = self.pose_model(
            crops, conf=cfg.POSE_CONF, imgsz=imgsz, half=USE_GPU, verbose=False
        )

        for result, (ox, oy) in zip(pose_results, offsets):

            if result.boxes is None or result.keypoints is None:
                continue

            for index, box in enumerate(result.boxes):
                if int(box.cls[0]) == PERSON_CLASS:
                    people.append(
                        self._person_from_result(result, index, box, ox, oy)
                    )

        helmet_results = self.helmet_model(
            crops, conf=cfg.HELMET_CONF, imgsz=imgsz, half=USE_GPU, verbose=False
        )

        for result, (ox, oy) in zip(helmet_results, offsets):
            for b in result.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().tolist()
                helmets.append({
                    "box": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
                    "conf": float(b.conf[0]),
                    "cls": int(b.cls[0]),
                })

        # kept for scripts/explain_photo.py (what the models saw before filtering)
        self.last_debug = {"raw_people": list(people), "raw_helmets": list(helmets)}

        people = dedupe_detections(dedupe_people(people), cfg.PERSON_DUP_IOU)
        # one rider reported twice must not become a second rider
        people = merge_duplicate_people(people)
        helmets = merge_nested_heads(dedupe_detections(helmets))
        # pieces of body (no head anywhere) are not riders
        people = drop_headless_people(people, helmets)
        return people, helmets

    # ---------------------------------------------------------------

    def _crop_imgsz(self, crops):
        """
        Model input size for bike crops. Videos use the fixed CROP_IMGSZ (speed).
        A photo is analysed once, so its size follows the crops' own pixels
        (rounded up to a multiple of 32) between CROP_IMGSZ and CROP_IMGSZ_MAX:
        small heads on a large photo then keep their detail.
        """
        if not self.photo_mode or not crops:
            return cfg.CROP_IMGSZ

        largest = max(max(crop.shape[:2]) for crop in crops)
        size = int(np.ceil(largest / 32.0) * 32)

        return max(cfg.CROP_IMGSZ, min(size, cfg.CROP_IMGSZ_MAX))

    def _note(self, kind, bike, det, look, verdict):
        """Record one re-check for scripts/explain_photo.py (only when debugging)."""
        if not self.debug_refine:
            return

        self.refine_log.append({
            "kind": kind, "bike": bike,
            "orig": (det["cls"], det["conf"]) if det is not None else None,
            "look": [(d["cls"], round(d["conf"], 2)) for d in look],
            "verdict": (verdict["cls"], round(verdict["conf"], 2)) if verdict else None,
        })

    def _recheck(self, frame, person, det, bike_index, kind):
        """
        Zoomed second look at one detected head. Returns the replacement
        detection, or None to keep ``det``:
          "With Helmet" below VERIFY_BELOW   -> "Without" if the zoom clearly says so
          "Without Helmet" below VERIFY_NO_BELOW -> "With" if the zoom clearly says so
        """
        if not cfg.VERIFY_ENABLED:
            return None

        if det["cls"] == self.helmet_class and det["conf"] < cfg.VERIFY_BELOW:
            check = verify_decision
            limit = cfg.VERIFY_WITHOUT_MIN
        elif det["cls"] == self.no_helmet_class and det["conf"] < cfg.VERIFY_NO_BELOW:
            check = verify_with_decision
            limit = cfg.VERIFY_WITH_MIN
        else:
            self._note(kind + " (not re-checked: sure enough)", bike_index, det, [], None)
            return None

        look = head_crop_detections(
            self.helmet_model, frame, person,
            cfg.RECOVERY_CONF, cfg.RECOVERY_IMGSZ, around=det["box"])
        verdict = check(look, self.helmet_class, self.no_helmet_class, limit)

        if verdict is None and det["cls"] == self.helmet_class and cfg.WEAK_WITH_BELOW:
            zoomed_with = max((d["conf"] for d in look if d["cls"] == self.helmet_class),
                              default=0.0)

            if max(det["conf"], zoomed_with) < cfg.WEAK_WITH_BELOW:
                # nothing confirms the "helmet": a cap, hair or a small child
                verdict = {"box": det["box"], "cls": self.no_helmet_class,
                           "conf": det["conf"]}

        self._note(kind, bike_index, det, look, verdict)

        return verdict

    def _refine_heads(self, frame, people, rider_boxes, owners, claims, due):
        """
        Second look at each rider's head on a zoomed crop (see head_check.py):
          * a rider with NO head yet -> try to recover one (or report "no
            helmet" if INFER_NO_HELMET, or if the face is clearly visible and
            INFER_NO_HELMET_IF_FACE);
          * a doubtful head is re-checked and flipped when the zoomed crop
            clearly disagrees (see ``_recheck``).
        Only bikes analysed this frame (``due``) are refined.
        """
        person_by_box = {tuple(p["box"]): p for p in people}
        claim_by_body = {tuple(body): det for body, det in claims}
        owners = list(owners)

        def replace(det, verdict):
            return [(verdict, o[1], o[2]) if o[0] is det else o for o in owners]

        for bike_index in due:
            for body in rider_boxes.get(bike_index, []):
                person = person_by_box.get(tuple(body))

                if person is None:
                    continue

                det = claim_by_body.get(tuple(body))

                if det is not None:
                    verdict = self._recheck(frame, person, det, bike_index, "rider's head")

                    if verdict is not None:
                        owners = replace(det, verdict)

                    continue

                found = head_crop_detections(
                    self.helmet_model, frame, person,
                    cfg.RECOVERY_CONF, cfg.RECOVERY_IMGSZ)
                self._note("rider with no head (recovery look)", bike_index, None, found, None)

                if found:
                    best = max(found, key=lambda d: d["conf"])

                    # not the same head as one already counted
                    if all(iou(best["box"], o[0]["box"]) < 0.5 for o in owners):
                        owners.append((best, bike_index, False))
                elif cfg.INFER_NO_HELMET or (
                        cfg.INFER_NO_HELMET_IF_FACE and person.get("face", 0) >= 2):
                    hx, hy = head_point(person)
                    r = max((body[2] - body[0]) * 0.20, (body[3] - body[1]) * 0.07, 10)
                    owners.append((
                        {"box": [hx - r, hy - r, hx + r, hy + r],
                         "cls": self.no_helmet_class, "conf": 0.0},
                        bike_index, False))

        # Heads with no body of their own (a pillion the pose model missed) get
        # the same second look.
        for det, owner, extra in list(owners):
            if extra and owner in due:
                verdict = self._recheck(
                    frame, {"box": list(det["box"]), "head": None}, det, owner,
                    "hidden pillion's head")

                if verdict is not None:
                    owners = replace(det, verdict)

        return owners

    def _analyze_cached(self, frame, motorcycles, cache, frame_count,
                        min_history_for_triple=cfg.MIN_HISTORY_FOR_TRIPLE):
        """
        Staggered rider/helmet analysis.

        Each bike is re-analysed every ANALYZE_EVERY frames (new bikes
        immediately). In between, its last result is reused and the heads
        are shifted by however far the bike box has moved.

        Returns (people, rider_counts, head_owners). rider_counts already
        includes hidden pillions.
        """
        due = [
            i for i, m in enumerate(motorcycles)
            if m["track_id"] not in cache
            or (m["track_id"] + frame_count) % cfg.ANALYZE_EVERY == 0
        ]

        people = []

        if due:
            due_bikes = [motorcycles[i] for i in due]

            # Crops are only cut for the bikes that are due (speed) ...
            people, helmet_dets = self._analyze_crops(frame, due_bikes)

            # ... but people and heads are matched against ALL current bikes,
            # so a neighbouring bike can claim its own riders.
            counts, rider_boxes, other_boxes = assign_people_to_bikes(
                people, motorcycles
            )
            claims = []
            owners = assign_heads_to_bikes(
                helmet_dets, motorcycles, rider_boxes, other_boxes, claims,
                people=people,
            )
            owners = self._refine_heads(
                frame, people, rider_boxes, owners, claims, due
            )

            for i in due:
                bike = motorcycles[i]
                track_id = bike["track_id"]

                heads = [
                    {"box": det["box"], "cls": det["cls"], "extra": extra}
                    for det, owner_i, extra in owners if owner_i == i
                ]

                # Riders whose head the model never found get a neutral
                # "helmet unknown" marker (never a violation, never counted).
                for box in unresolved_rider_heads(
                        rider_boxes.get(i, []), people,
                        [h["box"] for h in heads]):
                    heads.append({"box": box, "cls": UNKNOWN_HELMET, "extra": False})

                raw = counts[i] + sum(1 for h in heads if h["extra"])

                previous = cache.get(track_id)
                history = (previous["history"] if previous else []) + [raw]
                history = history[-cfg.COUNT_HISTORY:]

                cache[track_id] = {
                    "anchor": (bike["box"][0], bike["box"][1]),
                    "count": smoothed_count(history, min_history_for_triple),
                    "history": history,
                    "heads": heads,
                    "seen": frame_count,
                }

        rider_counts = {}
        head_owners = []

        for i, bike in enumerate(motorcycles):

            entry = cache[bike["track_id"]]
            entry["seen"] = frame_count
            rider_counts[i] = entry["count"]

            dx = bike["box"][0] - entry["anchor"][0]
            dy = bike["box"][1] - entry["anchor"][1]

            for head in entry["heads"]:
                b = head["box"]
                head_owners.append((
                    {"box": [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy],
                     "cls": head["cls"], "conf": 1.0},
                    i,
                    head["extra"],
                ))

        for track_id in [t for t, e in cache.items()
                         if frame_count - e["seen"] > cfg.CACHE_MAX_AGE]:
            del cache[track_id]

        return people, rider_counts, head_owners

    def run(self, source, show_window=False,
            frame_callback=None, callback_every=2, should_stop=None,
            live=False, save_video=True, pace_to_fps=True,
            realtime=False, display_callback=None, smooth_overlay=False,
            frame_stride=1):
        """
        Process a video file, or a live camera/stream when `source` is a
        camera index (int) or `live=True`. Returns a summary dict.

        frame_stride  (video files, not real-time) analyse only every Nth
                      frame: 3 is about three times faster and still tracks
                      bikes fine.

        live        read the newest frame only (no lag), use wall-clock time
        save_video  write the annotated output video
        pace_to_fps with a preview window, play files at their real speed
                    (never faster) so playback looks smooth
        realtime    play a video FILE at its real speed and let detection skip
                    frames to keep up (like a live camera)
        smooth_overlay  False (default): display_callback receives each frame
                    WITH its own boxes, so boxes always sit on the right
                    object (the picture updates at the detector speed).
                    True: the picture plays at full speed and the latest boxes
                    are pasted on top; on a moving bike the boxes lag behind.
        display_callback(frame_bgr)  if given, (with smooth_overlay) a separate thread calls it with
                    the video at full speed (about source fps), drawing the
                    latest detections on top. Detection speed no longer limits
                    how smooth the video looks; the boxes update as fast as
                    the models allow. Called from a background thread.

        frame_callback(annotated_frame, stats) is called every
        `callback_every` frames (used by the Streamlit dashboard).
        """

        live = live or isinstance(source, int)

        is_image = (
            not live
            and isinstance(source, str)
            and source.lower().endswith(IMAGE_EXTENSIONS)
        )

        realtime = (realtime or live) and not is_image
        self.photo_mode = is_image
        paced_file = realtime and not live
        wall_clock = live or paced_file

        if live:
            cap = ThreadedCapture(source)
        elif is_image:
            cap = ImageCapture(source)
        elif paced_file:
            cap = ThreadedCapture(source, paced=True, autostart=False)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open source: {redact_source(source)}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = 0 if live else (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1)

        if fps <= 0 or fps > 120:
            fps = 30 if live else 25

        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Invalid video dimensions.")

        style = Style(height, width)

        if live:
            video_name = f"camera_{source}" if isinstance(source, int) else "stream"
        else:
            video_name = os.path.basename(source)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{stamp}_{os.path.splitext(video_name)[0]}"

        output_video_path = os.path.join(self.video_folder, f"tracked_{run_id}.mp4")

        # only a plain file read can skip frames cheaply (grab = decode only)
        stride = (max(1, int(frame_stride))
                  if not (live or paced_file or is_image) else 1)

        writer = None

        if save_video and not is_image and not paced_file:
            # Live frames are dropped when detection is slow, so record at a
            # modest fixed rate instead of the camera's nominal fps.
            writer = cv2.VideoWriter(
                output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                15 if live else fps / stride, (width, height),
            )

            if not writer.isOpened():
                cap.release()
                raise RuntimeError("Could not create output video.")

        # Fresh detector + tracker per run so track IDs never carry over
        # from a previous video.
        traffic_model = None if is_image else YOLO(self.traffic_path)
        if is_image:
            # a photo has no motion to track: retry harder before giving up.
            # The (large) photo model is loaded once and kept, so a batch of
            # photos does not reload it every time.
            if self._photo_traffic is None:
                self._photo_traffic = YOLO(model_path(cfg.PHOTO_TRAFFIC_MODEL))

            tracker = PhotoMotorcycleDetector(
                self._photo_traffic, confidence=cfg.MOTORCYCLE_CONF
            )
        else:
            tracker = MotorcycleTracker(
                traffic_model, confidence=cfg.MOTORCYCLE_CONF
            )
        plate_tracker = PlateTracker()

        # One background worker per run (closed again at the end)
        plate_worker = (
            AsyncPlateReader(self.plate_reader)
            if self.plate_reader is not None else None
        )

        source_index = 0
        frame_count = 0
        evidence_count = 0
        annotated = None
        t_start = time.perf_counter()
        t_track = t_analyze = 0.0
        analysis_cache = {}

        seen_ids = set()
        helmet_ids = set()
        triple_ids = set()
        last_evidence_time = {}
        plate_images = {}

        def store_plate(track_id, result):
            """Keep a plate reading if it is the best so far (+ image + DB row)."""
            if result is None or not plate_tracker.update(track_id, result):
                return

            plate_path = os.path.join(
                self.plate_folder, f"{run_id}_ID{track_id}.jpg"
            )
            cv2.imwrite(plate_path, result["crop"])
            plate_images[track_id] = plate_path

            self.db.update_plate(
                run_id, track_id, result["text"],
                result["conf"], plate_path,
            )

        def collect_plate_results():
            """Store finished background OCR reads (best per track) + DB."""
            if plate_worker is None:
                return

            for track_id, result in plate_worker.poll():
                store_plate(track_id, result)

        def make_stats():
            return {
                "run_id": run_id,
                "frame": cap.frame_index if paced_file else source_index,
                "frames_analysed": frame_count,
                "total_frames": total_frames,
                "detector_fps": frame_count / max(1e-6, time.perf_counter() - t_start),
                "motorcycles": len(seen_ids),
                "helmet_violations": len(helmet_ids),
                "triple_riding": len(triple_ids),
                "plates_read": len(plate_tracker.plates),
                "evidence_saved": evidence_count,
            }

        # ---------- Smooth display, decoupled from detection ----------
        display_thread = None
        display_stop = threading.Event()
        overlay = {"data": None}
        last_push = {"t": time.perf_counter()}
        overlay_lock = threading.Lock()

        disp_w = min(width, 960)
        disp_h = max(1, int(height * disp_w / width))

        if display_callback is not None and realtime and smooth_overlay:

            def display_loop():
                last_seq = -1

                while not display_stop.is_set():
                    frame_now, seq = cap.peek()

                    if frame_now is None or seq == last_seq:
                        time.sleep(0.004)
                        continue

                    last_seq = seq

                    try:
                        small = cv2.resize(frame_now, (disp_w, disp_h),
                                           interpolation=cv2.INTER_AREA)

                        with overlay_lock:
                            data = overlay["data"]

                        if data is not None:
                            mask, pixels = data
                            np.copyto(small, pixels, where=mask)

                        display_callback(small)
                    except Exception as error:  # never let display kill the run
                        log.warning("Display error: %s", error)
                        time.sleep(0.05)

            display_thread = threading.Thread(target=display_loop, daemon=True)

        elif display_callback is not None and realtime:
            # Boxes are drawn on the frame they were computed on, so the picture
            # only changes as fast as detection runs. On a slow computer the
            # first result can take many seconds: until then (and whenever the
            # detector falls behind) show the plain video so the screen is
            # never blank, then switch back to the boxed pictures.
            def keepalive_loop():
                last_seq = -1

                while not display_stop.is_set():
                    time.sleep(0.04)

                    if time.perf_counter() - last_push["t"] < 0.8:
                        continue

                    frame_now, seq = cap.peek()

                    if frame_now is None or seq == last_seq:
                        continue

                    last_seq = seq

                    try:
                        small = cv2.resize(frame_now, (disp_w, disp_h),
                                           interpolation=cv2.INTER_AREA)
                        cv2.putText(small, "Analysing...", (10, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 0, 0), 4, cv2.LINE_AA)
                        cv2.putText(small, "Analysing...", (10, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 255, 255), 2, cv2.LINE_AA)
                        display_callback(small)
                    except Exception as error:
                        log.warning("Display error: %s", error)

            display_thread = threading.Thread(target=keepalive_loop, daemon=True)

        if paced_file:
            cap.start()          # the file starts playing now, models are loaded

        if display_thread is not None:
            display_thread.start()

        try:
            while True:

                if should_stop is not None and should_stop():
                    break

                for _ in range(stride - 1):
                    if cap.grab():
                        source_index += 1

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1
                source_index += 1

                if live:
                    video_time = time.perf_counter() - t_start
                elif paced_file:
                    video_time = cap.frame_index / fps
                else:
                    video_time = source_index / fps

                # ---------- Motorcycle tracking ----------
                t0 = time.perf_counter()

                motorcycles = dedupe_nested_bikes([
                    {"box": m["box"], "conf": m["conf"], "track_id": m["track_id"]}
                    for m in tracker.update(frame)
                ])

                t_track += time.perf_counter() - t0

                for moto in motorcycles:
                    seen_ids.add(moto["track_id"])

                # ---------- Riders + helmets ----------
                head_owners = None  # filled in crop mode
                t0 = time.perf_counter()

                if not motorcycles:
                    people, helmet_dets = [], []
                    rider_counts, riding_status = {}, {}

                elif cfg.USE_CROP_ANALYSIS:
                    people, rider_counts, head_owners = self._analyze_cached(
                        frame, motorcycles, analysis_cache, frame_count,
                        # a photo has one frame: do not wait for 3 agreeing checks
                        min_history_for_triple=(
                            1 if is_image else cfg.MIN_HISTORY_FOR_TRIPLE
                        ),
                    )
                    helmet_dets = []
                    riding_status = detect_triple_riding(rider_counts)

                else:
                    people = self._detect_people(frame)
                    associations = associate_riders(
                        people,
                        [{"box": m["box"], "conf": m["conf"]} for m in motorcycles],
                    )
                    rider_counts = count_riders(associations)
                    riding_status = detect_triple_riding(rider_counts)
                    helmet_dets = self._detect_helmets_fullframe(frame)

                t_analyze += time.perf_counter() - t0

                annotated = frame.copy()

                # Rectangles already used by labels on this frame, so labels
                # never sit on top of each other. The top-left corner is kept
                # free for the warning banners.
                banner_w, _ = cv2.getTextSize(
                    "TRIPLE RIDING DETECTED", cv2.FONT_HERSHEY_SIMPLEX,
                    style.warning_scale, style.warning_thickness)[0]
                banner_w = min(width, banner_w + 2 * style.pad + 20)
                # (a photo gets a header strip above the picture instead)
                placed = [] if is_image else [
                    (0, 0, banner_w,
                     style.warning_start + style.warning_line + style.pad)]

                if cfg.DEBUG_DRAW_PEOPLE:
                    for person in people:
                        px1, py1, px2, py2 = map(int, person["box"])
                        cv2.rectangle(annotated, (px1, py1), (px2, py2),
                                      BLUE, style.thin)

                        seat_x, seat_y = person_seat_point(person)
                        cv2.circle(annotated, (int(seat_x), int(seat_y)),
                                   max(4, style.thin * 2), (255, 0, 255), -1)

                # ---------- Draw helmet detections ----------
                no_helmet_by_moto = {}

                # Fallback (full-frame) mode: old geometric matching
                if head_owners is None:
                    head_owners = []

                    for det in helmet_dets:
                        idx = match_no_helmet_to_motorcycle(det["box"], motorcycles)

                        if idx is not None:
                            head_owners.append((det, idx, False))

                # Faces are kept clear: no label may be drawn over a head.
                head_rects = [tuple(int(v) for v in det["box"])
                              for det, _idx, _extra in head_owners]
                placed.extend(head_rects)

                # Only heads that belong to a rider are drawn / counted;
                # pedestrians' heads are ignored.
                for det, idx, _is_extra in head_owners:

                    coords = det["box"]
                    hx1, hy1, hx2, hy2 = map(int, coords)

                    if det["cls"] == self.no_helmet_class:

                        no_helmet_by_moto.setdefault(idx, []).append(coords)

                        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2),
                                      RED, style.box)
                        # tag stuck to ITS head (all other heads stay clear)
                        draw_tag(annotated, style, "NO HELMET",
                                 (hx1, hy1, hx2, hy2), WHITE, bg=(0, 0, 180),
                                 placed=placed)

                    elif det["cls"] == self.helmet_class:
                        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2),
                                      (0, 200, 0), style.thin)

                    elif det["cls"] == UNKNOWN_HELMET:
                        # head not found by the model: neutral yellow marker
                        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2),
                                      (0, 215, 255), style.thin)
                        draw_tag(annotated, style, "HELMET?",
                                 (hx1, hy1, hx2, hy2), (0, 0, 0),
                                 bg=(0, 215, 255), placed=placed)

                # ---------- Finished background plate reads ----------
                collect_plate_results()

                # ---------- Per-motorcycle logic ----------
                frame_violations = []
                any_triple = False

                for idx, moto in enumerate(motorcycles):

                    x1, y1, x2, y2 = map(int, moto["box"])
                    track_id = moto["track_id"]

                    count = rider_counts.get(idx, 0)
                    status = riding_status.get(idx, "OK")

                    has_no_helmet = idx in no_helmet_by_moto
                    is_triple = status == "TRIPLE RIDING"

                    if is_triple:
                        any_triple = True
                        triple_ids.add(track_id)
                        frame_violations.append((track_id, "triple_riding"))

                    if has_no_helmet:
                        helmet_ids.add(track_id)
                        frame_violations.append((track_id, "no_helmet"))

                    # ----- Number plate (only for violators) -----
                    is_violator = track_id in helmet_ids or track_id in triple_ids
                    wants_plate = is_violator or (is_image and cfg.READ_PLATES_IN_PHOTOS)

                    if (plate_worker is not None and wants_plate
                            and not plate_worker.is_pending(track_id)
                            and plate_tracker.should_try(track_id, 0 if is_image else frame_count)):

                        if is_image:
                            # A photo is one frame: read the plate right now, so
                            # it is on the annotated image and in the log.
                            plate_tracker.record_attempt(track_id)

                            try:
                                store_plate(track_id,
                                            self.plate_reader.read(frame, moto["box"]))
                            except Exception as error:
                                log.warning("Plate reading failed: %s", error)

                        # Video: runs in a background thread, playback continues.
                        elif plate_worker.submit(track_id, frame, moto["box"]):
                            plate_tracker.record_attempt(track_id)

                    best_plate = plate_tracker.best(track_id)

                    # ----- Draw motorcycle -----
                    violating = is_triple or has_no_helmet
                    color = RED if violating else GREEN

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, style.box)

                    if cfg.COMPACT_LABELS and not violating:
                        label = f"#{track_id} R:{count}"
                        scale = style.label_scale * 0.8
                    else:
                        # two short lines instead of one very long one
                        label = f"ID {track_id} | RIDERS: {count}"
                        problems = []

                        if is_triple:
                            problems.append("TRIPLE RIDING")

                        if has_no_helmet:
                            problems.append("NO HELMET")

                        if problems:
                            label += "\n" + " | ".join(problems)

                        scale = None

                    if best_plate:
                        label += f"\nPLATE: {best_plate['text']}"

                    # tucked inside the bike box, bottom-left (clear of faces)
                    draw_label(annotated, style, label, x1 + 4, y2 - 4,
                               font_scale=scale, placed=placed, anchor="bottom")

                # ---------- Warning banners ----------
                warning_y = style.warning_start

                if is_image:
                    banners = []

                    if no_helmet_by_moto:
                        banners.append("NO HELMET VIOLATION")

                    if any_triple:
                        banners.append("TRIPLE RIDING DETECTED")

                    header = make_header(width, style, banners)

                    if header is not None:
                        annotated = np.vstack([header, annotated])
                else:
                    if no_helmet_by_moto:
                        draw_warning(annotated, style, "NO HELMET VIOLATION",
                                     warning_y)
                        warning_y += style.warning_line

                    if any_triple:
                        draw_warning(annotated, style, "TRIPLE RIDING DETECTED",
                                     warning_y)

                # ---------- Evidence + database ----------
                due = [
                    key for key in frame_violations
                    if key not in last_evidence_time
                    or video_time - last_evidence_time[key] >= cfg.COOLDOWN_SECONDS
                ]

                if due:

                    evidence_path = os.path.join(
                        self.evidence_folder,
                        f"violation_{run_id}_{frame_count:06d}.jpg",
                    )

                    if cv2.imwrite(evidence_path, annotated):

                        evidence_count += 1

                        for track_id, vtype in due:

                            last_evidence_time[(track_id, vtype)] = video_time
                            plate = plate_tracker.best(track_id)

                            # INSERT OR IGNORE: keeps the first evidence image
                            self.db.log_violation(
                                run_id, video_name, track_id, vtype,
                                video_time, evidence_path,
                                plate["text"] if plate else None,
                                plate["conf"] if plate else None,
                                plate_images.get(track_id),
                            )

                        log.info(
                            "Evidence saved (%.1fs): %s", video_time,
                            ", ".join(f"ID {t}:{v}" for t, v in due),
                        )

                if cfg.PROGRESS_EVERY and frame_count % cfg.PROGRESS_EVERY == 0:
                    elapsed = time.perf_counter() - t_start
                    log.info(
                        "Frame %s/%s | %.1f fps overall | tracking %.0f ms, "
                        "people+helmets %.0f ms per frame",
                        frame_count, total_frames or "live",
                        frame_count / elapsed,
                        1000 * t_track / frame_count,
                        1000 * t_analyze / frame_count,
                    )

                if display_thread is not None and smooth_overlay:
                    changed = np.any(annotated != frame, axis=2).astype(np.uint8)
                    small_mask = cv2.resize(
                        changed, (disp_w, disp_h), interpolation=cv2.INTER_AREA
                    ) > 0
                    small_pixels = cv2.resize(
                        annotated, (disp_w, disp_h), interpolation=cv2.INTER_AREA
                    )

                    with overlay_lock:
                        overlay["data"] = (small_mask[:, :, None], small_pixels)

                elif display_callback is not None:
                    # boxes and picture come from the SAME frame: always aligned
                    try:
                        display_callback(cv2.resize(
                            annotated, (disp_w, disp_h),
                            interpolation=cv2.INTER_AREA))
                        last_push["t"] = time.perf_counter()
                    except Exception as error:  # never let display kill the run
                        log.warning("Display error: %s", error)

                if writer is not None:
                    writer.write(annotated)

                # ---------- Output ----------
                if frame_callback and frame_count % callback_every == 0:
                    frame_callback(annotated, make_stats())

                if show_window:
                    scale = min(1280 / width, 720 / height)
                    preview = cv2.resize(
                        annotated, (int(width * scale), int(height * scale))
                    )

                    try:
                        cv2.imshow("Two-Wheeler Safety", preview)
                        key = cv2.waitKey(1) & 0xFF
                    except cv2.error:
                        log.warning(
                            "No display available (headless OpenCV build or "
                            "no screen) - continuing without a preview window."
                        )
                        show_window = False
                        key = 0

                    if key == ord("q"):
                        break

                    # Never play faster than real time: keeps preview smooth
                    if pace_to_fps and not wall_clock:
                        wait = (t_start + frame_count / fps) - time.perf_counter()

                        if wait > 0:
                            time.sleep(wait)

        finally:
            display_stop.set()

            if display_thread is not None and display_thread.is_alive():
                display_thread.join(timeout=1.0)

            cap.release()

            if writer is not None:
                writer.release()

            if plate_worker is not None:
                plate_worker.close()
                collect_plate_results()  # reads that finished during shutdown

            if show_window:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

        stats = make_stats()
        stats["output_video"] = (
            output_video_path
            if (save_video and not is_image and not paced_file) else None
        )
        stats["output_image"] = None

        if is_image and annotated is not None:
            image_path = os.path.join(self.image_folder, f"annotated_{run_id}.jpg")

            if cv2.imwrite(image_path, annotated):
                stats["output_image"] = image_path
        stats["plates"] = {
            tid: res["text"] for tid, res in plate_tracker.plates.items()
        }

        if frame_callback and annotated is not None:
            frame_callback(annotated, stats)

        return stats