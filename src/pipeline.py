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
    match_no_helmet_to_motorcycle,
    person_seat_point,
    smoothed_count,
)
from capture import IMAGE_EXTENSIONS, ImageCapture, ThreadedCapture
from geometry import iou
from head_check import UNKNOWN_HEAD, head_crop_detections, verify_decision
from display import OverlayDisplay
from drawing import (AMBER, BLUE, DARK, DARK_RED, GREEN, RED, WHITE, LabelPlacer,
                     Style)
from plate_reader import AsyncPlateReader, PlateReader, PlateTracker
from rider_association import (
    head_point,
    associate_riders,
    count_riders,
    dedupe_people,
    detect_triple_riding,
    head_from_keypoints,
)
from rider_tracking import MotorcycleTracker
from settings import model_path, resolve_path
from violation_db import ViolationDB

log = logging.getLogger(__name__)

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
        self._photo_pose = None      # stronger pose model, loaded on first photo

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
        }

    def _helmet_conf(self):
        """Photos are analysed once, so they can afford a lower (more sensitive) threshold."""
        if self.photo_mode:
            return min(cfg.HELMET_CONF, cfg.PHOTO_HELMET_CONF)

        return cfg.HELMET_CONF

    def _active_pose(self):
        """Photos use the stronger pose model (better with hidden pillions); videos stay fast."""
        if not self.photo_mode or not cfg.POSE_MODEL_PHOTO:
            return self.pose_model

        if self._photo_pose is None:
            try:
                self._photo_pose = YOLO(model_path(cfg.POSE_MODEL_PHOTO))
            except Exception as error:      # offline / download failed: keep going
                log.warning("Could not load %s (%s); using the standard pose model.",
                            cfg.POSE_MODEL_PHOTO, error)
                self._photo_pose = self.pose_model

        return self._photo_pose

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
        result = self.helmet_model(frame, conf=self._helmet_conf(), verbose=False)[0]

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

        pose_results = self._active_pose()(
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
            crops, conf=self._helmet_conf(), imgsz=imgsz, half=USE_GPU, verbose=False
        )

        for result, (ox, oy) in zip(helmet_results, offsets):
            for b in result.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().tolist()
                helmets.append({
                    "box": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
                    "conf": float(b.conf[0]),
                    "cls": int(b.cls[0]),
                })

        people = dedupe_detections(dedupe_people(people), cfg.PERSON_DUP_IOU)
        return people, dedupe_detections(helmets)

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

    def _recovery_imgsz(self):
        return cfg.RECOVERY_IMGSZ_PHOTO if self.photo_mode else cfg.RECOVERY_IMGSZ

    def _refine_heads(self, frame, people, rider_boxes, owners, claims, due):
        """
        Second look at each rider's head on a zoomed crop (see head_check.py):
          * a rider with NO head yet -> try to recover one (or, if
            INFER_NO_HELMET, report "no helmet");
          * a doubtful "With Helmet" -> override it when the zoomed crop
            clearly says "Without Helmet".
        Only bikes analysed this frame (``due``) are refined.
        """
        person_by_box = {tuple(p["box"]): p for p in people}
        claim_by_body = {tuple(body): det for body, det in claims}
        owners = list(owners)

        for bike_index in due:
            for body in rider_boxes.get(bike_index, []):
                person = person_by_box.get(tuple(body))

                if person is None:
                    continue

                det = claim_by_body.get(tuple(body))

                if det is None:
                    found = head_crop_detections(
                        self.helmet_model, frame, person,
                        cfg.RECOVERY_CONF, self._recovery_imgsz())

                    if found:
                        best = max(found, key=lambda d: d["conf"])

                        # not the same head as one already counted
                        if all(iou(best["box"], o[0]["box"]) < 0.5 for o in owners):
                            owners.append((best, bike_index, False))
                    elif cfg.INFER_NO_HELMET or self.photo_mode:
                        hx, hy = head_point(person)
                        r = max((body[2] - body[0]) * 0.20, (body[3] - body[1]) * 0.07, 10)

                        # INFER_NO_HELMET on  -> a real "no helmet" violation.
                        # Otherwise (photos)  -> an amber "HELMET NOT FOUND" marker
                        # for a human to check; it is NOT counted as a violation.
                        owners.append((
                            {"box": [hx - r, hy - r, hx + r, hy + r],
                             "cls": (self.no_helmet_class if cfg.INFER_NO_HELMET
                                     else UNKNOWN_HEAD),
                             "conf": 0.0},
                            bike_index, False))

                elif (cfg.VERIFY_ENABLED
                        and det["cls"] == self.helmet_class
                        and det["conf"] < cfg.VERIFY_BELOW):
                    verdict = verify_decision(
                        head_crop_detections(
                            self.helmet_model, frame, person,
                            cfg.RECOVERY_CONF, self._recovery_imgsz()),
                        self.helmet_class, self.no_helmet_class,
                        cfg.VERIFY_WITHOUT_MIN)

                    if verdict is not None:
                        owners = [(verdict, o[1], o[2]) if o[0] is det else o
                                  for o in owners]

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
                helmet_dets, motorcycles, rider_boxes, other_boxes, claims, people
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
            realtime=False, display_callback=None):
        """
        Process a video file, or a live camera/stream when `source` is a
        camera index (int) or `live=True`. Returns a summary dict.

        live        read the newest frame only (no lag), use wall-clock time
        save_video  write the annotated output video
        pace_to_fps with a preview window, play files at their real speed
                    (never faster) so playback looks smooth
        realtime    play a video FILE at its real speed and let detection skip
                    frames to keep up (like a live camera)
        display_callback(frame_bgr)  if given, a separate thread calls it with
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

        style = Style(width, height)

        if live:
            video_name = f"camera_{source}" if isinstance(source, int) else "stream"
        else:
            video_name = os.path.basename(source)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{stamp}_{os.path.splitext(video_name)[0]}"

        output_video_path = os.path.join(self.video_folder, f"tracked_{run_id}.mp4")

        writer = None

        if save_video and not is_image and not paced_file:
            # Live frames are dropped when detection is slow, so record at a
            # modest fixed rate instead of the camera's nominal fps.
            writer = cv2.VideoWriter(
                output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                15 if live else fps, (width, height),
            )

            if not writer.isOpened():
                cap.release()
                raise RuntimeError("Could not create output video.")

        # Fresh detector + tracker per run so track IDs never carry over
        # from a previous video.
        traffic_model = YOLO(self.traffic_path)
        tracker = MotorcycleTracker(traffic_model, confidence=cfg.MOTORCYCLE_CONF)
        plate_tracker = PlateTracker()

        # One background worker per run (closed again at the end)
        plate_worker = (
            AsyncPlateReader(self.plate_reader)
            if self.plate_reader is not None else None
        )

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
                "frame": cap.frame_index if paced_file else frame_count,
                "total_frames": total_frames,
                "detector_fps": frame_count / max(1e-6, time.perf_counter() - t_start),
                "motorcycles": len(seen_ids),
                "helmet_violations": len(helmet_ids),
                "triple_riding": len(triple_ids),
                "plates_read": len(plate_tracker.plates),
                "evidence_saved": evidence_count,
            }

        # ---------- Smooth display, decoupled from detection ----------
        display = None

        disp_w = min(width, 960)
        disp_h = max(1, int(height * disp_w / width))

        if display_callback is not None and realtime:
            display = OverlayDisplay(
                cap, display_callback, (disp_w, disp_h), fps,
                aligned=paced_file and cfg.ALIGNED_FILE_DISPLAY,
                min_delay=cfg.DISPLAY_MIN_DELAY, max_delay=cfg.DISPLAY_MAX_DELAY,
            )

        if paced_file:
            cap.start()          # the file starts playing now, models are loaded

        if display is not None:
            display.start()

        try:
            while True:

                if should_stop is not None and should_stop():
                    break

                ret, frame = cap.read()

                if not ret:
                    break

                t_frame_read = time.perf_counter()
                frame_seq = getattr(cap, "last_read_seq", 0)
                frame_count += 1
                if live:
                    video_time = time.perf_counter() - t_start
                elif paced_file:
                    video_time = cap.frame_index / fps
                else:
                    video_time = frame_count / fps

                # ---------- Motorcycle tracking ----------
                t0 = time.perf_counter()

                motorcycles = [
                    {"box": m["box"], "conf": m["conf"], "track_id": m["track_id"]}
                    for m in tracker.update(frame)
                ]

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
                placer = LabelPlacer(annotated.shape, style)

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

                # Only heads that belong to a rider are drawn / counted;
                # pedestrians' heads are ignored.
                # Keep label chips off every head box
                for det, _idx, _extra in head_owners:
                    placer.taken.append(tuple(int(v) for v in det["box"]))

                for det, idx, _is_extra in head_owners:

                    coords = det["box"]
                    hx1, hy1, hx2, hy2 = map(int, coords)

                    if det["cls"] == self.no_helmet_class:

                        no_helmet_by_moto.setdefault(idx, []).append(coords)

                        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2),
                                      RED, style.box)
                        placer.draw(annotated, "NO HELMET", hx1, hy1,
                                    anchor="above", bg=DARK_RED,
                                    scale=style.tag_scale)

                    elif det["cls"] == self.helmet_class:
                        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2),
                                      (0, 200, 0), style.thin)

                    elif det["cls"] == UNKNOWN_HEAD:
                        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2),
                                      AMBER, style.thin)
                        placer.draw(annotated, "HELMET NOT FOUND", hx1, hy1,
                                    anchor="above", color=DARK, bg=AMBER,
                                    scale=style.tag_scale)

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

                    if (plate_worker is not None and is_violator
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

                    plural = "rider" if count == 1 else "riders"
                    lines = [(f"#{track_id}  {count} {plural}", WHITE,
                              DARK_RED if violating else DARK)]

                    if is_triple:
                        lines.append(("TRIPLE RIDING", WHITE, DARK_RED))

                    if has_no_helmet:
                        lines.append(("NO HELMET", WHITE, DARK_RED))

                    if best_plate:
                        lines.append((f"PLATE {best_plate['text']}", DARK, (0, 220, 255)))

                    # Labels go BELOW the bike (over the road), not above it where
                    # they would cover the riders' heads.
                    placer.stack(annotated, lines, x1, y2, anchor="below")

                # ---------- Warning banners (small, top-left) ----------
                banner = []

                if no_helmet_by_moto:
                    banner.append(("NO HELMET VIOLATION", WHITE, DARK_RED))

                if any_triple:
                    banner.append(("TRIPLE RIDING DETECTED", WHITE, DARK_RED))

                if banner:
                    placer.stack(annotated, banner, style.gap, 0, anchor="below",
                                 scale=style.warning_scale)

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

                if display is not None:
                    changed = np.any(annotated != frame, axis=2).astype(np.uint8)
                    small_mask = cv2.resize(
                        changed, (disp_w, disp_h), interpolation=cv2.INTER_AREA
                    ) > 0
                    small_pixels = cv2.resize(
                        annotated, (disp_w, disp_h), interpolation=cv2.INTER_AREA
                    )

                    display.push_overlay(
                        frame_seq, small_mask[:, :, None], small_pixels)
                    display.report_latency(time.perf_counter() - t_frame_read)

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
            if display is not None:
                display.stop()

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