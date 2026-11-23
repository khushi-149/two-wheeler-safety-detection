"""
Two-wheeler safety pipeline:
  motorcycle tracking -> rider counting -> helmet check -> triple riding
  -> number plate OCR -> evidence images -> SQLite log.

v2: riders and helmets are analysed on a crop around EACH tracked motorcycle
(USE_CROP_ANALYSIS), so small/distant riders in wide traffic scenes are
detected. Set USE_CROP_ANALYSIS = False for the original full-frame method.

Used by detect_video_full.py (command line) and dashboard.py (Streamlit).
"""

import os
import threading
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

from rider_association import (
    dedupe_people,
    associate_riders,
    count_riders,
    detect_triple_riding,
    head_from_keypoints,
)
from rider_tracking import MotorcycleTracker
from violation_db import ViolationDB
from plate_reader import PlateReader, PlateTracker, AsyncPlateReader


# -------------------- Settings --------------------

try:
    import torch
    USE_GPU = torch.cuda.is_available()
except ImportError:
    USE_GPU = False

FONT = cv2.FONT_HERSHEY_SIMPLEX

MOTORCYCLE_CONF = 0.25
HELMET_CONF = 0.40
POSE_CONF = 0.30
COOLDOWN_SECONDS = 3

PERSON_CLASS = 0
HELMET_CLASS = 0       # "with helmet"  (check your model's class order)
NO_HELMET_CLASS = 1    # "without helmet"

# Re-analyse each bike's riders/helmets only every N frames (staggered across
# bikes so the cost per frame stays flat). Tracking still runs every frame, so
# boxes move smoothly; heads are shifted along with their bike in between.
# 1 = analyse every bike every frame (slowest, most exact).
ANALYZE_EVERY = 3
CACHE_MAX_AGE = 90   # forget a bike's cached analysis after this many frames unseen

# Input size for the per-bike crops. Crops are small, so a small size is
# much faster than the default 640 with little accuracy loss.
CROP_IMGSZ = 384

# Without models/plate_detector.pt, plate reading falls back to blind OCR,
# which is slow and rarely works. Keep this False unless you want to try it.
ALLOW_OCR_FALLBACK = False

# Print timing every N frames (set 0 to turn off)
PROGRESS_EVERY = 30

# Analyse a crop around each motorcycle (recommended for wide scenes)
USE_CROP_ANALYSIS = True

# Crop region around a motorcycle box, as a fraction of the box size.
# Riders sit ABOVE the bike box, so the top padding is large.
CROP_PAD_SIDE = 0.25
CROP_PAD_TOP = 1.0
CROP_PAD_BOTTOM = 0.10

# A person counts as a rider if at least this share of the person box
# overlaps the motorcycle box.
MIN_RIDER_OVERLAP = 0.10

# Head ownership (crop mode).
# A head belongs to a rider if its centre is in the top HEAD_ZONE of that
# rider's body box (plus a small allowance above the box).
HEAD_ZONE = 0.60
HEAD_TOP_ALLOWANCE = 0.15

# A head not inside ANY person box (e.g. a pillion hidden behind the rider)
# is only accepted if it sits in this tight area around a bike, as a
# fraction of the bike box. Pedestrians farther away are ignored.
HIDDEN_SIDE = 0.10
HIDDEN_TOP = 0.75
HIDDEN_BELOW = 0.50

# Head -> motorcycle matching used by the full-frame fallback mode only
EXPAND_TOP = 1.0
EXPAND_SIDE = 0.15
EXPAND_BOTTOM = 0.0

# Bikes without a violation get a short label ("#12 R:2")
COMPACT_LABELS = True

# Draw thin blue boxes around detected people (to debug rider counting)
DEBUG_DRAW_PEOPLE = False

RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
BLUE = (255, 120, 0)
DARK = (25, 25, 25)


class Style:
    """Annotation sizes that scale with the video height."""

    def __init__(self, height):
        s = max(0.6, height / 720)
        self.label_scale = 0.7 * s
        self.label_thickness = max(2, int(2 * s))
        self.warning_scale = 1.0 * s
        self.warning_thickness = max(2, int(3 * s))
        self.box = max(2, int(3 * s))
        self.thin = max(1, int(2 * s))
        self.pad = max(6, int(6 * s))
        self.warning_line = int(65 * s)
        self.warning_start = int(55 * s)


# -------------------- Drawing helpers --------------------

def draw_label(frame, style, text, x, y, color=WHITE, bg=DARK,
               font_scale=None, thickness=None):

    font_scale = font_scale or style.label_scale
    thickness = thickness or style.label_thickness
    pad = style.pad

    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)

    x = int(max(0, min(x, frame.shape[1] - tw - 2 * pad)))
    y = int(max(th + 2 * pad, y))

    cv2.rectangle(frame, (x, y - th - pad),
                  (x + tw + 2 * pad, y + baseline + pad // 2), bg, -1)

    cv2.putText(frame, text, (x + pad, y), FONT, font_scale,
                color, thickness, cv2.LINE_AA)


def draw_warning(frame, style, text, y):
    draw_label(frame, style, text, 20, y, WHITE, bg=(0, 0, 200),
               font_scale=style.warning_scale,
               thickness=style.warning_thickness)


# -------------------- Geometry helpers --------------------

def box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def expanded_box(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - EXPAND_SIDE * w, y1 - EXPAND_TOP * h,
            x2 + EXPAND_SIDE * w, y2 + EXPAND_BOTTOM * h)


def point_in_box(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def match_no_helmet_to_motorcycle(no_helmet_box, motorcycles):
    center = box_center(no_helmet_box)
    best_index, best_dist = None, float("inf")

    for index, moto in enumerate(motorcycles):
        if not point_in_box(center, expanded_box(moto["box"])):
            continue

        mx, my = box_center(moto["box"])
        dist = (mx - center[0]) ** 2 + (my - center[1]) ** 2

        if dist < best_dist:
            best_dist, best_index = dist, index

    return best_index


def bike_region(box, frame_w, frame_h):
    """Crop area around a motorcycle, including the space above it."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    return (
        max(0, int(x1 - CROP_PAD_SIDE * bw)),
        max(0, int(y1 - CROP_PAD_TOP * bh)),
        min(frame_w, int(x2 + CROP_PAD_SIDE * bw)),
        min(frame_h, int(y2 + CROP_PAD_BOTTOM * bh)),
    )


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe_detections(dets, iou_threshold=0.5):
    """Greedy dedupe: crops of neighbouring bikes can see the same head twice."""
    kept = []

    for det in sorted(dets, key=lambda d: d["conf"], reverse=True):
        if all(box_iou(det["box"], k["box"]) < iou_threshold for k in kept):
            kept.append(det)

    return kept


def assign_people_to_bikes(people, motorcycles):
    """
    Count riders per motorcycle index. A person belongs to the bike whose
    box overlaps most of their body, if their centre is horizontally
    within that bike.

    Returns (counts, rider_boxes, other_boxes):
      counts       {bike index: number of riders}
      rider_boxes  {bike index: [person boxes that ride it]}
      other_boxes  [person boxes not riding any tracked bike (pedestrians)]
    """
    counts = {i: 0 for i in range(len(motorcycles))}
    rider_boxes = {i: [] for i in range(len(motorcycles))}
    other_boxes = []

    for person in people:
        px1, py1, px2, py2 = person["box"]
        pcx = (px1 + px2) / 2.0
        parea = max(1.0, (px2 - px1) * (py2 - py1))

        best, best_score = None, 0.0

        for idx, moto in enumerate(motorcycles):
            bx1, by1, bx2, by2 = moto["box"]
            bw = bx2 - bx1

            if not (bx1 - 0.15 * bw <= pcx <= bx2 + 0.15 * bw):
                continue

            ix1, iy1 = max(px1, bx1), max(py1, by1)
            ix2, iy2 = min(px2, bx2), min(py2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            score = inter / parea

            if score > best_score:
                best, best_score = idx, score

        if best is not None and best_score >= MIN_RIDER_OVERLAP:
            counts[best] += 1
            rider_boxes[best].append(person["box"])
        else:
            other_boxes.append(person["box"])

    return counts, rider_boxes, other_boxes


def _head_in_person(center, person_box):
    """True if a head centre lies in the upper part of a person box."""
    px1, py1, px2, py2 = person_box
    ph = py2 - py1

    return (
        px1 <= center[0] <= px2
        and py1 - HEAD_TOP_ALLOWANCE * ph <= center[1] <= py1 + HEAD_ZONE * ph
    )


def assign_heads_to_bikes(helmet_dets, motorcycles, rider_boxes, other_boxes):
    """
    Decide which bike (if any) each detected head belongs to.

    1. Each rider's body claims its single best head (nearest the top-centre
       of the body box) -> that bike, rider already counted.
    2. Other heads inside a non-rider's body box (a pedestrian) -> ignored.
    3. Other heads close above a bike (e.g. a pillion hidden behind the
       rider that pose detection missed) -> that bike, and each counts as
       an EXTRA rider.
    4. Anything else -> ignored.

    Returns a list of (detection, bike index, is_extra_rider).
    """
    owners = []
    claimed = set()

    # 1. Each rider's body claims ONE head: the one nearest the top-centre
    #    of the body box. Any other head in that zone is a second person.
    for idx, boxes in rider_boxes.items():
        for body in boxes:

            candidates = [
                (i, det) for i, det in enumerate(helmet_dets)
                if i not in claimed
                and _head_in_person(box_center(det["box"]), body)
            ]

            if not candidates:
                continue

            target_x = (body[0] + body[2]) / 2.0
            target_y = body[1] + 0.10 * (body[3] - body[1])

            def distance(item):
                cx, cy = box_center(item[1]["box"])
                return (cx - target_x) ** 2 + (cy - target_y) ** 2

            i, det = min(candidates, key=distance)
            claimed.add(i)
            owners.append((det, idx, False))

    # 2-4. Heads nobody claimed
    for i, det in enumerate(helmet_dets):

        if i in claimed:
            continue

        center = box_center(det["box"])

        in_rider_zone = any(
            _head_in_person(center, b)
            for boxes in rider_boxes.values() for b in boxes
        )

        # head of a pedestrian -> ignore
        if not in_rider_zone and any(
            _head_in_person(center, b) for b in other_boxes
        ):
            continue

        # unexplained head close above a bike -> hidden rider (e.g. pillion)
        best_idx, best_dist = None, float("inf")

        for idx, moto in enumerate(motorcycles):
            x1, y1, x2, y2 = moto["box"]
            bw, bh = x2 - x1, y2 - y1

            inside = (
                x1 - HIDDEN_SIDE * bw <= center[0] <= x2 + HIDDEN_SIDE * bw
                and y1 - HIDDEN_TOP * bh <= center[1] <= y1 + HIDDEN_BELOW * bh
            )

            if not inside:
                continue

            mx, my = box_center(moto["box"])
            dist = (mx - center[0]) ** 2 + (my - center[1]) ** 2

            if dist < best_dist:
                best_dist, best_idx = dist, idx

        if best_idx is not None:
            owners.append((det, best_idx, True))

    return owners


# -------------------- Photo input --------------------

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class ImageCapture:
    """Presents one photo as a one-frame 'video' so the same pipeline runs on it."""

    def __init__(self, path):
        self.image = cv2.imread(path)
        self._done = False

    def isOpened(self):
        return self.image is not None

    def get(self, prop):
        if self.image is None:
            return 0

        h, w = self.image.shape[:2]

        return {
            cv2.CAP_PROP_FRAME_WIDTH: w,
            cv2.CAP_PROP_FRAME_HEIGHT: h,
            cv2.CAP_PROP_FRAME_COUNT: 1,
        }.get(prop, 0)

    def read(self):
        if self._done or self.image is None:
            return False, None

        self._done = True
        return True, self.image.copy()

    def release(self):
        pass


# -------------------- Live camera capture --------------------

class ThreadedCapture:
    """
    Reads a camera / stream in a background thread and always hands out the
    NEWEST frame. Stale frames are dropped, so the view never lags behind
    even when detection is slower than the camera. Mimics the parts of
    cv2.VideoCapture that the pipeline uses.
    """

    def __init__(self, source, width=1280, height=720):

        if isinstance(source, str) and source.lower().startswith("rtsp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        if isinstance(source, int) and os.name == "nt":
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # faster on Windows
        else:
            self.cap = cv2.VideoCapture(source)

        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._last_seq = 0
        self._running = self.cap.isOpened()

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ok, frame = self.cap.read()

            if not ok:
                time.sleep(0.01)
                continue

            with self._lock:
                self._frame = frame
                self._seq += 1

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)

    def read(self, timeout=3.0):
        """Wait for a NEW frame. Returns (False, None) if the camera stalls."""
        deadline = time.time() + timeout

        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._seq != self._last_seq:
                    self._last_seq = self._seq
                    return True, self._frame.copy()

            time.sleep(0.003)

        return False, None

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# -------------------- Pipeline --------------------

class ViolationPipeline:

    def __init__(self,
                 pose_path="yolo11n-pose.pt",
                 helmet_path="models/helmet_detector.pt",
                 plate_model_path="models/plate_detector.pt",
                 use_plates=True,
                 db_path="outputs/violations.db",
                 evidence_folder="outputs/evidence",
                 video_folder="outputs/videos",
                 plate_folder="outputs/plates",
                 image_folder="outputs/images"):

        self.pose_model = YOLO(pose_path)
        self.helmet_model = YOLO(helmet_path)

        self.db = ViolationDB(db_path)

        self.evidence_folder = evidence_folder
        self.video_folder = video_folder
        self.plate_folder = plate_folder
        self.image_folder = image_folder

        for folder in (evidence_folder, video_folder, plate_folder, image_folder):
            os.makedirs(folder, exist_ok=True)

        self.plate_reader = None

        try:
            import torch
            device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU only"
        except ImportError:
            device = "unknown"

        print(f"Compute device: {device}")

        if use_plates:
            if not (os.path.exists(plate_model_path) or ALLOW_OCR_FALLBACK):
                print(f"{plate_model_path} not found -> plate reading is OFF "
                      "(it would be slow and unreliable without a plate "
                      "detector). Helmet and triple-riding checks still run.")
            else:
                try:
                    self.plate_reader = PlateReader(plate_model_path)
                except ImportError:
                    print("easyocr not installed - plate reading disabled. "
                          "Run: pip install easyocr")

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
            "conf": float(box.conf[0]),
        }

    def _detect_people(self, frame):
        """Original method: pose on the whole frame."""
        people = []
        pose = self.pose_model(frame, conf=POSE_CONF, verbose=False)[0]

        if pose.boxes is None or pose.keypoints is None:
            return people

        for index, box in enumerate(pose.boxes):
            if int(box.cls[0]) == PERSON_CLASS:
                people.append(self._person_from_result(pose, index, box))

        return dedupe_people(people)

    def _detect_helmets_fullframe(self, frame):
        result = self.helmet_model(frame, conf=HELMET_CONF, verbose=False)[0]

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

        pose_results = self.pose_model(
            crops, conf=POSE_CONF, imgsz=CROP_IMGSZ, half=USE_GPU, verbose=False
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
            crops, conf=HELMET_CONF, imgsz=CROP_IMGSZ, half=USE_GPU, verbose=False
        )

        for result, (ox, oy) in zip(helmet_results, offsets):
            for b in result.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().tolist()
                helmets.append({
                    "box": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
                    "conf": float(b.conf[0]),
                    "cls": int(b.cls[0]),
                })

        return dedupe_people(people), dedupe_detections(helmets)

    # ---------------------------------------------------------------

    def _analyze_cached(self, frame, motorcycles, cache, frame_count):
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
            or (m["track_id"] + frame_count) % ANALYZE_EVERY == 0
        ]

        people = []

        if due:
            due_bikes = [motorcycles[i] for i in due]

            people, helmet_dets = self._analyze_crops(frame, due_bikes)

            counts, rider_boxes, other_boxes = assign_people_to_bikes(
                people, due_bikes
            )
            owners = assign_heads_to_bikes(
                helmet_dets, due_bikes, rider_boxes, other_boxes
            )

            for j, i in enumerate(due):
                bike = motorcycles[i]

                heads = [
                    {"box": det["box"], "cls": det["cls"], "extra": extra}
                    for det, owner_j, extra in owners if owner_j == j
                ]

                cache[bike["track_id"]] = {
                    "anchor": (bike["box"][0], bike["box"][1]),
                    "count": counts[j] + sum(1 for h in heads if h["extra"]),
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
                         if frame_count - e["seen"] > CACHE_MAX_AGE]:
            del cache[track_id]

        return people, rider_counts, head_owners

    def run(self, source, show_window=False,
            frame_callback=None, callback_every=2, should_stop=None,
            live=False, save_video=True, pace_to_fps=True):
        """
        Process a video file, or a live camera/stream when `source` is a
        camera index (int) or `live=True`. Returns a summary dict.

        live        read the newest frame only (no lag), use wall-clock time
        save_video  write the annotated output video
        pace_to_fps with a preview window, play files at their real speed
                    (never faster) so playback looks smooth

        frame_callback(annotated_frame, stats) is called every
        `callback_every` frames (used by the Streamlit dashboard).
        """

        live = live or isinstance(source, int)

        is_image = (
            not live
            and isinstance(source, str)
            and source.lower().endswith(IMAGE_EXTENSIONS)
        )

        if live:
            cap = ThreadedCapture(source)
        elif is_image:
            cap = ImageCapture(source)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = 0 if live else (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1)

        if fps <= 0 or fps > 120:
            fps = 30 if live else 25

        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Invalid video dimensions.")

        style = Style(height)

        if live:
            video_name = f"camera_{source}" if isinstance(source, int) else "stream"
        else:
            video_name = os.path.basename(source)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{stamp}_{os.path.splitext(video_name)[0]}"

        output_video_path = os.path.join(self.video_folder, f"tracked_{run_id}.mp4")

        writer = None

        if save_video and not is_image:
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
        traffic_model = YOLO("yolo11n.pt")
        tracker = MotorcycleTracker(traffic_model, confidence=MOTORCYCLE_CONF)
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

        def collect_plate_results():
            """Store finished background OCR reads (best per track) + DB."""
            if plate_worker is None:
                return

            for track_id, result in plate_worker.poll():

                if result is None:
                    continue

                if plate_tracker.update(track_id, result):

                    plate_path = os.path.join(
                        self.plate_folder, f"{run_id}_ID{track_id}.jpg"
                    )
                    cv2.imwrite(plate_path, result["crop"])
                    plate_images[track_id] = plate_path

                    self.db.update_plate(
                        run_id, track_id, result["text"],
                        result["conf"], plate_path,
                    )

        def make_stats():
            return {
                "run_id": run_id,
                "frame": frame_count,
                "total_frames": total_frames,
                "motorcycles": len(seen_ids),
                "helmet_violations": len(helmet_ids),
                "triple_riding": len(triple_ids),
                "plates_read": len(plate_tracker.plates),
                "evidence_saved": evidence_count,
            }

        try:
            while True:

                if should_stop is not None and should_stop():
                    break

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1
                video_time = (
                    time.perf_counter() - t_start if live else frame_count / fps
                )

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

                elif USE_CROP_ANALYSIS:
                    people, rider_counts, head_owners = self._analyze_cached(
                        frame, motorcycles, analysis_cache, frame_count
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

                if DEBUG_DRAW_PEOPLE:
                    for person in people:
                        px1, py1, px2, py2 = map(int, person["box"])
                        cv2.rectangle(annotated, (px1, py1), (px2, py2),
                                      BLUE, style.thin)

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
                for det, idx, _is_extra in head_owners:

                    coords = det["box"]
                    hx1, hy1, hx2, hy2 = map(int, coords)

                    if det["cls"] == NO_HELMET_CLASS:

                        no_helmet_by_moto.setdefault(idx, []).append(coords)

                        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2),
                                      RED, style.box)
                        draw_label(annotated, style, "NO HELMET",
                                   hx1, hy1 - 8, WHITE, bg=(0, 0, 180),
                                   font_scale=style.label_scale * 0.8)

                    elif det["cls"] == HELMET_CLASS:
                        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2),
                                      (0, 200, 0), style.thin)

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
                            and plate_tracker.should_try(track_id, frame_count)):

                        # Runs in a background thread: the video keeps playing.
                        if plate_worker.submit(track_id, frame, moto["box"]):
                            plate_tracker.record_attempt(track_id)

                    best_plate = plate_tracker.best(track_id)

                    # ----- Draw motorcycle -----
                    violating = is_triple or has_no_helmet
                    color = RED if violating else GREEN

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, style.box)

                    if COMPACT_LABELS and not violating:
                        label = f"#{track_id} R:{count}"
                        scale = style.label_scale * 0.8
                    else:
                        label = f"ID: {track_id} | RIDERS: {count} | {status}"

                        if has_no_helmet:
                            label += " | NO HELMET"

                        scale = None

                    if best_plate:
                        label += f" | PLATE: {best_plate['text']}"

                    draw_label(annotated, style, label, x1, y1 - 8,
                               font_scale=scale)

                # ---------- Warning banners ----------
                warning_y = style.warning_start

                if no_helmet_by_moto:
                    draw_warning(annotated, style, "NO HELMET VIOLATION", warning_y)
                    warning_y += style.warning_line

                if any_triple:
                    draw_warning(annotated, style, "TRIPLE RIDING DETECTED", warning_y)

                # ---------- Evidence + database ----------
                due = [
                    key for key in frame_violations
                    if key not in last_evidence_time
                    or video_time - last_evidence_time[key] >= COOLDOWN_SECONDS
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

                        print(
                            f"Evidence saved ({video_time:.1f}s): "
                            + ", ".join(f"ID {t}:{v}" for t, v in due)
                        )

                if PROGRESS_EVERY and frame_count % PROGRESS_EVERY == 0:
                    elapsed = time.perf_counter() - t_start
                    print(
                        f"Frame {frame_count}/{total_frames or 'live'} | "
                        f"{frame_count / elapsed:.1f} fps overall | "
                        f"tracking {1000 * t_track / frame_count:.0f} ms, "
                        f"people+helmets {1000 * t_analyze / frame_count:.0f} ms "
                        f"per frame"
                    )

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
                    cv2.imshow("Two-Wheeler Safety", preview)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    # Never play faster than real time: keeps preview smooth
                    if pace_to_fps and not live:
                        wait = (t_start + frame_count / fps) - time.perf_counter()

                        if wait > 0:
                            time.sleep(wait)

        finally:
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
            output_video_path if (save_video and not is_image) else None
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