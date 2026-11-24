"""
Central configuration for the Two-Wheeler Safety project.

All tunable thresholds and every file-system location live here, so nothing
else in the code base needs hard-coded paths or "magic numbers".

Paths are anchored to the project root (the folder that contains ``src/``),
so every script works no matter which directory it is started from.
"""

from __future__ import annotations

import os
from pathlib import Path

# -------------------- Locations --------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

VIDEO_DIR = DATA_DIR / "videos"
VIDEO_UPLOAD_DIR = VIDEO_DIR / "uploads"
IMAGE_DIR = DATA_DIR / "images"
IMAGE_UPLOAD_DIR = IMAGE_DIR / "uploads"
TEST_IMAGE_DIR = DATA_DIR / "test_images"

DB_PATH = OUTPUTS_DIR / "violations.db"
EVIDENCE_DIR = OUTPUTS_DIR / "evidence"
OUTPUT_VIDEO_DIR = OUTPUTS_DIR / "videos"
PLATE_DIR = OUTPUTS_DIR / "plates"
OUTPUT_IMAGE_DIR = OUTPUTS_DIR / "images"

# Model files. Official Ultralytics weights are downloaded automatically the
# first time they are used; the custom helmet / plate weights must be placed
# in ``models/`` by hand (see models/README.md).
TRAFFIC_MODEL = "yolo11n.pt"
POSE_MODEL = "yolo11n-pose.pt"
HELMET_MODEL = "helmet_detector.pt"
PLATE_MODEL = "plate_detector.pt"


def resolve_path(path: str | os.PathLike) -> str:
    """Return ``path`` as an absolute string; relative paths start at the project root."""
    p = Path(path)

    if not p.is_absolute():
        p = PROJECT_ROOT / p

    return str(p)


def model_path(name: str | os.PathLike) -> str:
    """
    Locate a model file.

    Looks in ``models/`` first, then at the given path. If neither exists the
    plain name is returned, which lets Ultralytics download official weights
    (for example ``yolo11n.pt``) automatically.
    """
    candidate = Path(name)

    if candidate.is_absolute():
        return str(candidate)

    in_models = MODELS_DIR / candidate

    if in_models.exists():
        return str(in_models)

    in_root = PROJECT_ROOT / candidate

    if in_root.exists():
        return str(in_root)

    return str(name)


# -------------------- Detection thresholds --------------------

MOTORCYCLE_CONF = 0.25
HELMET_CONF = 0.40
POSE_CONF = 0.30

# COCO class ids used by the YOLO models
COCO_PERSON_CLASS = 0
COCO_MOTORCYCLE_CLASS = 3

# Default class order of the custom helmet model (verified against the model's
# own class names at load time, see ``pipeline.resolve_helmet_classes``).
HELMET_CLASS = 0       # "With Helmet"
NO_HELMET_CLASS = 1    # "Without Helmet"

# -------------------- Evidence --------------------

# Minimum video-seconds before the same violation on the same motorcycle is
# saved again.
COOLDOWN_SECONDS = 3

# -------------------- Speed / smoothing --------------------

# Re-analyse each bike's riders/helmets only every N frames (staggered across
# bikes so the cost per frame stays flat). Tracking still runs every frame, so
# boxes move smoothly; heads are shifted along with their bike in between.
# 1 = analyse every bike every frame (slowest, most exact).
ANALYZE_EVERY = 3

# Forget a bike's cached analysis after this many frames unseen.
CACHE_MAX_AGE = 90

# Rider counts are smoothed: the reported count is the (lower) median of the
# last COUNT_HISTORY analyses, and a count of 3+ is only reported once at
# least MIN_HISTORY_FOR_TRIPLE analyses agree. Single-frame glitches can no
# longer raise a triple-riding violation. (Photos have only one frame, so
# they skip this requirement.)
COUNT_HISTORY = 5
MIN_HISTORY_FOR_TRIPLE = 3

# Input size for the per-bike crops. Bigger = more pixels per head = better
# helmet / veil / turban recognition, but slower. On a 4000 px photo the helmet
# model missed 2 of 3 heads at the default 640 but found all 3 at 1280; the
# crop around a bike is smaller than the full photo, so 640 here is a good
# balance. Use 800-1280 for photos, 384-480 if video is too slow on a CPU.
CROP_IMGSZ = 640

# Photos are analysed only once, so speed does not matter: the crop size is
# raised automatically to the crop's own pixel size, up to this limit.
CROP_IMGSZ_MAX = 1280

# Print timing every N frames (0 = off)
PROGRESS_EVERY = 30

# -------------------- Rider / head assignment --------------------

# Analyse a crop around each motorcycle (recommended for wide scenes).
USE_CROP_ANALYSIS = True

# Crop region around a motorcycle box, as a fraction of the box size.
# Riders sit ABOVE the bike box, so the top padding is large.
CROP_PAD_SIDE = 0.25
CROP_PAD_TOP = 1.0
CROP_PAD_BOTTOM = 0.10

# A person rides a bike when their SEAT (hip point) is inside the bike box,
# widened by these fractions of the box size.
SEAT_MARGIN_SIDE = 0.10
SEAT_MARGIN_TOP = 0.15
SEAT_MARGIN_BOTTOM = 0.10

# Rider validation: a person is NOT a rider if their head sits far above the
# bike, measured against BOTH the bike's height and their own height. This
# rejects mannequins, pedestrians and bystanders standing behind a parked bike.
# (Measured on real photos with the photo tester: riders were at most 0.59
# bike-heights / 0.51 person-heights above the bike box, mannequins beside a
# parked scooter at least 0.94 / 0.64.)
MAX_HEAD_ABOVE_MOTO = 0.80
MAX_HEAD_ABOVE_PERSON = 0.50

# Fallback: a person counts as a rider if at least this share of the person
# box overlaps the motorcycle box.
MIN_RIDER_OVERLAP = 0.10

# Person boxes from overlapping crops with at least this IoU are the same person.
PERSON_DUP_IOU = 0.75

# Head ownership: a head belongs to a rider if its centre is in the top
# HEAD_ZONE of that rider's body box (plus a small allowance above the box).
HEAD_ZONE = 0.60
HEAD_TOP_ALLOWANCE = 0.15

# A head not inside ANY person box (e.g. a pillion hidden behind the rider) is
# only accepted if it sits in this tight area around a bike, as a fraction of
# the bike box. Pedestrians farther away are ignored.
HIDDEN_SIDE = 0.10
HIDDEN_TOP = 0.75
HIDDEN_BELOW = 0.50

# Head -> motorcycle matching used by the full-frame fallback mode only.
EXPAND_TOP = 1.0
EXPAND_SIDE = 0.15
EXPAND_BOTTOM = 0.0

# -------------------- Head re-check --------------------

# Second look on a zoomed crop of the rider's head. It recovers heads the main
# helmet pass missed (small, partly hidden, children) and double-checks a
# doubtful "With Helmet" (hair, turbans, scarves score low).
RECOVERY_CONF = 0.10
RECOVERY_IMGSZ = 320

VERIFY_ENABLED = True
VERIFY_BELOW = 0.75         # re-check a "With Helmet" scoring below this
VERIFY_WITHOUT_MIN = 0.30   # the re-check must say "Without" at least this sure

# A rider whose head is not found even after the re-check: report "no helmet"?
# False (default) = leave them unflagged, so a helmeted rider the model cannot
# see never causes a false violation. True = flag them (the photo tester does).
INFER_NO_HELMET = False

# -------------------- Plates --------------------

# Without models/plate_detector.pt, plate reading falls back to blind OCR,
# which is slow and rarely works. Keep this False unless you want to try it.
ALLOW_OCR_FALLBACK = False

# -------------------- Display --------------------

# Bikes without a violation get a short label ("#12 R:2").
COMPACT_LABELS = True

# Draw thin blue boxes around detected people (to debug rider counting).
DEBUG_DRAW_PEOPLE = False

# -------------------- File types --------------------

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")