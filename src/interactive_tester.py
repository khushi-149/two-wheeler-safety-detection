"""
Interactive photo / video tester (menu driven, shows results in a window).

    python src/interactive_tester.py

Uses a heavier full-frame method with extra helmet re-checks, which suits
single photos. For videos, tracking, evidence and the database use
``detect_video_full.py`` or the dashboard instead.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
from ultralytics import YOLO

import settings as cfg
from geometry import ioa as box_ioa
from geometry import iou as box_iou
from rider_association import (
    associate_riders,
    count_riders,
    dedupe_motorcycles,
    dedupe_people,
    detect_triple_riding,
    head_from_keypoints,
    head_point,
)

# ============================================================
# PROJECT SETTINGS
# ============================================================

VIDEO_DIR = cfg.VIDEO_DIR
IMAGE_DIR = cfg.TEST_IMAGE_DIR

OUTPUT_IMAGE_DIR = cfg.PROJECT_ROOT / "runs" / "test_images"
OUTPUT_VIDEO_DIR = cfg.PROJECT_ROOT / "runs" / "videos"

IMAGE_EXTENSIONS = set(cfg.IMAGE_EXTENSIONS)
VIDEO_EXTENSIONS = set(cfg.VIDEO_EXTENSIONS)

# Detects motorcycles. Tested on a hard close-up photo taken through a car
# window: yolo11n found the bike at only 0.16 confidence, yolo11s at 0.61
# and yolo11m at 0.93. For video on a slow computer use "yolo11s.pt" or
# "yolo11n.pt" (faster, but misses more bikes).
TRAFFIC_MODEL = "yolo11m.pt"
POSE_MODEL = cfg.POSE_MODEL
HELMET_MODEL = cfg.HELMET_MODEL

IMAGE_IMGSZ = 960
VIDEO_IMGSZ = 640
PROCESS_EVERY = 3

MOTO_CONF = 0.25
MOTO_CONF_RETRY = 0.10   # second try if nothing is found (photos only)
PERSON_CONF = 0.15
HELMET_CONF = 0.25

# Second-chance helmet check on a head crop for riders the full-frame
# helmet model missed (small / occluded heads such as children).
RECOVERY_CONF = 0.10
RECOVERY_IMGSZ = 320

# If a rider still has no helmet detection after recovery, mark them
# "NO HELMET (INFERRED)" instead of silently leaving them UNKNOWN.
INFER_NO_HELMET = True

# Duplicate person suppression (same person detected twice).
# OFF by default: it wrongly removed a real rider in testing, and the
# rider-only filter already hides stray boxes. Enable only for debugging.
USE_DUP_SUPPRESSION = False
DUP_IOU = 0.60
DUP_HEAD_DIST = 0.25  # fraction of box width

# Rider validation (applied after the motorcycle association).
# A person is rejected only if their head sits too high above the bike
# measured BOTH against the bike's height AND against their own height.
# Requiring both keeps tall riders on short bike boxes (close-up photos)
# while still rejecting standing people / mannequins beside a bike.
# Measured with real detections on 6 test photos: riders were at most
# 0.59 bike-heights / 0.51 person-heights above the bike box, while
# mannequins beside a parked scooter were at least 0.94 / 0.64.
MAX_HEAD_ABOVE_MOTO = 0.80
MAX_HEAD_ABOVE_PERSON = 0.50

# Two "riders" on one motorcycle are the same person if their boxes
# overlap this much AND their heads are almost the same point.
MERGE_HEAD_DIST = 0.15   # fraction of the wider box
MERGE_OVERLAP = 0.50     # intersection / smaller box
MERGE_IOU = 0.70         # or simply near-identical boxes

# A helmet must sit on the rider's keypoint head (box grown by this).
HELMET_HEAD_MARGIN = 0.15

# Stricter threshold for claiming "WITH HELMET" (a false "safe" hides a
# violation). Raise it if hair / turbans are being labelled as helmets;
# the helmet confidence is now printed so you can pick a value.
HELMET_CONF_WITH = HELMET_CONF

# A matched helmet below this confidence is shown as "needs review"
# instead of a confident WITH HELMET (hair / turbans score low).
HELMET_CONF_CONFIDENT = 0.50

# Second look: a matched WITH-helmet below VERIFY_BELOW is re-checked on
# a zoomed head crop. A clear "without" result overrides it, and a
# confident "with" result upgrades it. Otherwise it stays as it was.
VERIFY_ENABLED = True
VERIFY_BELOW = 0.75
VERIFY_WITHOUT_MIN = 0.30

# Draw motorcycles that have no riders (parked bikes)?
SHOW_EMPTY_MOTORCYCLES = False

# Status labels
WITH = "WITH HELMET"
WITHOUT = "WITHOUT HELMET"
INFERRED = "NO HELMET (INFERRED)"
UNCERTAIN = "HELMET? (LOW CONF)"
UNKNOWN = "UNKNOWN"

_model_cache = None


# ============================================================
# MODEL LOADING
# ============================================================

def get_models():
    """Load all three YOLO models once."""

    global _model_cache

    if _model_cache is None:

        print("\nLoading YOLO11 models...")
        print("Please wait.")

        helmet_file = cfg.model_path(HELMET_MODEL)

        if not Path(helmet_file).exists():
            raise FileNotFoundError(
                f"Helmet model not found: {helmet_file}\n"
                "Place helmet_detector.pt in the models/ folder "
                "(see models/README.md)."
            )

        _model_cache = (
            YOLO(cfg.model_path(TRAFFIC_MODEL)),
            YOLO(cfg.model_path(POSE_MODEL)),
            YOLO(helmet_file),
        )

        print("All models loaded successfully!")

    return _model_cache


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def head_of(person):
    """Head point from keypoints, or an estimate from the box."""

    head = head_point(person)

    if head is not None:
        return head

    x1, y1, x2, y2 = person["box"]

    return ((x1 + x2) / 2, y1 + (y2 - y1) * 0.15)


def is_no_helmet(status):
    return status in (WITHOUT, INFERRED)


def rider_status(helmet):
    """Status for a rider from their matched helmet detection."""

    if helmet["status"] == WITH and helmet["conf"] < HELMET_CONF_CONFIDENT:
        return UNCERTAIN

    return helmet["status"]


def is_plausible_rider(person, moto):
    """Reject people whose head is far above the motorcycle."""

    moto_top = moto["box"][1]
    moto_height = max(moto["box"][3] - moto["box"][1], 1)
    person_height = max(person["box"][3] - person["box"][1], 1)

    _, head_y = head_of(person)

    above = moto_top - head_y  # how far the head sits above the bike

    too_high_for_bike = above > MAX_HEAD_ABOVE_MOTO * moto_height
    too_high_for_person = above > MAX_HEAD_ABOVE_PERSON * person_height

    return not (too_high_for_bike and too_high_for_person)


def best_overlap_moto(person, motorcycles):
    """The motorcycle that overlaps this person's box the most."""

    px1, py1, px2, py2 = person["box"]
    area = max((px2 - px1) * (py2 - py1), 1)

    best, best_score = None, 0.0

    for moto in motorcycles:

        mx1, my1, mx2, my2 = moto["box"]

        ix = max(0, min(px2, mx2) - max(px1, mx1))
        iy = max(0, min(py2, my2) - max(py1, my1))

        score = ix * iy / area

        if score > best_score:
            best, best_score = moto, score

    return best


def rejection_note(person, moto):
    """Human-readable reason a person was rejected as a rider."""

    moto_height = max(moto["box"][3] - moto["box"][1], 1)
    person_height = max(person["box"][3] - person["box"][1], 1)

    above = moto["box"][1] - head_of(person)[1]

    return (
        f"P{person['id']} rejected as non-rider: head is "
        f"{above / moto_height:.2f} bike-heights and "
        f"{above / person_height:.2f} person-heights above the bike"
    )


def same_person(a, b):
    """True if two person boxes are really one person detected twice."""

    if box_iou(a["box"], b["box"]) >= MERGE_IOU:
        return True

    ax, ay = head_of(a)
    bx, by = head_of(b)

    width = max(
        a["box"][2] - a["box"][0],
        b["box"][2] - b["box"][0],
        1
    )

    heads_close = math.hypot(ax - bx, ay - by) / width < MERGE_HEAD_DIST

    return heads_close and box_ioa(a["box"], b["box"]) >= MERGE_OVERLAP


def clean_associations(associations, people, motorcycles):
    """
    Remove non-riders and merge duplicate detections of one rider,
    only among people already attached to the same motorcycle.
    Returns (cleaned_associations, rejected_count, merged_count, notes).
    """

    people_by_id = {p["id"]: p for p in people}
    motos_by_id = {m["id"]: m for m in motorcycles}

    cleaned = {}
    rejected = merged = 0
    notes = []

    for moto_id, ids in associations.items():

        moto = motos_by_id.get(moto_id)

        candidates = []

        for pid in ids:

            person = people_by_id.get(pid)

            if person is None:
                continue

            check_moto = moto

            if check_moto is None:
                # The association key does not match any motorcycle id:
                # never skip the check, use the closest bike instead.
                check_moto = best_overlap_moto(person, motorcycles)
                notes.append(
                    f"WARNING: association key {moto_id} matches no "
                    f"motorcycle id; checked P{pid} against the "
                    f"best-overlapping bike"
                )

            if check_moto is not None and not is_plausible_rider(
                person, check_moto
            ):
                rejected += 1
                notes.append(rejection_note(person, check_moto))
                continue

            candidates.append(person)

        kept = []

        for person in sorted(
            candidates, key=lambda p: p["conf"], reverse=True
        ):

            if any(same_person(person, other) for other in kept):
                merged += 1
                continue

            kept.append(person)

        cleaned[moto_id] = sorted(p["id"] for p in kept)

    return cleaned, rejected, merged, list(dict.fromkeys(notes))


# ============================================================
# MOTORCYCLE DETECTION
# ============================================================

def detect_motorcycles(
    det_model, frame, imgsz, conf=MOTO_CONF, augment=False
):

    result = det_model.predict(
        frame,
        conf=conf,
        imgsz=imgsz,
        classes=[3],
        augment=augment,
        verbose=False
    )[0]

    if result.boxes is None:
        return []

    boxes = result.boxes.xyxy.cpu().tolist()
    confs = result.boxes.conf.cpu().tolist()

    return [
        {"id": i, "box": b, "conf": c}
        for i, (b, c) in enumerate(zip(boxes, confs))
    ]


def find_motorcycles(det_model, frame, imgsz, retry=True):
    """
    Normal detection first. If nothing is found and retry is allowed,
    try (1) a lower confidence, then (2) lower confidence at a larger
    image size with test-time augmentation. Returns (motorcycles, retried).
    """

    found = detect_motorcycles(det_model, frame, imgsz)

    if found or not retry:
        return found, False

    big = int(round(imgsz * 1.5 / 32)) * 32

    for conf, size, augment in (
        (MOTO_CONF_RETRY, imgsz, False),
        (MOTO_CONF_RETRY, big, True),
    ):

        found = detect_motorcycles(
            det_model, frame, size, conf=conf, augment=augment
        )

        if found:
            return found, True

    return [], False


# ============================================================
# PERSON DETECTION
# ============================================================

def detect_people(pose_model, frame, imgsz):

    result = pose_model.predict(
        frame,
        conf=PERSON_CONF,
        imgsz=imgsz,
        iou=0.7,
        verbose=False
    )[0]

    if result.boxes is None:
        return []

    boxes = result.boxes.xyxy.cpu().tolist()
    confs = result.boxes.conf.cpu().tolist()

    kp_xy = kp_conf = None

    if result.keypoints is not None:

        kp_xy = result.keypoints.xy.cpu().tolist()

        if result.keypoints.conf is not None:
            kp_conf = result.keypoints.conf.cpu().tolist()

    people = []

    for i, (box, conf) in enumerate(zip(boxes, confs)):

        head = None

        if kp_xy is not None and kp_conf is not None:
            head = head_from_keypoints(kp_xy[i], kp_conf[i])

        people.append({"id": i, "box": box, "conf": conf, "head": head})

    return people


def suppress_duplicate_people(people):
    """
    Drop a lower-confidence person if it overlaps heavily with a
    higher-confidence one AND their heads are almost the same point.
    Real riders overlap a lot too, but their heads are far apart.
    """

    kept = []

    for person in sorted(people, key=lambda p: p["conf"], reverse=True):

        duplicate = False

        for other in kept:

            if box_iou(person["box"], other["box"]) < DUP_IOU:
                continue

            hx1, hy1 = head_of(person)
            hx2, hy2 = head_of(other)

            width = max(other["box"][2] - other["box"][0], 1)

            if math.hypot(hx1 - hx2, hy1 - hy2) / width < DUP_HEAD_DIST:
                duplicate = True
                break

        if duplicate:
            pass  # duplicate person dropped
        else:
            kept.append(person)

    return sorted(kept, key=lambda p: p["id"])


# ============================================================
# HELMET DETECTION
# ============================================================

def parse_helmets(result, offset=(0, 0), scale=1.0):

    helmets = []

    if result.boxes is None:
        return helmets

    boxes = result.boxes.xyxy.cpu().tolist()
    confs = result.boxes.conf.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()

    ox, oy = offset

    for box, conf, cls in zip(boxes, confs, classes):

        cls = int(cls)

        if cls not in (0, 1):
            continue

        if cls == 0 and conf < HELMET_CONF_WITH:
            continue

        x1, y1, x2, y2 = box

        helmets.append({
            "box": [x1 / scale + ox, y1 / scale + oy,
                    x2 / scale + ox, y2 / scale + oy],
            "conf": conf,
            "class_id": cls,
            "status": WITH if cls == 0 else WITHOUT,
        })

    return helmets


def detect_helmets(helmet_model, frame, imgsz):

    result = helmet_model.predict(
        frame,
        conf=HELMET_CONF,
        imgsz=imgsz,
        verbose=False
    )[0]

    return parse_helmets(result)


# ============================================================
# HELMET <-> PERSON ASSOCIATION (global, one-to-one)
# ============================================================

def helmet_person_cost(helmet, person):
    """Lower is better. None means the pair is not plausible."""

    hx1, hy1, hx2, hy2 = helmet["box"]
    hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2

    px1, py1, px2, py2 = person["box"]
    pw, ph = max(px2 - px1, 1), max(py2 - py1, 1)

    head_x, head_y = head_of(person)

    head_inside = hx1 <= head_x <= hx2 and hy1 <= head_y <= hy2

    if head_point(person) is not None:

        # Real keypoint head: the helmet must actually sit on it.
        mx = (hx2 - hx1) * HELMET_HEAD_MARGIN
        my = (hy2 - hy1) * HELMET_HEAD_MARGIN

        if not (
            hx1 - mx <= head_x <= hx2 + mx
            and hy1 - my <= head_y <= hy2 + my
        ):
            return None

    else:

        in_upper_region = (
            px1 - pw * 0.10 <= hcx <= px2 + pw * 0.10
            and py1 - ph * 0.10 <= hcy <= py1 + ph * 0.55
        )

        if not in_upper_region:
            return None

    cost = math.hypot((hcx - head_x) / pw, (hcy - head_y) / ph)

    if head_inside:
        cost *= 0.5

    return cost


def associate_helmets(riders, helmets):
    """
    Global matching: build every plausible (helmet, rider) pair, then
    assign the cheapest pairs first so one helmet can never steal
    another rider's head.
    """

    statuses = {p["id"]: UNKNOWN for p in riders}
    matches = []

    pairs = []

    for h_index, helmet in enumerate(helmets):
        for person in riders:

            cost = helmet_person_cost(helmet, person)

            if cost is not None:
                pairs.append((cost, h_index, person["id"]))

    pairs.sort()

    used_helmets, used_people = set(), set()

    for cost, h_index, person_id in pairs:

        if h_index in used_helmets or person_id in used_people:
            continue

        used_helmets.add(h_index)
        used_people.add(person_id)

        statuses[person_id] = rider_status(helmets[h_index])
        matches.append({"person_id": person_id, "helmet": helmets[h_index]})

    return statuses, matches


# ============================================================
# RECOVERY: re-check riders with no helmet match using a head crop
# ============================================================

def head_crop_detections(helmet_model, frame, person):

    px1, py1, px2, py2 = person["box"]
    pw, ph = px2 - px1, py2 - py1

    hx, hy = head_of(person)

    radius = max(pw * 0.40, ph * 0.14, 20)

    h_img, w_img = frame.shape[:2]

    cx1 = int(max(0, hx - radius))
    cy1 = int(max(0, hy - radius))
    cx2 = int(min(w_img, hx + radius))
    cy2 = int(min(h_img, hy + radius))

    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return []

    crop = frame[cy1:cy2, cx1:cx2]

    scale = max(1.0, 224 / min(crop.shape[:2]))

    if scale > 1.0:
        crop = cv2.resize(
            crop, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    result = helmet_model.predict(
        crop,
        conf=RECOVERY_CONF,
        imgsz=RECOVERY_IMGSZ,
        verbose=False
    )[0]

    found = parse_helmets(result, offset=(cx1, cy1), scale=scale)

    # Ignore neighbours' helmets that happen to be inside the crop.
    found = [
        h for h in found
        if math.hypot(
            (h["box"][0] + h["box"][2]) / 2 - hx,
            (h["box"][1] + h["box"][3]) / 2 - hy
        ) <= radius * 0.6
    ]

    return found


def recover_helmet(helmet_model, frame, person):

    found = head_crop_detections(helmet_model, frame, person)

    return max(found, key=lambda h: h["conf"]) if found else None


def verify_decision(crop_dets):
    """
    Decide from head-crop detections whether to override a doubtful
    WITH-helmet result. Returns a detection dict, or None to keep it.
    """

    withs = [d for d in crop_dets if d["class_id"] == 0]
    withouts = [d for d in crop_dets if d["class_id"] == 1]

    best_with = max(withs, key=lambda d: d["conf"], default=None)
    best_without = max(withouts, key=lambda d: d["conf"], default=None)

    with_conf = best_with["conf"] if best_with else 0.0
    without_conf = best_without["conf"] if best_without else 0.0

    if best_without and without_conf >= VERIFY_WITHOUT_MIN \
            and without_conf > with_conf:
        return best_without

    if best_with and with_conf >= HELMET_CONF_CONFIDENT:
        return best_with

    return None


# ============================================================
# FRAME ANALYSIS
# ============================================================

def analyze_frame(
    frame, det_model, pose_model, helmet_model, imgsz,
    retry_motorcycle=True
):

    raw_motorcycles, moto_retried = find_motorcycles(
        det_model, frame, imgsz, retry=retry_motorcycle
    )

    raw_people = detect_people(pose_model, frame, imgsz)
    raw_helmets = detect_helmets(helmet_model, frame, imgsz)

    people = dedupe_people(raw_people)

    if USE_DUP_SUPPRESSION:
        people = suppress_duplicate_people(people)

    motorcycles = dedupe_motorcycles(raw_motorcycles)

    associations = associate_riders(people, motorcycles)
    associations, rejected, merged, notes = clean_associations(
        associations, people, motorcycles
    )
    rider_counts = count_riders(associations)
    violations = detect_triple_riding(rider_counts)

    # Only people on a motorcycle matter for helmet checks.
    rider_ids = {pid for ids in associations.values() for pid in ids}
    riders = [p for p in people if p["id"] in rider_ids]

    helmets = list(raw_helmets)

    helmet_statuses, helmet_matches = associate_helmets(riders, helmets)

    # Second chance for riders with no helmet match.
    recovered = 0

    for person in riders:

        if helmet_statuses[person["id"]] != UNKNOWN:
            continue

        found = recover_helmet(helmet_model, frame, person)

        if found is not None:

            helmet_statuses[person["id"]] = rider_status(found)
            helmets.append(found)
            helmet_matches.append(
                {"person_id": person["id"], "helmet": found}
            )
            recovered += 1

        elif INFER_NO_HELMET:

            helmet_statuses[person["id"]] = INFERRED

    # Second look at doubtful WITH-helmet riders.
    verified = 0

    if VERIFY_ENABLED:

        match_by_person = {
            m["person_id"]: m["helmet"] for m in helmet_matches
        }

        for person in riders:

            pid = person["id"]
            helmet = match_by_person.get(pid)

            if (
                helmet is None
                or helmet["status"] != WITH
                or helmet["conf"] >= VERIFY_BELOW
            ):
                continue

            verdict = verify_decision(
                head_crop_detections(helmet_model, frame, person)
            )

            if verdict is None:
                continue

            new_status = rider_status(verdict)

            if new_status == helmet_statuses[pid]:
                continue

            helmet_statuses[pid] = new_status
            helmets = [h for h in helmets if h is not helmet] + [verdict]
            helmet_matches = [
                m for m in helmet_matches if m["person_id"] != pid
            ] + [{"person_id": pid, "helmet": verdict}]
            verified += 1

    return {
        "raw_people": len(raw_people),
        "raw_motorcycles": len(raw_motorcycles),
        "raw_helmets": len(raw_helmets),
        "recovered_helmets": recovered,
        "people": people,
        "riders": riders,
        "ignored_people": len(people) - len(riders),
        "moto_retried": moto_retried,
        "verified_changes": verified,
        "rejection_notes": notes,
        "rejected_riders": rejected,
        "merged_duplicates": merged,
        "motorcycles": motorcycles,
        "associations": associations,
        "rider_counts": rider_counts,
        "violations": violations,
        "helmets": helmets,
        "helmet_statuses": helmet_statuses,
        "helmet_matches": helmet_matches,
    }


# ============================================================
# DISPLAY DETECTION SUMMARY
# ============================================================

def print_summary(analysis, expected=None):

    expected_text = str(expected) if expected is not None else "Not provided"

    print("\nDETECTION SUMMARY")
    print("-" * 55)

    print(
        f"People detected: {analysis['raw_people']} raw "
        f"| {len(analysis['people'])} after deduplication "
        f"| {len(analysis['riders'])} on motorcycles "
        f"({analysis['ignored_people']} ignored)"
    )
    print(
        f"Motorcycles detected: {analysis['raw_motorcycles']} raw "
        f"| {len(analysis['motorcycles'])} after merging"
        + (" (found on low-confidence retry)"
           if analysis["moto_retried"] else "")
    )
    print(
        f"Helmet detections: {analysis['raw_helmets']} "
        f"(+{analysis['recovered_helmets']} recovered from head crops)"
    )

    print(
        f"Rejected as non-riders: {analysis['rejected_riders']} "
        f"| duplicates merged: {analysis['merged_duplicates']} "
        f"| changed by head-crop check: {analysis['verified_changes']}"
    )

    for note in analysis["rejection_notes"]:
        print(f"  {note}")

    for helmet in analysis["helmets"]:
        print(f"  helmet box: {helmet['status']} conf={helmet['conf']:.2f}")

    print("\nRIDER HELMET STATUS")

    for person in analysis["riders"]:
        print(
            f"Person {person['id']}: "
            f"{analysis['helmet_statuses'].get(person['id'], UNKNOWN)}"
        )

    if not analysis["motorcycles"]:
        print("\nNo motorcycles detected.")

    for moto in analysis["motorcycles"]:

        moto_id = moto["id"]

        print(f"\nMotorcycle ID: {moto_id}")
        print(f"Rider IDs: {analysis['associations'].get(moto_id, [])}")
        print(f"Rider count: {analysis['rider_counts'].get(moto_id, 0)}")
        print(f"Status: {analysis['violations'].get(moto_id, 'OK')}")
        print(f"Expected riders: {expected_text}")

    print("-" * 55)


# ============================================================
# DRAW TEXT WITH BACKGROUND
# ============================================================

def draw_label(image, text, x, y, color, font_scale, thickness):

    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    ih, iw = image.shape[:2]

    x = max(0, min(x, iw - tw - 10))
    y = max(th + 10, min(y, ih - baseline - 5))

    cv2.rectangle(
        image, (x, y - th - 8), (x + tw + 8, y + baseline), color, -1
    )

    cv2.putText(
        image, text, (x + 4, y - 3), font,
        font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )

    return th + baseline + 20  # height used, for stacking


def status_color(status):

    if status == WITH:
        return (0, 180, 0)
    if status == WITHOUT:
        return (0, 0, 255)
    if status == INFERRED:
        return (0, 140, 255)
    if status == UNCERTAIN:
        return (0, 200, 255)

    return (150, 150, 150)


# ============================================================
# DRAW DETECTION RESULTS
# ============================================================

def draw_results(frame, analysis):

    annotated = frame.copy()

    short_side = min(annotated.shape[:2])
    thickness = max(2, short_side // 250)
    font_scale = max(0.65, short_side / 900)

    # Riders only (non-riders are ignored). Labels are stacked
    # left-to-right order by head height so they never overlap.
    riders = sorted(analysis["riders"], key=lambda p: head_of(p)[1])

    (_, text_h), _ = cv2.getTextSize(
        "Ag", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    label_y = text_h + 10

    pending_labels = []  # drawn last so no box outline crosses them

    for person in riders:

        x1, y1, x2, y2 = map(int, person["box"])

        status = analysis["helmet_statuses"].get(person["id"], UNKNOWN)
        color = status_color(status)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        hx, hy = map(int, head_of(person))
        cv2.circle(annotated, (hx, hy), thickness * 3, (255, 0, 255), -1)

        text = f"P{person['id']} - {status}"

        pending_labels.append((text, x1, label_y, color))

        (_, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        label_y += th + baseline + 20

    # Helmet boxes (including ones recovered from head crops).
    matched = {id(m["helmet"]) for m in analysis["helmet_matches"]}

    pending_conf = []

    for helmet in analysis["helmets"]:

        # Hide helmets that are not on a rider (e.g. mannequins), but if
        # no rider was found at all, show every helmet detection.
        if analysis["riders"] and id(helmet) not in matched:
            continue

        hx1, hy1, hx2, hy2 = map(int, helmet["box"])

        helmet_color = status_color(rider_status(helmet))

        cv2.rectangle(
            annotated, (hx1, hy1), (hx2, hy2),
            helmet_color, max(1, thickness - 1)
        )

        pending_conf.append(
            (f"{helmet['conf']:.2f}", hx1, hy2 + 24, helmet_color)
        )

        if not analysis["riders"]:
            # Helmet-only fallback: no rider was found, so say what the
            # helmet model saw directly on the box.
            pending_labels.append(
                (rider_status(helmet), hx1, hy1 - 8, helmet_color)
            )

    # Motorcycles and rider counts.
    for moto in analysis["motorcycles"]:

        x1, y1, x2, y2 = map(int, moto["box"])

        count = analysis["rider_counts"].get(moto["id"], 0)
        status = analysis["violations"].get(moto["id"], "OK")

        if count == 0 and not SHOW_EMPTY_MOTORCYCLES:
            continue

        color = (0, 0, 255) if status != "OK" else (255, 140, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        draw_label(
            annotated,
            f"RIDERS: {count} | {status}",
            x1,
            (y2 + 30
             if y2 + 30 < annotated.shape[0] - 10
             else min(y2, annotated.shape[0]) - 25),
            color, font_scale, thickness
        )

    banner = None

    if not analysis["motorcycles"]:
        banner = "NO MOTORCYCLE DETECTED"
    elif not analysis["riders"]:
        banner = (
            f"NO RIDERS FOUND ({analysis['rejected_riders']} "
            f"rejected as non-riders)"
        )

    if banner:
        draw_label(
            annotated, banner, 10, annotated.shape[0] - 15,
            (80, 80, 80), font_scale, thickness
        )

    for text, x, y, color in pending_conf:
        draw_label(
            annotated, text, x, y, color,
            font_scale * 0.6, max(1, thickness - 1)
        )

    for text, x, y, color in pending_labels:
        draw_label(annotated, text, x, y, color, font_scale, thickness)

    return annotated


# ============================================================
# RESIZE IMAGE FOR DISPLAY
# ============================================================

def fit_to_screen(image, max_width=1100, max_height=750):

    h, w = image.shape[:2]

    scale = min(max_width / w, max_height / h, 1.0)

    return cv2.resize(image, (int(w * scale), int(h * scale)))


# ============================================================
# PHOTO TESTING
# ============================================================

def run_photo_test(image_path, expected=None):

    print("\nStarting photo detection...")

    det_model, pose_model, helmet_model = get_models()

    frame = cv2.imread(str(image_path))

    if frame is None:
        print("ERROR: Could not read the selected photo.")
        return

    analysis = analyze_frame(
        frame, det_model, pose_model, helmet_model, IMAGE_IMGSZ
    )

    print_summary(analysis, expected)

    annotated = draw_results(frame, analysis)

    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_IMAGE_DIR / f"{image_path.stem}_annotated.jpg"

    cv2.imwrite(str(output_path), annotated)

    print(f"\nAnnotated photo saved at: {output_path}")

    try:
        cv2.imshow(
            "Two-Wheeler Safety - Photo Testing", fit_to_screen(annotated)
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("(No display available - open the saved photo instead.)")


# ============================================================
# VIDEO TESTING
# ============================================================

def run_video_test(video_path, expected=None):

    print("\nStarting video detection...")
    print(f"Analysing every {PROCESS_EVERY} frames.")
    print("Press Q in the video window to stop.")

    det_model, pose_model, helmet_model = get_models()

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print("ERROR: Could not open the selected video.")
        return

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_VIDEO_DIR / f"{video_path.stem}_annotated.mp4"

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    if not writer.isOpened():
        print("ERROR: Could not create output video.")
        capture.release()
        return

    frame_index = 0
    analysis = None
    last_signature = None
    max_riders = 0

    try:

        while True:

            ok, frame = capture.read()

            if not ok:
                break

            if frame_index % PROCESS_EVERY == 0:

                analysis = analyze_frame(
                    frame, det_model, pose_model, helmet_model, VIDEO_IMGSZ,
                    retry_motorcycle=False  # keep video fast
                )

                counts = list(analysis["rider_counts"].values())

                signature = (
                    tuple(sorted(counts)),
                    tuple(sorted(analysis["helmet_statuses"].items())),
                )

                if counts:
                    max_riders = max(max_riders, max(counts))

                if signature != last_signature:

                    print(f"\nFrame {frame_index + 1}/{total_frames}")
                    print_summary(analysis, expected)

                    last_signature = signature

            annotated = (
                draw_results(frame, analysis)
                if analysis is not None
                else frame
            )

            writer.write(annotated)

            try:
                cv2.imshow(
                    "Two-Wheeler Safety - Video Testing",
                    fit_to_screen(annotated)
                )
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                key = 0  # headless: keep processing, just no window

            if key == ord("q"):
                print("\nVideo stopped by user.")
                break

            frame_index += 1

    finally:

        capture.release()
        writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print("\nVIDEO TESTING SUMMARY")
    print("-" * 45)
    print(f"Frames processed: {frame_index}/{total_frames}")
    print(f"Highest rider count on one motorcycle: {max_riders}")
    print(
        f"Expected riders: "
        f"{expected if expected is not None else 'Not provided'}"
    )
    print(f"Annotated video saved at: {output_path}")


# ============================================================
# EXPECTED RIDER COUNT
# ============================================================

def get_expected_count():

    while True:

        try:
            value = input(
                "\nEnter expected rider count (or press Enter to skip): "
            ).strip()
        except EOFError:
            return None

        if not value:
            return None

        try:

            count = int(value)

            if count >= 1:
                return count

            print("Please enter a number greater than zero.")

        except ValueError:

            print("Please enter a valid whole number.")


# ============================================================
# FILE SELECTION MENU
# ============================================================

def select_file(folder, extensions, file_type):

    folder.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [f for f in folder.iterdir()
         if f.is_file() and f.suffix.lower() in extensions],
        key=lambda f: f.name.lower()
    )

    print("\n" + "=" * 60)
    print(f"              {file_type.upper()} SELECTION")
    print("=" * 60)
    print(f"\nFolder: {folder}")

    if not files:
        print("\nNo files found in this folder.")
        print("Please add files and try again.")
        return None

    print("\nAVAILABLE FILES")
    print("-" * 60)

    for index, file in enumerate(files, start=1):

        size_mb = file.stat().st_size / (1024 * 1024)

        print(f"  {index:>2}. {file.name:<35} {size_mb:>7.2f} MB")

    print("-" * 60)
    print("  0. Back to main menu")

    while True:

        try:
            choice = input("\nSelect a file by entering its number: ").strip()
        except EOFError:
            return None

        if choice == "0":
            return None

        try:

            index = int(choice)

            if 1 <= index <= len(files):
                return files[index - 1]

            print("Invalid selection. Choose a number from the list.")

        except ValueError:

            print("Please enter a valid number.")


# ============================================================
# MAIN MENU
# ============================================================

def main():

    try:
        get_models()
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        return 1

    while True:

        print("\n" + "=" * 60)
        print("          TWO-WHEELER SAFETY DETECTION SYSTEM")
        print("=" * 60)
        print("\n                    MAIN MENU")
        print("-" * 60)
        print("  1. Photo Testing")
        print("  2. Video Testing")
        print("  0. Exit")
        print("-" * 60)

        try:
            choice = input("\nEnter your choice: ").strip()
        except EOFError:
            choice = "0"

        if choice == "0":

            print("\nThank you for using the system.")
            print("Exiting...")
            return 0

        elif choice == "1":

            selected = select_file(IMAGE_DIR, IMAGE_EXTENSIONS, "Photo Testing")

            if selected is not None:

                expected = get_expected_count()

                print("\nSelected photo:", selected.name)
                run_photo_test(selected, expected)
                print("\nPhoto testing completed.")

        elif choice == "2":

            selected = select_file(VIDEO_DIR, VIDEO_EXTENSIONS, "Video Testing")

            if selected is not None:

                expected = get_expected_count()

                print("\nSelected video:", selected.name)
                run_video_test(selected, expected)
                print("\nVideo testing completed.")

        else:

            print("\nInvalid choice. Please select 1, 2 or 0.")


if __name__ == "__main__":
    raise SystemExit(main())