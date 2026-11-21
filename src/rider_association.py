"""Helpers for associating people with motorcycles and detecting triple riding."""

import math


# -------------------- Tunable settings --------------------

HEAD_KEYPOINT_CONF = 0.30
BOX_DUP_IOU = 0.85
MOTO_MERGE_IOU = 0.50

LOWER_BODY_FRACTION = 0.25
MOTO_EXPAND_UP = 0.50
MOTO_EXPAND_SIDE = 0.10

MIN_HEIGHT_RATIO = 0.50
TRIPLE_THRESHOLD = 3


# -------------------- Geometry helpers --------------------

def iou(a, b):
    """Calculate intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def box_area(box):
    x1, y1, x2, y2 = map(float, box)
    return max(0, x2 - x1) * max(0, y2 - y1)


def head_from_keypoints(points, confidences=None, min_conf=HEAD_KEYPOINT_CONF):
    """
    Estimate head centre from COCO keypoints:
    nose=0, eyes=1-2, ears=3-4.
    """
    if points is None:
        return None

    valid = []

    for index in range(min(5, len(points))):
        x, y = points[index][:2]

        confidence = 1.0
        if confidences is not None and index < len(confidences):
            confidence = float(confidences[index])

        if confidence >= min_conf and math.isfinite(float(x)) and math.isfinite(float(y)):
            valid.append((float(x), float(y)))

    if not valid:
        return None

    return (
        sum(point[0] for point in valid) / len(valid),
        sum(point[1] for point in valid) / len(valid),
    )


def head_point(person):
    """Get a person's head point, falling back to the top-centre of the box."""
    head = person.get("head")

    if head is not None:
        return tuple(map(float, head))

    x1, y1, x2, y2 = map(float, person["box"])
    return ((x1 + x2) / 2, y1 + (y2 - y1) * 0.08)


# -------------------- Duplicate removal --------------------

def dedupe_motorcycles(motorcycles, thresh=MOTO_MERGE_IOU):
    """Remove highly overlapping duplicate motorcycle detections."""
    kept = []

    for moto in sorted(motorcycles, key=lambda item: item.get("conf", 0), reverse=True):
        if not any(iou(moto["box"], old["box"]) >= thresh for old in kept):
            kept.append(moto)

    return kept


def dedupe_people(people):
    """
    Remove duplicate person detections.
    Avoid merging people solely because their head points are close.
    """
    kept = []

    for person in sorted(people, key=lambda item: item.get("conf", 0), reverse=True):
        duplicate = False

        for old in kept:
            overlap = iou(person["box"], old["box"])

            if overlap >= BOX_DUP_IOU:
                duplicate = True
                break

        if not duplicate:
            kept.append(person)

    return kept


# -------------------- Rider association --------------------

def associate_riders(people, motorcycles):
    """
    Associate each detected person with at most one motorcycle.

    Input:
      people: [{"box": [x1,y1,x2,y2], "head": (x,y), "conf": 0.9}, ...]
      motorcycles: [{"box": [x1,y1,x2,y2], "conf": 0.9}, ...]

    Output:
      {motorcycle_index: [person_index, ...]}
    """
    associations = {index: [] for index in range(len(motorcycles))}

    for person_index, person in enumerate(people):
        px1, py1, px2, py2 = map(float, person["box"])
        person_height = py2 - py1

        if person_height <= 0:
            continue

        # Use the lower-body position as the association point.
        foot_x = (px1 + px2) / 2
        foot_y = py2 - person_height * LOWER_BODY_FRACTION

        best_moto = None
        best_score = 0.0

        for moto_index, moto in enumerate(motorcycles):
            mx1, my1, mx2, my2 = map(float, moto["box"])
            moto_height = my2 - my1
            moto_width = mx2 - mx1

            if moto_height <= 0 or moto_width <= 0:
                continue

            if person_height < moto_height * MIN_HEIGHT_RATIO:
                continue

            # Expand the motorcycle region upward and sideways.
            ex1 = mx1 - moto_width * MOTO_EXPAND_SIDE
            ex2 = mx2 + moto_width * MOTO_EXPAND_SIDE
            ey1 = my1 - moto_height * MOTO_EXPAND_UP
            ey2 = my2

            if not (ex1 <= foot_x <= ex2 and ey1 <= foot_y <= ey2):
                continue

            # Prefer motorcycles whose box overlaps the lower part of the person.
            lower_box = [px1, py1 + person_height * 0.55, px2, py2]
            score = iou(lower_box, [mx1, my1, mx2, my2])

            # A small positive score allows valid point-in-region matches.
            score += 0.001

            if score > best_score:
                best_score = score
                best_moto = moto_index

        if best_moto is not None:
            associations[best_moto].append(person_index)

    return associations


# -------------------- Rider counting --------------------

def count_riders(associations):
    """Return the number of associated people for each motorcycle."""
    return {
        motorcycle_id: len(person_ids)
        for motorcycle_id, person_ids in associations.items()
    }


def detect_triple_riding(rider_counts):
    """Flag motorcycles carrying three or more detected people."""
    return {
        motorcycle_id: (
            "TRIPLE RIDING"
            if count >= TRIPLE_THRESHOLD
            else "OK"
        )
        for motorcycle_id, count in rider_counts.items()
    }
