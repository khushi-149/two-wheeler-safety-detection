"""Helpers for associating people with motorcycles and detecting triple riding."""

from __future__ import annotations

import math
from typing import Optional, Sequence

from geometry import box_area, iou  # noqa: F401  (re-exported for older imports)

# -------------------- Tunable settings --------------------

HEAD_KEYPOINT_CONF = 0.30
BOX_DUP_IOU = 0.85
MOTO_MERGE_IOU = 0.50

LOWER_BODY_FRACTION = 0.25
MOTO_EXPAND_UP = 0.50
MOTO_EXPAND_SIDE = 0.10

MIN_HEIGHT_RATIO = 0.50
TRIPLE_THRESHOLD = 3


# -------------------- Keypoints --------------------

def head_from_keypoints(
    points: Optional[Sequence[Sequence[float]]],
    confidences: Optional[Sequence[float]] = None,
    min_conf: float = HEAD_KEYPOINT_CONF,
) -> Optional[tuple[float, float]]:
    """
    Estimate the head centre from COCO keypoints
    (nose=0, eyes=1-2, ears=3-4). Returns ``None`` if no keypoint is reliable.
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
        sum(p[0] for p in valid) / len(valid),
        sum(p[1] for p in valid) / len(valid),
    )


def head_point(person: dict) -> tuple[float, float]:
    """A person's head point, falling back to the top of their box."""
    head = person.get("head")

    if head is not None:
        return (float(head[0]), float(head[1]))

    x1, y1, x2, y2 = map(float, person["box"])
    return ((x1 + x2) / 2, y1 + (y2 - y1) * 0.08)


# -------------------- Duplicate removal --------------------

def dedupe_motorcycles(motorcycles: list[dict], thresh: float = MOTO_MERGE_IOU) -> list[dict]:
    """Remove highly overlapping duplicate motorcycle detections (keeps the most confident)."""
    kept: list[dict] = []

    for moto in sorted(motorcycles, key=lambda m: m.get("conf", 0), reverse=True):
        if not any(iou(moto["box"], old["box"]) >= thresh for old in kept):
            kept.append(moto)

    return kept


def dedupe_people(people: list[dict]) -> list[dict]:
    """
    Remove duplicate person detections (near-identical boxes).

    People are NOT merged just because their heads are close: two riders on
    one bike legitimately overlap a lot.
    """
    kept: list[dict] = []

    for person in sorted(people, key=lambda p: p.get("conf", 0), reverse=True):
        if all(iou(person["box"], old["box"]) < BOX_DUP_IOU for old in kept):
            kept.append(person)

    return kept


# -------------------- Rider association --------------------

def associate_riders(people: list[dict], motorcycles: list[dict]) -> dict[int, list[int]]:
    """
    Associate each detected person with at most one motorcycle.

    people:      [{"box": [x1,y1,x2,y2], "head": (x,y), "conf": 0.9}, ...]
    motorcycles: [{"box": [x1,y1,x2,y2], "conf": 0.9}, ...]

    Returns ``{motorcycle_index: [person_index, ...]}``.
    """
    associations: dict[int, list[int]] = {i: [] for i in range(len(motorcycles))}

    for person_index, person in enumerate(people):
        px1, py1, px2, py2 = map(float, person["box"])
        person_height = py2 - py1

        if person_height <= 0:
            continue

        # The lower-body point is used as the association point.
        foot_x = (px1 + px2) / 2
        foot_y = py2 - person_height * LOWER_BODY_FRACTION

        best_moto, best_score = None, 0.0

        for moto_index, moto in enumerate(motorcycles):
            mx1, my1, mx2, my2 = map(float, moto["box"])
            moto_height, moto_width = my2 - my1, mx2 - mx1

            if moto_height <= 0 or moto_width <= 0:
                continue

            if person_height < moto_height * MIN_HEIGHT_RATIO:
                continue

            # Expand the motorcycle region upward and sideways.
            ex1 = mx1 - moto_width * MOTO_EXPAND_SIDE
            ex2 = mx2 + moto_width * MOTO_EXPAND_SIDE
            ey1 = my1 - moto_height * MOTO_EXPAND_UP

            if not (ex1 <= foot_x <= ex2 and ey1 <= foot_y <= my2):
                continue

            # Prefer motorcycles whose box overlaps the lower part of the person.
            lower_box = [px1, py1 + person_height * 0.55, px2, py2]
            score = iou(lower_box, [mx1, my1, mx2, my2]) + 0.001  # allow point-in-region matches

            if score > best_score:
                best_moto, best_score = moto_index, score

        if best_moto is not None:
            associations[best_moto].append(person_index)

    return associations


# -------------------- Rider counting --------------------

def count_riders(associations: dict[int, list[int]]) -> dict[int, int]:
    """Number of associated people for each motorcycle."""
    return {moto_id: len(ids) for moto_id, ids in associations.items()}


def detect_triple_riding(rider_counts: dict[int, int]) -> dict[int, str]:
    """Flag motorcycles carrying three or more people as ``"TRIPLE RIDING"``, else ``"OK"``."""
    return {
        moto_id: "TRIPLE RIDING" if count >= TRIPLE_THRESHOLD else "OK"
        for moto_id, count in rider_counts.items()
    }