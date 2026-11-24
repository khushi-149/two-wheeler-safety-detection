"""Small, dependency-free box geometry helpers. Boxes are ``(x1, y1, x2, y2)``."""

from __future__ import annotations

from typing import Sequence

Box = Sequence[float]
Point = Sequence[float]


def box_area(box: Box) -> float:
    """Area of a box (0 for an empty or inverted box)."""
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box: Box) -> tuple[float, float]:
    """Centre point of a box."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def point_in_box(point: Point, box: Box) -> bool:
    """True if ``point`` lies inside (or on the edge of) ``box``."""
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _intersection(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two boxes (0.0 when they do not overlap)."""
    a = tuple(map(float, a))
    b = tuple(map(float, b))

    inter = _intersection(a, b)
    union = box_area(a) + box_area(b) - inter

    return inter / union if union > 0 else 0.0


def ioa(a: Box, b: Box) -> float:
    """Intersection over the SMALLER box (1.0 = one box fully inside the other)."""
    a = tuple(map(float, a))
    b = tuple(map(float, b))

    smaller = min(box_area(a), box_area(b))

    return _intersection(a, b) / smaller if smaller > 0 else 0.0