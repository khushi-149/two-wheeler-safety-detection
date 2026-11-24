"""
Geometry that decides WHO rides WHICH motorcycle and which head belongs to whom.

Pure functions (no models, no I/O) so they can be unit-tested.
"""

from geometry import box_center, iou, point_in_box
from rider_association import head_from_keypoints, head_point  # noqa: F401  (re-exported)
from settings import (
    CROP_PAD_BOTTOM, CROP_PAD_SIDE, CROP_PAD_TOP,
    EXPAND_BOTTOM, EXPAND_SIDE, EXPAND_TOP,
    HEAD_TOP_ALLOWANCE, HEAD_ZONE,
    HIDDEN_BELOW, HIDDEN_SIDE, HIDDEN_TOP,
    MAX_HEAD_ABOVE_MOTO, MAX_HEAD_ABOVE_PERSON,
    MIN_HIDDEN_HEAD_TO_BIKE, MIN_HIDDEN_HEAD_VS_RIDERS, MIN_PERSON_TO_BIKE,
    MIN_RIDER_BODY_OVERLAP,
    MIN_HISTORY_FOR_TRIPLE, MIN_RIDER_OVERLAP,
    SEAT_MARGIN_BOTTOM, SEAT_MARGIN_SIDE, SEAT_MARGIN_TOP,
)


def expanded_box(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - EXPAND_SIDE * w, y1 - EXPAND_TOP * h,
            x2 + EXPAND_SIDE * w, y2 + EXPAND_BOTTOM * h)


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


def dedupe_detections(dets, iou_threshold=0.5):
    """Greedy dedupe: crops of neighbouring bikes can see the same head twice."""
    kept = []

    for det in sorted(dets, key=lambda d: d["conf"], reverse=True):
        if all(iou(det["box"], k["box"]) < iou_threshold for k in kept):
            kept.append(det)

    return kept


def hip_from_keypoints(keypoints, confidences):
    """Mid-point of the detected hips (COCO keypoints 11 and 12), or None."""
    points = []

    for idx in (11, 12):
        if idx >= len(keypoints):
            continue

        x, y = keypoints[idx][0], keypoints[idx][1]
        conf = confidences[idx] if confidences is not None else 1.0

        if conf >= 0.30 and (x or y):
            points.append((x, y))

    if not points:
        return None

    return (sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points))


def person_seat_point(person):
    """Where the person sits: the hips if known, else 60% down their body box."""
    hip = person.get("hip")

    if hip is not None:
        return hip

    x1, y1, x2, y2 = person["box"]
    return ((x1 + x2) / 2.0, y1 + 0.6 * (y2 - y1))


def smoothed_count(history, min_history_for_triple=MIN_HISTORY_FOR_TRIPLE):
    """
    Lower median of recent rider counts.

    A count of 3 or more is only reported once ``min_history_for_triple``
    analyses exist; until then it is capped at 2 so one glitchy frame cannot
    raise a triple-riding violation. Photos have exactly one analysis, so they
    pass ``min_history_for_triple=1``.
    """
    if not history:
        return 0

    ordered = sorted(history)
    value = ordered[(len(ordered) - 1) // 2]

    if value >= 3 and len(history) < min_history_for_triple:
        value = 2

    return value


def is_plausible_rider(person, moto):
    """
    False for a person whose head is far above the motorcycle (a mannequin,
    pedestrian or bystander standing behind a parked bike). Both the bike-
    relative AND the person-relative distance must be exceeded, so tall
    riders on short bike boxes (close-up photos) are kept.
    """
    moto_top = moto["box"][1]
    moto_height = max(moto["box"][3] - moto["box"][1], 1.0)
    person_height = max(person["box"][3] - person["box"][1], 1.0)

    if person_height < MIN_PERSON_TO_BIKE * moto_height:
        return False                                  # tiny = far-away bystander

    # a rider's body overlaps the bike; a person sitting on steps behind it does not
    px1, py1, px2, py2 = person["box"]
    mx1, my1, mx2, my2 = moto["box"]
    shared = (max(0.0, min(px2, mx2) - max(px1, mx1))
              * max(0.0, min(py2, my2) - max(py1, my1)))

    if shared < MIN_RIDER_BODY_OVERLAP * max((px2 - px1) * (py2 - py1), 1.0):
        return False

    above = moto_top - head_point(person)[1]

    return not (
        above > MAX_HEAD_ABOVE_MOTO * moto_height
        and above > MAX_HEAD_ABOVE_PERSON * person_height
    )


def assign_people_to_bikes(people, motorcycles):
    """
    Count riders per motorcycle index.

    A person rides the bike their SEAT point (hips) is inside. If the seat is
    inside several bike boxes (bikes side by side) the bike whose centre is
    horizontally closest wins, so a neighbour's rider is not credited to a
    big bike next to it. Only if the seat is inside no bike box do we fall
    back to the bike whose box overlaps most of the person's body.

    Returns (counts, rider_boxes, other_boxes):
      counts       {bike index: number of riders}
      rider_boxes  {bike index: [person boxes that ride it]}
      other_boxes  [person boxes not riding any tracked bike (pedestrians)]
    """
    counts = {i: 0 for i in range(len(motorcycles))}
    rider_boxes = {i: [] for i in range(len(motorcycles))}
    other_boxes = []

    for person in people:

        sx, sy = person_seat_point(person)

        best, best_dist = None, float("inf")

        for idx, moto in enumerate(motorcycles):
            bx1, by1, bx2, by2 = moto["box"]
            bw, bh = bx2 - bx1, by2 - by1

            inside = (
                bx1 - SEAT_MARGIN_SIDE * bw <= sx <= bx2 + SEAT_MARGIN_SIDE * bw
                and by1 - SEAT_MARGIN_TOP * bh <= sy <= by2 + SEAT_MARGIN_BOTTOM * bh
            )

            if not inside or not is_plausible_rider(person, moto):
                continue

            dist = abs(sx - (bx1 + bx2) / 2.0) / max(bw, 1.0)

            if dist < best_dist:
                best, best_dist = idx, dist

        # Fallback: seat inside no bike -> most body overlap
        if best is None:
            px1, py1, px2, py2 = person["box"]
            parea = max(1.0, (px2 - px1) * (py2 - py1))
            best_score = 0.0

            for idx, moto in enumerate(motorcycles):
                bx1, by1, bx2, by2 = moto["box"]
                bw = bx2 - bx1

                if not (bx1 - 0.15 * bw <= (px1 + px2) / 2.0 <= bx2 + 0.15 * bw):
                    continue

                if not is_plausible_rider(person, moto):
                    continue

                ix1, iy1 = max(px1, bx1), max(py1, by1)
                ix2, iy2 = min(px2, bx2), min(py2, by2)
                score = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) / parea

                if score > best_score:
                    best, best_score = idx, score

            if best_score < MIN_RIDER_OVERLAP:
                best = None

        if best is not None:
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


def _hidden_head_is_believable(det, moto, known_heads):
    """
    A head with no body must be about as big as the riders' heads. A much
    smaller head belongs to someone far behind the bike, not to a pillion.
    """
    head_h = det["box"][3] - det["box"][1]
    bike_h = max(moto["box"][3] - moto["box"][1], 1.0)

    if head_h < MIN_HIDDEN_HEAD_TO_BIKE * bike_h:
        return False

    sizes = sorted(d["box"][3] - d["box"][1] for d in known_heads)

    if sizes and head_h < MIN_HIDDEN_HEAD_VS_RIDERS * sizes[len(sizes) // 2]:
        return False

    return True


def assign_heads_to_bikes(helmet_dets, motorcycles, rider_boxes, other_boxes,
                          claims=None, people=None):
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

    If a list is passed as ``claims`` it is filled with (rider body box,
    detection) pairs, so the caller can tell which riders have no head yet.
    If ``people`` is passed, their pose head points are used to match heads.
    """
    owners = []
    claimed = set()

    # 1. Match riders and heads GLOBALLY: every (rider, head) pair that fits is
    #    scored by how far the head is from where the rider's head should be
    #    (the pose model's head point; else the top-centre of the body box), and
    #    the best pairs are taken first. Each rider gets one head, each head one
    #    rider, so riders standing close together can no longer steal each
    #    other's helmets. Any other head in the zone is a second person.
    head_of_body = {tuple(p["box"]): head_point(p) for p in (people or [])}
    pairs = []

    for idx, boxes in rider_boxes.items():
        for body in boxes:
            body_h = max(body[3] - body[1], 1.0)
            target = head_of_body.get(tuple(body)) or (
                (body[0] + body[2]) / 2.0, body[1] + 0.10 * body_h)

            for i, det in enumerate(helmet_dets):
                cx, cy = box_center(det["box"])

                if _head_in_person((cx, cy), body):
                    cost = (((cx - target[0]) ** 2 + (cy - target[1]) ** 2) ** 0.5) / body_h
                    pairs.append((cost, idx, tuple(body), i))

    pairs.sort(key=lambda item: item[0])
    done_bodies = set()

    for cost, idx, body, i in pairs:
        if i in claimed or (idx, body) in done_bodies:
            continue

        claimed.add(i)
        done_bodies.add((idx, body))
        owners.append((helmet_dets[i], idx, False))

        if claims is not None:
            claims.append((list(body), helmet_dets[i]))

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

        if best_idx is not None and _hidden_head_is_believable(
                det, motorcycles[best_idx], [d for d, i, _ in owners if i == best_idx]):
            owners.append((det, best_idx, True))

    return owners