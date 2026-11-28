"""
Geometry that decides WHO rides WHICH motorcycle and which head belongs to whom.

Pure functions (no models, no I/O) so they can be unit-tested.
"""

from geometry import box_area, box_center, ioa, iou, point_in_box
from rider_association import head_from_keypoints, head_point  # noqa: F401  (re-exported)
from settings import (
    CROP_PAD_BOTTOM, CROP_PAD_SIDE, CROP_PAD_TOP,
    DUP_PERSON_HEAD_DIST, DUP_PERSON_OVERLAP,
    HEADLESS_KEEP_CONF, HEAD_MERGE_IOA, HIDDEN_HEAD_SIZE_MAX, HIDDEN_HEAD_SIZE_MIN,
    HIDDEN_NEAR_RIDER, MIN_RIDER_AREA_RATIO, NESTED_BIKE_IOA,
    HEAD_INSIDE_MOTO_MAX, MIN_PERSON_TO_BIKE_AREA,
    ALONE_MIN_CONF, BODY_PART_HEAD_FRAC, NESTED_PERSON_IOA, STRAY_DISTANCE, STRONG_PERSON_CONF,
    EXPAND_BOTTOM, EXPAND_SIDE, EXPAND_TOP,
    HEAD_TOP_ALLOWANCE, HEAD_ZONE,
    HIDDEN_BELOW, HIDDEN_SIDE, HIDDEN_TOP,
    MAX_HEAD_ABOVE_MOTO, MAX_HEAD_ABOVE_PERSON,
    MIN_HISTORY_FOR_TRIPLE, MIN_RIDER_OVERLAP, MIN_RIDER_WIDTH_OVERLAP,
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


def same_person(a, b):
    """
    True if two pose detections are really ONE person.

    Both must hold: their heads are almost the same point (closer than
    DUP_PERSON_HEAD_DIST of the wider box), and the smaller box lies at least
    DUP_PERSON_OVERLAP inside the other. Two real riders sitting close together
    have clearly separate heads, so they are kept apart.
    """
    ax1, ay1, ax2, ay2 = a["box"]
    bx1, by1, bx2, by2 = b["box"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller = max(1.0, min(box_area(a["box"]), box_area(b["box"])))

    if inter / smaller < DUP_PERSON_OVERLAP:
        return False

    ha, hb = head_point(a), head_point(b)
    wider = max(ax2 - ax1, bx2 - bx1, 1.0)
    dist = ((ha[0] - hb[0]) ** 2 + (ha[1] - hb[1]) ** 2) ** 0.5

    return dist < DUP_PERSON_HEAD_DIST * wider


def merge_duplicate_people(people):
    """Drop a person who is the same as a more confident one (keeps the best)."""
    kept = []

    for person in sorted(people, key=lambda p: p.get("conf", 0.0), reverse=True):
        if not any(same_person(person, k) for k in kept):
            kept.append(person)

    return kept


def dedupe_nested_bikes(motorcycles, threshold=NESTED_BIKE_IOA):
    """Drop a motorcycle box that lies (almost) inside a bigger one: a partial box
    over the front of the same bike is not a second motorcycle."""
    kept = []

    for moto in sorted(motorcycles, key=lambda m: box_area(m["box"]), reverse=True):
        if all(ioa(moto["box"], k["box"]) < threshold for k in kept):
            kept.append(moto)

    # keep the original order
    return [m for m in motorcycles if any(m is k for k in kept)]


def drop_body_parts(persons):
    """
    Remove a "person" that is really part of another one: its box lies mostly
    inside the other's box and its head point is down in the other's body
    (a rider's legs reported as a second rider).
    """
    kept = []

    for b in persons:
        is_part = False

        for a in persons:
            if a is b or box_area(b["box"]) >= box_area(a["box"]):
                continue

            ax1, ay1, ax2, ay2 = a["box"]
            hx, hy = head_point(b)

            if (ioa(b["box"], a["box"]) >= NESTED_PERSON_IOA
                    and ax1 <= hx <= ax2
                    and hy > ay1 + BODY_PART_HEAD_FRAC * (ay2 - ay1)):
                is_part = True
                break

        if not is_part:
            kept.append(b)

    return kept


def drop_stray_riders(persons):
    """
    Remove doubtful detections far from the confident riders of the same bike
    (see STRONG_PERSON_CONF / ALONE_MIN_CONF / STRAY_DISTANCE in settings).
    """
    # a person without a confidence value (hand-built in tests) counts as sure
    anchors = [p for p in persons if p.get("conf", 1.0) >= STRONG_PERSON_CONF]
    kept = []

    for p in persons:
        conf = p.get("conf", 1.0)

        if conf >= STRONG_PERSON_CONF:
            kept.append(p)
        elif not anchors:
            if conf >= ALONE_MIN_CONF:
                kept.append(p)
        else:
            widths = sorted(a["box"][2] - a["box"][0] for a in anchors)
            limit = STRAY_DISTANCE * widths[len(widths) // 2]
            hx, hy = head_point(p)

            nearest = min(
                ((hx - head_point(a)[0]) ** 2 + (hy - head_point(a)[1]) ** 2) ** 0.5
                for a in anchors
            )

            if nearest <= limit:
                kept.append(p)

    return kept


def body_overlap(person, moto):
    """Share of the person's box that lies over the motorcycle box."""
    px1, py1, px2, py2 = person["box"]
    bx1, by1, bx2, by2 = moto["box"]
    parea = max(1.0, (px2 - px1) * (py2 - py1))
    width = max(0.0, min(px2, bx2) - max(px1, bx1))
    height = max(0.0, min(py2, by2) - max(py1, by1))

    return width * height / parea


def width_overlap(person, moto):
    """Share of the person's WIDTH that lies inside the motorcycle box's width."""
    px1, _py1, px2, _py2 = person["box"]
    bx1, _by1, bx2, _by2 = moto["box"]

    return max(0.0, min(px2, bx2) - max(px1, bx1)) / max(1.0, px2 - px1)


def merge_nested_heads(dets, threshold=HEAD_MERGE_IOA):
    """
    Keep one box per head: a detection mostly INSIDE a more confident one (a face
    box inside a helmet box) is the same head, not a second person.
    """
    kept = []

    for det in sorted(dets, key=lambda d: d["conf"], reverse=True):
        if all(ioa(det["box"], k["box"]) < threshold for k in kept):
            kept.append(det)

    return kept


def drop_headless_people(people, helmet_dets, keep_conf=HEADLESS_KEEP_CONF):
    """
    Remove doubtful "people": no helmet-model head in their head zone AND the pose
    model not sure of them (legs, blurred background, bags, ghosts).
    """
    kept = []

    for person in people:
        has_head = (
            person.get("conf", 0.0) >= keep_conf
            or any(_head_in_person(box_center(h["box"]), person["box"])
                   for h in helmet_dets)
        )

        if has_head:
            kept.append(person)

    return kept


def _near_rider(center, body):
    """True if a head centre is beside/at the head level of a rider's body box."""
    px1, py1, px2, py2 = body
    pw, ph = px2 - px1, py2 - py1

    return (
        px1 - HIDDEN_NEAR_RIDER * pw <= center[0] <= px2 + HIDDEN_NEAR_RIDER * pw
        and py1 - 0.30 * ph <= center[1] <= py1 + HEAD_ZONE * ph
    )


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

    above = moto_top - head_point(person)[1]

    # a head inside the bike box is a knee/leg, not a head
    if -above > HEAD_INSIDE_MOTO_MAX * moto_height:
        return False

    # a speck next to a big bike is not a rider
    if box_area(person["box"]) < MIN_PERSON_TO_BIKE_AREA * box_area(moto["box"]):
        return False

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
    assigned = {i: [] for i in range(len(motorcycles))}

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

            # a real rider's body is over the bike (legs, torso); someone
            # sitting or standing BEHIND a bike is not
            if body_overlap(person, moto) < MIN_RIDER_OVERLAP:
                continue

            # ... and is over it side to side (a pedestrian walking past is not)
            if width_overlap(person, moto) < MIN_RIDER_WIDTH_OVERLAP:
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

                if width_overlap(person, moto) < MIN_RIDER_WIDTH_OVERLAP:
                    continue

                ix1, iy1 = max(px1, bx1), max(py1, by1)
                ix2, iy2 = min(px2, bx2), min(py2, by2)
                score = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) / parea

                if score > best_score:
                    best, best_score = idx, score

            if best_score < MIN_RIDER_OVERLAP:
                best = None

        if best is not None:
            assigned[best].append(person)
        else:
            other_boxes.append(person["box"])

    # Someone tiny next to the other riders on the same bike is a pedestrian
    # in the distance, not a rider.
    for idx, riders in assigned.items():
        found = riders
        riders = drop_stray_riders(drop_body_parts(found))
        other_boxes.extend(p["box"] for p in found if p not in riders)

        biggest = max((box_area(p["box"]) for p in riders), default=0.0)

        for person in riders:
            if len(riders) > 1 and box_area(person["box"]) < MIN_RIDER_AREA_RATIO * biggest:
                other_boxes.append(person["box"])
            else:
                counts[idx] += 1
                rider_boxes[idx].append(person["box"])

    return counts, rider_boxes, other_boxes


def _head_in_person(center, person_box):
    """True if a head centre lies in the upper part of a person box."""
    px1, py1, px2, py2 = person_box
    ph = py2 - py1

    return (
        px1 <= center[0] <= px2
        and py1 - HEAD_TOP_ALLOWANCE * ph <= center[1] <= py1 + HEAD_ZONE * ph
    )


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

    If ``people`` is given, a rider claims the head nearest THEIR OWN head
    keypoint (more exact than the top-centre of the body box, which can pick a
    neighbour's head, e.g. a child in front of the rider).
    """
    owners = []
    claimed = set()
    keypoint_of = {tuple(p["box"]): p.get("head") for p in (people or [])}

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
            own_head = keypoint_of.get(tuple(body))

            if own_head is not None:
                target_x, target_y = own_head

            def distance(item):
                cx, cy = box_center(item[1]["box"])
                return (cx - target_x) ** 2 + (cy - target_y) ** 2

            i, det = min(candidates, key=distance)
            claimed.add(i)
            owners.append((det, idx, False))

            if claims is not None:
                claims.append((body, det))

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

            # a hidden pillion sits right next to a rider, not far away ...
            riders = rider_boxes.get(idx, [])

            if riders and not any(_near_rider(center, b) for b in riders):
                continue

            # ... and its head is about as big as the riders' heads
            widths = sorted(
                d["box"][2] - d["box"][0]
                for d, owner, extra in owners if owner == idx and not extra
            )

            if widths:
                reference = widths[len(widths) // 2]
                width = det["box"][2] - det["box"][0]

                if not (HIDDEN_HEAD_SIZE_MIN * reference <= width
                        <= HIDDEN_HEAD_SIZE_MAX * reference):
                    continue

            mx, my = box_center(moto["box"])
            dist = (mx - center[0]) ** 2 + (my - center[1]) ** 2

            if dist < best_dist:
                best_dist, best_idx = dist, idx

        if best_idx is not None:
            owners.append((det, best_idx, True))

    return owners

def unresolved_rider_heads(riders, people, head_boxes):
    """
    Riders whose head the helmet model never found.

    ``riders`` are the person boxes riding one bike, ``people`` the person dicts
    (to look up the head keypoint) and ``head_boxes`` the head boxes already
    matched to that bike. A rider counts as unresolved when no head box has its
    centre in the upper part of the rider's body. Returns an estimated head box
    for each of them, so the picture can show "helmet unknown" instead of
    leaving that rider unlabelled.
    """
    by_box = {tuple(round(v, 1) for v in p["box"]): p for p in people}
    found = []

    for rider in riders:
        x1, y1, x2, y2 = rider
        width, height = x2 - x1, y2 - y1

        if width <= 0 or height <= 0:
            continue

        covered = any(
            x1 <= (h[0] + h[2]) / 2 <= x2 and y1 <= (h[1] + h[3]) / 2 <= y1 + 0.55 * height
            for h in head_boxes
        )

        if covered:
            continue

        person = by_box.get(tuple(round(v, 1) for v in rider))
        head = person.get("head") if person else None
        size = max(0.38 * width, 0.12 * height)

        if head:
            cx, cy = head[0], head[1]
        else:
            cx, cy = (x1 + x2) / 2, y1 + 0.6 * size

        found.append([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2])

    return found


def merge_fullframe_heads(crop_heads, full_heads, regions, min_gain=0.05,
                          with_cls=0, with_min=0.50, without_max=0.75):
    """
    Combine helmet detections from the bike crops with those from the whole frame.

    * a whole-frame head outside every bike ``region`` is ignored (not a rider);
    * one the crops did not see at all is added;
    * one that matches a crop head (same place) with a DIFFERENT class replaces
      it only when it is clearly more confident (``min_gain``);
    * same place and same class: the crop's box is kept.
    """
    merged = list(crop_heads)

    for head in full_heads:
        cx, cy = box_center(head["box"])

        if not any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in regions):
            continue

        match = None

        for index, existing in enumerate(merged):
            ex, ey = box_center(existing["box"])

            if (iou(head["box"], existing["box"]) >= 0.3
                    or point_in_box((cx, cy), existing["box"])
                    or point_in_box((ex, ey), head["box"])):
                match = index
                break

        if match is None:
            merged.append(head)
        elif merged[match]["cls"] != head["cls"]:
            crop = merged[match]

            # A "without helmet" call is an accusation, so a whole-frame
            # "with helmet" that is reasonably sure beats a crop "without"
            # that is not very sure (dark helmets are often misread in crops).
            helmet_wins = (head["cls"] == with_cls and head["conf"] >= with_min
                           and crop["conf"] < without_max)

            if helmet_wins or head["conf"] >= crop["conf"] + min_gain:
                merged[match] = head

    return merged