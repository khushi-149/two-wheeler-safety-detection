"""
Explain how the pipeline counts riders and helmets in ONE photo.

    python scripts/debug_photo.py data/test_images/triple_3.jpg

For every motorcycle it lists each person found (and whether they count as a
rider, with the reason if not), each head found (helmet / no helmet + confidence)
and which rider owns which head. Send me this output when a count looks wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

import settings as cfg  # noqa: E402
from assignment import (assign_heads_to_bikes, assign_people_to_bikes,  # noqa: E402
                        is_plausible_rider, person_seat_point)
from geometry import box_center, point_in_box  # noqa: E402
from rider_association import head_point  # noqa: E402


def _fmt(box):
    return "(%d,%d)-(%d,%d)" % tuple(int(v) for v in box)


def describe(pipeline, frame, bikes):
    """Return the report as a list of text lines. ``bikes``: [{"box", "conf"}]."""
    pipeline.photo_mode = True
    people, heads = pipeline._analyze_crops(frame, bikes)
    counts, rider_boxes, others = assign_people_to_bikes(people, bikes)
    claims = []
    owners = assign_heads_to_bikes(heads, bikes, rider_boxes, others, claims, people)
    # same second look the dashboard does (zoomed head crops)
    owners = pipeline._refine_heads(frame, people, rider_boxes, owners, claims,
                                    list(range(len(bikes))))

    names = {pipeline.helmet_class: "With Helmet", pipeline.no_helmet_class: "Without Helmet",
             -1: "HEAD NOT FOUND (amber marker)"}
    lines = [f"Photo {frame.shape[1]}x{frame.shape[0]}: {len(bikes)} motorcycle(s), "
             f"{len(people)} people, {len(heads)} heads\n"]

    for i, bike in enumerate(bikes):
        lines.append(f"MOTORCYCLE {i}  box {_fmt(bike['box'])}   -> {counts.get(i, 0)} rider(s) "
                     f"(+{sum(1 for _, b, e in owners if b == i and e)} hidden)")

        for person in people:
            seat = person_seat_point(person)
            near = point_in_box(seat, [bike["box"][0] - 200, bike["box"][1] - 200,
                                       bike["box"][2] + 200, bike["box"][3] + 200])

            if not near:
                continue

            is_rider = tuple(person["box"]) in {tuple(b) for b in rider_boxes.get(i, [])}
            why = "RIDER" if is_rider else (
                "not a rider (" + ("implausible: far above / too small / no overlap"
                                   if not is_plausible_rider(person, bike)
                                   else "seat not on this bike") + ")")
            hx, hy = head_point(person)
            lines.append(f"    person {_fmt(person['box'])} head=({int(hx)},{int(hy)}) "
                         f"seat=({int(seat[0])},{int(seat[1])})  {why}")

        for det, owner, extra in owners:
            if owner == i:
                cx, cy = box_center(det["box"])
                lines.append(f"    head {_fmt(det['box'])} {names.get(det['cls'], det['cls'])} "
                             f"conf {det['conf']:.2f}" + ("  [hidden pillion]" if extra else ""))

    unowned = [d for d in heads if not any(d is o[0] for o in owners)]

    for det in unowned:
        lines.append(f"  ignored head {_fmt(det['box'])} {names.get(det['cls'], det['cls'])} "
                     f"conf {det['conf']:.2f}  (not on any bike)")

    return lines


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1

    path = Path(sys.argv[1])
    frame = cv2.imread(str(path))

    if frame is None:
        print(f"ERROR: cannot read {path.resolve()}")
        return 1

    from pipeline import ViolationPipeline
    from ultralytics import YOLO

    pipeline = ViolationPipeline(use_plates=False)
    traffic = YOLO(pipeline.traffic_path)
    result = traffic(frame, conf=cfg.MOTORCYCLE_CONF,
                     classes=[cfg.COCO_MOTORCYCLE_CLASS], verbose=False)[0]
    bikes = [{"box": b.xyxy[0].cpu().tolist(), "conf": float(b.conf[0])} for b in result.boxes]

    print("\n".join(describe(pipeline, frame, bikes)))
    return 0


if __name__ == "__main__":
    sys.exit(main())