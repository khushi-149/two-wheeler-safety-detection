"""
Explain, stage by stage, how the REAL pipeline counts riders and helmets on one
photo, so a wrong answer can be traced to the stage that caused it.

    python scripts/explain_photo.py path/to/photo.jpg

Stages printed: bikes found -> people (pose) -> who rides which bike ->
heads (helmet model) -> which head belongs to which rider.
Paste the whole output when asking for help.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

import settings as cfg  # noqa: E402
from assignment import (  # noqa: E402
    assign_heads_to_bikes, assign_people_to_bikes, bike_region, dedupe_nested_bikes,
)
from pipeline import ViolationPipeline  # noqa: E402
from rider_tracking import PhotoMotorcycleDetector  # noqa: E402


def fmt(box):
    return "[" + ", ".join(f"{v:.0f}" for v in box) + "]"


def save_debug_image(frame, path, bikes, raw_people, kept_people, raw_helmets, names):
    """Draw EVERYTHING the models found (also what the filters removed)."""
    img = frame.copy()
    thick = max(2, int(min(img.shape[:2]) / 350))
    scale = max(0.6, min(img.shape[:2]) / 900)

    def tag(text, x, y, color):
        cv2.putText(img, text, (int(x), int(max(y, 12 * scale))), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, max(1, thick - 1), cv2.LINE_AA)

    for i, b in enumerate(bikes):                       # yellow = motorcycles
        x1, y1, x2, y2 = map(int, b["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), thick)
        tag(f"BIKE {i} {b['conf']:.2f}", x1 + 4, y2 - 8, (0, 220, 255))

    for i, p in enumerate(raw_people):                  # blue = kept person, grey = removed
        kept = any(p is q for q in kept_people)
        color = (255, 120, 0) if kept else (150, 150, 150)
        x1, y1, x2, y2 = map(int, p["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        tag(f"P{i} {p['conf']:.2f}{'' if kept else ' REMOVED'}", x1 + 4, y1 + 22 * scale, color)

        if p.get("head"):
            cv2.circle(img, tuple(int(v) for v in p["head"]), thick * 3, (0, 255, 255), -1)

        if p.get("hip"):
            cv2.circle(img, tuple(int(v) for v in p["hip"]), thick * 3, (255, 0, 255), -1)

    for i, h in enumerate(raw_helmets):                 # green = With Helmet, red = Without
        color = (0, 200, 0) if names.get(h["cls"]) == "With Helmet" else (0, 0, 255)
        x1, y1, x2, y2 = map(int, h["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, thick - 1))
        tag(f"H{i} {h['conf']:.2f}", x1, y1 - 6, color)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def main(path: str) -> int:
    frame = cv2.imread(path)

    if frame is None:
        print(f"Cannot read image: {path}")
        return 1

    pipe = ViolationPipeline(use_plates=False)
    pipe.photo_mode = True   # exactly like the dashboard: larger crops for photos
    traffic = YOLO(cfg.model_path(cfg.PHOTO_TRAFFIC_MODEL))

    print(f"Image {frame.shape[1]}x{frame.shape[0]}   traffic model: {cfg.PHOTO_TRAFFIC_MODEL}")
    print(f"thresholds: moto={cfg.MOTORCYCLE_CONF} pose={cfg.POSE_CONF} helmet={cfg.HELMET_CONF}")

    detector = PhotoMotorcycleDetector(traffic, confidence=cfg.MOTORCYCLE_CONF)
    bikes = dedupe_nested_bikes(
        [{"box": m["box"], "conf": m["conf"], "track_id": m["track_id"]}
         for m in detector.update(frame)])   # same as the real pipeline

    print(f"\n1. MOTORCYCLES: {len(bikes)}  (found on attempt {detector.used_attempt})")

    for i, b in enumerate(bikes):
        print(f"   bike {i}: box {fmt(b['box'])} conf {b['conf']:.2f}")

    if not bikes:
        print("   -> no bike found: nothing else can work. Try a bigger traffic model.")
        return 0

    people, helmets = pipe._analyze_crops(frame, bikes)

    crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in
             (bike_region(b["box"], frame.shape[1], frame.shape[0]) for b in bikes)]
    print(f"   (crops analysed at {pipe._crop_imgsz(crops)} px)")

    print(f"\n2. PEOPLE (pose, after duplicate removal): {len(people)}")

    for i, p in enumerate(people):
        print(f"   person {i}: box {fmt(p['box'])} conf {p['conf']:.2f} "
              f"head {p['head'] and tuple(round(v) for v in p['head'])} "
              f"face-points {p.get('face')} "
              f"hip {p['hip'] and tuple(round(v) for v in p['hip'])}")

    counts, rider_boxes, other_boxes = assign_people_to_bikes(people, bikes)

    print("\n3. RIDERS PER BIKE:", {i: n for i, n in counts.items()})
    print(f"   people not on any bike: {len(other_boxes)}")

    names = {pipe.helmet_class: "With Helmet", pipe.no_helmet_class: "Without Helmet"}
    print(f"\n4. HEADS (helmet model): {len(helmets)}")

    for i, h in enumerate(helmets):
        print(f"   head {i}: box {fmt(h['box'])} conf {h['conf']:.2f} {names.get(h['cls'], h['cls'])}")

    claims = []
    owners = assign_heads_to_bikes(helmets, bikes, rider_boxes, other_boxes, claims,
                                   people=people)

    print("\n5. HEAD OWNERS (bike, extra rider?):")

    for det, bike, extra in owners:
        print(f"   {names.get(det['cls'], det['cls'])} {det['conf']:.2f} -> bike {bike}"
              f"{'  (hidden extra rider)' if extra else ''}")

    total = {i: counts[i] + sum(1 for _, b, e in owners if b == i and e) for i in counts}

    # the dashboard's second look at each head (zoomed crop)
    pipe.debug_refine, pipe.refine_log = True, []
    owners = pipe._refine_heads(frame, people, rider_boxes, owners, claims,
                                due=list(range(len(bikes))))

    print("\n5b. RE-CHECK of each head (zoomed look): orig = main pass, look = what the")
    print("    zoomed crop found, verdict = replacement chosen (None = keep the original):")

    for entry in pipe.refine_log:
        orig = entry["orig"] and (names.get(entry["orig"][0], entry["orig"][0]),
                                  round(entry["orig"][1], 2))
        look = [(names.get(c, c), conf) for c, conf in entry["look"]]
        verdict = entry["verdict"] and (names.get(entry["verdict"][0]), entry["verdict"][1])
        print(f"   bike {entry['bike']} {entry['kind']}: orig {orig} look {look} -> {verdict}")

    print("\n   FINAL helmet status per head (what the dashboard shows):")

    for det, bike, extra in owners:
        print(f"   bike {bike}: {names.get(det['cls'], det['cls'])} {det['conf']:.2f}"
              f"{'  (hidden pillion)' if extra else ''}")
    print("\nFINAL rider count per bike (riders + hidden pillions):", total)
    print("Compare with the photo: which stage first disagrees with what you see?")

    debug = getattr(pipe, "last_debug", {})
    out = cfg.OUTPUTS_DIR / "explain" / (Path(path).stem + "_explain.jpg")
    save_debug_image(frame, out, bikes, debug.get("raw_people", people), people,
                     debug.get("raw_helmets", helmets), names)
    print(f"\nPicture of everything the models found: {out}")
    print("  yellow = bike, blue = person kept, grey = person REMOVED by a filter,")
    print("  yellow dot = head point, pink dot = hip point, green/red = helmet boxes")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)

    sys.exit(main(sys.argv[1]))