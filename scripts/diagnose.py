"""
Show what each model sees in a video or photo, to find out WHY nothing is detected.

    python scripts/diagnose.py data/videos/clip.mp4
    python scripts/diagnose.py photo.jpg --samples 12
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

import settings  # noqa: E402


def sample_frames(path: Path, samples: int):
    """Yield (frame_number, frame) spread evenly over the video (or the single photo)."""
    if path.suffix.lower() in settings.IMAGE_EXTENSIONS:
        frame = cv2.imread(str(path))
        if frame is not None:
            yield 0, frame
        return

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, total // max(1, samples)) if total else 30
    number = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if number % step == 0:
            yield number, frame

        number += 1

    cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("source", help="video or photo")
    parser.add_argument("--samples", type=int, default=8, help="frames to test")
    parser.add_argument("--helmet-conf", type=float, default=settings.HELMET_CONF,
                        help=f"helmet confidence to test (settings.py value: {settings.HELMET_CONF})")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="image size for the helmet model, e.g. 1280 (bigger sees small heads)")
    args = parser.parse_args()

    path = Path(args.source)

    if not path.exists():
        print(f"ERROR: file not found: {path.resolve()}")
        return 1

    frames = list(sample_frames(path, args.samples))

    if not frames:
        print(f"ERROR: OpenCV could not read any frame from {path}.\n"
              "The file may be corrupt or use a codec OpenCV cannot decode. "
              "Try converting it to .mp4 (H.264).")
        return 1

    height, width = frames[0][1].shape[:2]
    print(f"Read {len(frames)} sample frames, size {width}x{height}\n")

    from ultralytics import YOLO

    helmet_file = settings.model_path(settings.HELMET_MODEL)

    if not Path(helmet_file).exists():
        print(f"ERROR: helmet model missing: {helmet_file}")
        return 1

    traffic = YOLO(settings.model_path(settings.TRAFFIC_MODEL))
    pose = YOLO(settings.model_path(settings.POSE_MODEL))
    helmet = YOLO(helmet_file)

    print(f"Helmet model classes: {helmet.names}")
    print(f"Testing helmet confidence = {args.helmet_conf}, "
          f"image size = {args.imgsz or 'default (640)'}")
    print(f"(settings.py currently has HELMET_CONF = {settings.HELMET_CONF}, "
          f"CROP_IMGSZ = {settings.CROP_IMGSZ})")

    from pipeline import resolve_helmet_classes

    with_id, without_id = resolve_helmet_classes(helmet.names)
    print(f"  -> 'with helmet' = class {with_id}, 'without helmet' = class {without_id}\n")

    total_bikes = total_people = total_heads = 0

    print(f"{'frame':>6} | {'motorcycles':>11} | {'people':>6} | helmet-model detections")
    print("-" * 70)

    for number, frame in frames:
        bikes = traffic(frame, conf=settings.MOTORCYCLE_CONF,
                        classes=[settings.COCO_MOTORCYCLE_CLASS], verbose=False)[0]
        everything = traffic(frame, conf=settings.MOTORCYCLE_CONF, verbose=False)[0]
        people = pose(frame, conf=settings.POSE_CONF, verbose=False)[0]
        extra = {} if args.imgsz is None else {"imgsz": args.imgsz}
        heads = helmet(frame, conf=args.helmet_conf, verbose=False, **extra)[0]

        n_bikes = len(bikes.boxes)
        n_people = len(people.boxes)
        head_counts = Counter(helmet.names[int(b.cls[0])] for b in heads.boxes)

        total_bikes += n_bikes
        total_people += n_people
        total_heads += sum(head_counts.values())

        others = Counter(traffic.names[int(b.cls[0])] for b in everything.boxes)
        print(f"{number:>6} | {n_bikes:>11} | {n_people:>6} | "
              f"{dict(head_counts) or 'none'}   (traffic model sees: {dict(others) or 'nothing'})")

        for b in heads.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            print(f"         head: {helmet.names[int(b.cls[0])]:<15} "
                  f"conf {float(b.conf[0]):.2f}  box ({x1},{y1})-({x2},{y2})")

    print("\nDIAGNOSIS")

    if total_bikes == 0:
        print("- NO motorcycles found by the traffic model. Nothing can be checked without a "
              "motorcycle, so no violations are recorded.\n"
              "  Try a clearer / closer video, or a bigger model: set TRAFFIC_MODEL = "
              "\"yolo11s.pt\" (or yolo11m.pt) in src/settings.py.")
    else:
        print(f"- Motorcycles found ({total_bikes}): the traffic model works.")

    if total_people == 0:
        print("- NO people found by the pose model, so riders cannot be counted.")
    else:
        print(f"- People found ({total_people}): the pose model works.")

    if total_heads == 0:
        print("- The helmet model found NOTHING at confidence "
              f"{args.helmet_conf}. Try --helmet-conf 0.1 and/or --imgsz 1280.")
    else:
        print(f"- Helmet detections found ({total_heads}): the helmet model works.")

    if total_bikes and total_people and total_heads:
        print("\nAll three models work on this footage. If the dashboard still shows "
              "nothing, the problem is in the dashboard run, so send me what it displays "
              "and any red text in the terminal.")

    return 0


if __name__ == "__main__":
    sys.exit(main())