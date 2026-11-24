"""
Train the helmet detector locally (the project's published weights were
trained on Google Colab, see notebooks/helmet_detection_training.ipynb).

    python src/train_helmet.py                       defaults: 30 epochs, 640 px
    python src/train_helmet.py --epochs 50 --device 0

Dataset: data/helmet_dataset/data.yaml (see data/README.md).
The best weights are written to runs/helmet/<name>/weights/best.pt; copy them
to models/helmet_detector.pt to use them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import settings as cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the YOLO helmet detector.")
    parser.add_argument("--data", default=str(cfg.DATA_DIR / "helmet_dataset" / "data.yaml"))
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0,
                        help="data-loader workers (0 is safest on Windows)")
    parser.add_argument("--device", default=None,
                        help="e.g. 0 for the first GPU, or cpu (default: auto)")
    parser.add_argument("--project", default=str(cfg.PROJECT_ROOT / "runs" / "helmet"))
    parser.add_argument("--name", default="helmet_detector")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    data = Path(args.data)

    if not data.exists():
        print(f"ERROR: dataset config not found: {data}\n"
              "Download the dataset first (see data/README.md).")
        return 1

    from ultralytics import YOLO

    model = YOLO(args.base_model)

    extra = {} if args.device is None else {"device": args.device}

    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        **extra,
    )

    print("Helmet model training completed!")
    print(f"Best weights: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())