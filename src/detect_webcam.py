"""
Live detection from a webcam or a network stream.

    python src/detect_webcam.py                      menu of detected cameras
    python src/detect_webcam.py 0                    camera 0 directly
    python src/detect_webcam.py http://IP:8080/video network / phone camera
    add --record to also save the annotated video

Press Q in the preview window to stop.
"""

from __future__ import annotations

import argparse
import logging
import sys

import settings as cfg
from video_menu import choose_camera_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live helmet / triple-riding detection.")
    parser.add_argument(
        "source", nargs="?",
        help="camera index (0, 1, ...) or stream URL; omit to choose from a menu",
    )
    parser.add_argument("--record", action="store_true",
                        help="save the annotated video")
    parser.add_argument("--no-plates", action="store_true",
                        help="skip number plate reading")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show detailed progress logging")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.source is not None:
        source = int(args.source) if args.source.isdigit() else args.source
    else:
        source = choose_camera_cli()

        if source is None:
            print("Goodbye.")
            return 0

    try:
        from pipeline import ViolationPipeline, redact_source
    except ImportError as error:
        print(f"Missing dependency ({error.name}). Run: pip install -r requirements.txt")
        return 1

    try:
        print("Loading models...")
        pipeline = ViolationPipeline(use_plates=not args.no_plates)

        print(f"\nStarting live detection on: {redact_source(source)}")
        print("Press Q in the video window to stop.")

        summary = pipeline.run(
            source, show_window=True, live=True, save_video=args.record
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}")
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130

    print("\nLive session finished.")
    print(f"Frames processed: {summary['frame']}")
    print(f"Motorcycles tracked: {summary['motorcycles']}")
    print(f"Helmet violations (unique bikes): {summary['helmet_violations']}")
    print(f"Triple riding (unique bikes): {summary['triple_riding']}")
    print(f"Plates read: {summary['plates'] or 'none'}")
    print(f"Evidence images saved: {summary['evidence_saved']}")

    if summary.get("output_video"):
        print(f"Recording: {summary['output_video']}")

    print(f"Violation records: {cfg.DB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())