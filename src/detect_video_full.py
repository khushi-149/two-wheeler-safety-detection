"""
Run the two-wheeler safety pipeline on video files or photos.

    python src/detect_video_full.py                    menu of videos in data/videos
    python src/detect_video_full.py clip.mp4 a.jpg     run these files directly
    python src/detect_video_full.py clip.mp4 --no-preview    fastest, no window
"""

from __future__ import annotations

import argparse
import logging
import sys

import settings as cfg
from video_menu import choose_videos_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect helmet violations and triple riding in videos or photos."
    )
    parser.add_argument(
        "paths", nargs="*",
        help="video / photo files to process (omit to choose from a menu)",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="do not open a preview window (fastest)",
    )
    parser.add_argument(
        "--no-plates", action="store_true",
        help="skip number plate reading",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="show detailed progress logging",
    )
    return parser


def run_one(pipeline, path: str, preview: bool) -> None:
    print(f"\nProcessing: {path}")

    if preview:
        print("Press Q in the video window to stop this video.")
    else:
        print("No preview window: processing at full speed.")

    try:
        summary = pipeline.run(path, show_window=preview)
    except RuntimeError as error:
        print(f"ERROR: {error}")
        return

    print("\nDetection completed!")
    print(f"Run ID: {summary['run_id']}")
    print(f"Frames processed: {summary['frame']}")
    print(f"Motorcycles tracked: {summary['motorcycles']}")
    print(f"Helmet violations (unique bikes): {summary['helmet_violations']}")
    print(f"Triple riding (unique bikes): {summary['triple_riding']}")
    print(f"Plates read: {summary['plates'] or 'none'}")
    print(f"Evidence images saved: {summary['evidence_saved']}")

    if summary.get("output_video"):
        print(f"Processed video: {summary['output_video']}")

    if summary.get("output_image"):
        print(f"Processed photo: {summary['output_image']}")

    print(f"Violation records: {cfg.DB_PATH}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Models are loaded only when there is something to process, so the menu
    # appears instantly.
    pipeline = None

    def get_pipeline():
        nonlocal pipeline

        if pipeline is None:
            try:
                from pipeline import ViolationPipeline
            except ImportError as error:
                raise SystemExit(
                    f"Missing dependency ({error.name}). "
                    "Run: pip install -r requirements.txt"
                )

            print("Loading models (first run only)...")
            pipeline = ViolationPipeline(use_plates=not args.no_plates)

        return pipeline

    try:
        if args.paths:
            for path in args.paths:
                run_one(get_pipeline(), path, preview=not args.no_preview)
            return 0

        # Menu mode: keep returning to the menu until the user quits
        while True:
            chosen = choose_videos_cli(str(cfg.VIDEO_DIR))

            if not chosen:
                print("Goodbye.")
                return 0

            if args.no_preview:
                preview = False
            else:
                answer = input(
                    "Show live preview window? [Y/n] "
                    "(n = fastest; watch the saved video afterwards): "
                ).strip().lower()
                preview = answer not in ("n", "no")

            for path in chosen:
                run_one(get_pipeline(), path, preview)

    except FileNotFoundError as error:   # missing model weights
        print(f"ERROR: {error}")
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130


if __name__ == "__main__":
    sys.exit(main())