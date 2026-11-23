"""
Run the two-wheeler safety pipeline from the command line.

    python src/detect_video_full.py                 -> shows a menu of videos in data/videos
    python src/detect_video_full.py path/to/a.mp4   -> runs that video directly
    add --no-preview to skip the window (fastest)
"""

import sys

from video_menu import choose_videos_cli

VIDEO_FOLDER = "data/videos"

pipeline = None  # loaded lazily so the menu appears instantly


def run_video(path, preview=True):
    global pipeline

    if pipeline is None:
        from pipeline import ViolationPipeline

        print("Loading models (first run only)...")
        pipeline = ViolationPipeline(use_plates=True)

    print(f"\nProcessing: {path}")

    if preview:
        print("Press Q in the video window to stop this video.")
    else:
        print("No preview window: processing at full speed.")

    summary = pipeline.run(path, show_window=preview)

    print("\nTracking and detection completed!")
    print(f"Run ID: {summary['run_id']}")
    print(f"Frames processed: {summary['frame']}")
    print(f"Motorcycles tracked: {summary['motorcycles']}")
    print(f"Helmet violations (unique bikes): {summary['helmet_violations']}")
    print(f"Triple riding (unique bikes): {summary['triple_riding']}")
    print(f"Plates read: {summary['plates']}")
    print(f"Evidence images saved: {summary['evidence_saved']}")
    print(f"Processed video: {summary['output_video']}")
    print("Violation records: outputs/violations.db")


def main():

    # Direct mode: paths given on the command line skip the menu
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]

    if paths:
        for path in paths:
            run_video(path, preview="--no-preview" not in sys.argv)
        return

    # Menu mode: keep returning to the menu until the user quits
    while True:

        chosen = choose_videos_cli(VIDEO_FOLDER)

        if not chosen:
            print("Goodbye.")
            break

        if "--no-preview" in sys.argv:
            preview = False
        else:
            answer = input(
                "Show live preview window? [Y/n] "
                "(n = fastest; watch the saved video afterwards): "
            ).strip().lower()

            preview = answer not in ("n", "no")

        for path in chosen:
            run_video(path, preview)


if __name__ == "__main__":
    main()