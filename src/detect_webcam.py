"""
Live detection from a webcam or stream.

    python src/detect_webcam.py                 -> menu of detected cameras
    python src/detect_webcam.py 0               -> camera 0 directly
    python src/detect_webcam.py http://IP:8080/video   -> network stream
    add --record to also save the annotated video

Press Q in the preview window to stop.
"""

import sys

from video_menu import choose_camera_cli


def main():

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    record = "--record" in sys.argv

    if args:
        source = int(args[0]) if args[0].isdigit() else args[0]
    else:
        source = choose_camera_cli()

        if source is None:
            print("Goodbye.")
            return

    from pipeline import ViolationPipeline

    print("Loading models...")
    pipeline = ViolationPipeline(use_plates=True)

    print(f"\nStarting live detection on source: {source}")
    print("Press Q in the video window to stop.")

    summary = pipeline.run(
        source,
        show_window=True,
        live=True,
        save_video=record,
    )

    print("\nLive session finished.")
    print(f"Frames processed: {summary['frame']}")
    print(f"Motorcycles tracked: {summary['motorcycles']}")
    print(f"Helmet violations (unique bikes): {summary['helmet_violations']}")
    print(f"Triple riding (unique bikes): {summary['triple_riding']}")
    print(f"Plates read: {summary['plates']}")
    print(f"Evidence images saved: {summary['evidence_saved']}")

    if summary["output_video"]:
        print(f"Recording: {summary['output_video']}")

    print("Violation records: outputs/violations.db")


if __name__ == "__main__":
    main()