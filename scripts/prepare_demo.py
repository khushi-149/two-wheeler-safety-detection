"""
Analyse videos BEFORE a presentation, so the demo itself is instant.

    python scripts/prepare_demo.py data/videos/clip.mp4
    python scripts/prepare_demo.py data/videos/*.mp4 --stride 3

For every video this saves the annotated video (boxes, labels, plates) to
outputs/videos/, the evidence pictures to outputs/evidence/ and the violation
records to the database, so the dashboard's Results and History tabs show them
immediately. On the day, just play the saved video and open the dashboard.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import settings  # noqa: E402
from pipeline import ViolationPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("videos", nargs="+", help="video files to analyse")
    parser.add_argument("--stride", type=int, default=2,
                        help="analyse every Nth frame (1 = most thorough, default 2)")
    parser.add_argument("--no-plates", action="store_true",
                        help="skip number-plate reading (faster)")
    args = parser.parse_args()

    print("Loading models ...")
    pipe = ViolationPipeline(use_plates=not args.no_plates, db_path=settings.DB_PATH)

    for name in args.videos:
        path = Path(name)

        if not path.exists():
            print(f"  skipped (not found): {path}")
            continue

        started = time.perf_counter()

        def progress(_frame, stats, label=path.name):
            print(f"\r  {label}: frame {stats['frame']} / {stats['total_frames']}",
                  end="", flush=True)

        stats = pipe.run(str(path), frame_callback=progress, callback_every=5,
                         frame_stride=args.stride)
        seconds = time.perf_counter() - started

        print(f"\n  done in {seconds:.0f}s | helmet violations: "
              f"{stats['helmet_violations']} | triple riding: "
              f"{stats['triple_riding']} | plates: {stats['plates_read']}")
        print(f"  annotated video: {stats.get('output_video')}")

    print("\nReady. Play the annotated videos and open the dashboard's "
          "Results / History tabs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())