"""
Check that this machine is ready to run the project.

    python scripts/check_setup.py            quick check (files + packages)
    python scripts/check_setup.py --load     also load every model (slower)

Replaces the old throw-away scripts test_yolo.py and test_tracking.py.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import settings  # noqa: E402

REQUIRED = ["cv2", "numpy", "pandas", "ultralytics", "streamlit"]
OPTIONAL = {
    "easyocr": "number plate reading",
    "imageio_ffmpeg": "in-browser playback of processed videos",
}


def check_packages() -> bool:
    ok = True
    print("Python packages")

    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            print(f"  OK       {name} {getattr(module, '__version__', '')}")
        except ImportError:
            ok = False
            print(f"  MISSING  {name}   -> pip install -r requirements.txt")

    for name, purpose in OPTIONAL.items():
        try:
            importlib.import_module(name)
            print(f"  OK       {name} (optional: {purpose})")
        except ImportError:
            print(f"  skipped  {name} (optional: {purpose})")

    return ok


def check_files() -> bool:
    print("\nModel files")
    helmet = Path(settings.model_path(settings.HELMET_MODEL))
    plate = Path(settings.model_path(settings.PLATE_MODEL))

    ok = helmet.exists()
    print(f"  {'OK      ' if ok else 'MISSING '} helmet model  {helmet}")

    if not ok:
        print("           download helmet_detector.pt from the GitHub Releases "
              "page into models/")

    print(f"  {'OK      ' if plate.exists() else 'optional'} plate model   {plate}"
          + ("" if plate.exists() else "  (plate reading stays off)"))

    print("\nFolders")
    for folder in (settings.VIDEO_DIR, settings.IMAGE_DIR, settings.OUTPUTS_DIR):
        print(f"  {'OK      ' if folder.exists() else 'will be created'} {folder}")

    return ok


def load_models() -> bool:
    print("\nLoading models")
    try:
        from ultralytics import YOLO

        for label, name in (("traffic", settings.TRAFFIC_MODEL),
                            ("pose", settings.POSE_MODEL),
                            ("helmet", settings.HELMET_MODEL)):
            model = YOLO(settings.model_path(name))
            print(f"  OK       {label}: {len(model.names)} classes -> {model.names}")
        return True
    except Exception as error:  # report any failure, this is a diagnostic tool
        print(f"  FAILED   {error}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--load", action="store_true", help="also load every model")
    args = parser.parse_args()

    results = [check_packages(), check_files()]

    if args.load:
        results.append(load_models())

    print("\nAll good." if all(results) else "\nSome checks failed (see above).")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())