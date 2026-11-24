"""Find test videos in data/videos and let the user choose from a menu."""

from __future__ import annotations

import os

import cv2

from settings import VIDEO_DIR, VIDEO_EXTENSIONS


def _ask(prompt: str) -> str:
    """input() that treats Ctrl+D / closed stdin as 'quit'."""
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "q"


def list_videos(folder=VIDEO_DIR):
    """All video files under `folder` (including subfolders), sorted."""
    videos = []

    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(VIDEO_EXTENSIONS):
                videos.append(os.path.join(root, name))

    return sorted(videos)


def describe_video(path):
    """Short description: resolution, duration, file size."""
    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return "unreadable"

    info = ""

    cap = cv2.VideoCapture(path)

    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frames / fps if fps > 0 else 0
        info = f"{w}x{h}, {duration:.0f}s, "

    cap.release()

    return f"{info}{size_mb:.1f} MB"


def choose_videos_cli(folder=VIDEO_DIR):
    """
    Show a numbered menu. Returns a list of chosen video paths,
    or an empty list if the user quits.

    Accepts: a number (2), several numbers (1,3,4 or "1 3 4"),
    'a' for all videos, 'q' to quit.
    """
    folder = str(folder)
    videos = list_videos(folder)

    if not videos:
        print(f"\nNo videos found in '{folder}'. Add .mp4/.avi/.mov files there.")
        return []

    while True:

        print("\n=== Available test videos ===")

        for number, path in enumerate(videos, start=1):
            shown = os.path.relpath(path, folder)
            print(f"  {number:>2}. {shown}  ({describe_video(path)})")

        print("   a. Run ALL videos one after another")
        print("   q. Quit")

        answer = _ask("\nChoose a video (number, e.g. 1,3 for several): ")

        if answer in ("q", "quit", "exit"):
            return []

        if answer in ("a", "all"):
            return videos

        parts = answer.replace(",", " ").split()

        if parts and all(p.isdigit() for p in parts):

            numbers = [int(p) for p in parts]

            if all(1 <= n <= len(videos) for n in numbers):
                return [videos[n - 1] for n in numbers]

        print(f"Invalid choice. Enter numbers between 1 and {len(videos)}, 'a' or 'q'.")


def find_cameras(max_index: int = 5) -> list[int]:
    """Return the indexes of cameras that actually deliver frames."""
    found = []

    for index in range(max_index):

        if os.name == "nt":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)

        ok = False

        if cap.isOpened():
            ok, _ = cap.read()

        cap.release()

        if ok:
            found.append(index)

    return found


def choose_camera_cli():
    """
    Menu of detected cameras. Returns a camera index (int), a stream URL
    (str), or None to quit. A phone can be used as a camera through an
    IP-camera app: choose 'u' and paste its URL.
    """
    print("\nSearching for cameras (a few seconds)...")
    cameras = find_cameras()

    while True:

        print("\n=== Available cameras ===")

        if cameras:
            for index in cameras:
                print(f"  {index}. Camera {index}")
        else:
            print("  (no local camera found)")

        print("  u. Enter a stream URL (phone IP-camera app, RTSP, etc.)")
        print("  q. Quit")

        answer = _ask("\nChoose a camera: ")

        if answer in ("q", "quit", "exit"):
            return None

        if answer in ("u", "url"):
            try:
                url = input("Stream URL: ").strip()  # case matters in URLs
            except EOFError:
                return None

            if url:
                return url

            continue

        if answer.isdigit() and int(answer) in cameras:
            return int(answer)

        print("Invalid choice.")