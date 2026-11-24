"""Video sources: a single photo, or a camera / stream read on a background thread."""

import logging
import os
import threading
import time

import cv2

from settings import IMAGE_EXTENSIONS  # noqa: F401  (re-exported)

log = logging.getLogger(__name__)


class ImageCapture:
    """Presents one photo as a one-frame 'video' so the same pipeline runs on it."""

    def __init__(self, path):
        self.image = cv2.imread(path)
        self._done = False

    def isOpened(self):
        return self.image is not None

    def get(self, prop):
        if self.image is None:
            return 0

        h, w = self.image.shape[:2]

        return {
            cv2.CAP_PROP_FRAME_WIDTH: w,
            cv2.CAP_PROP_FRAME_HEIGHT: h,
            cv2.CAP_PROP_FRAME_COUNT: 1,
        }.get(prop, 0)

    def read(self):
        if self._done or self.image is None:
            return False, None

        self._done = True
        return True, self.image.copy()

    def release(self):
        pass


# -------------------- Live camera capture --------------------

class ThreadedCapture:
    """
    Reads a camera / stream / video file in a background thread and keeps the
    NEWEST frame. Stale frames are dropped, so the view never lags behind even
    when detection is slower than the source.

    paced=True  plays a video FILE at its real speed, like a live camera
                (frames the detector cannot keep up with are simply skipped).
    autostart=False  open now, call start() later (so a file does not begin
                playing while models are still loading).

    read()  waits for a NEW frame (used by the detector).
    peek()  returns the newest frame without consuming it (used for display).
    """

    def __init__(self, source, width=1280, height=720,
                 paced=False, autostart=True):

        if isinstance(source, str) and source.lower().startswith("rtsp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        if isinstance(source, int) and os.name == "nt":
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # faster on Windows
        else:
            self.cap = cv2.VideoCapture(source)

        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.paced = paced
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._last_seq = 0
        self.last_read_seq = 0          # seq of the frame the last read() returned
        self._index = 0
        self._ended = False
        self._running = self.cap.isOpened()
        self._thread = None

        if autostart:
            self.start()

    def start(self):
        if self._thread is not None or not self._running:
            return

        target = self._read_paced if self.paced else self._read_live
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    # ---- background readers ----

    def _read_live(self):
        while self._running:
            ok, frame = self.cap.read()

            if not ok:
                time.sleep(0.01)
                continue

            with self._lock:
                self._frame = frame
                self._seq += 1

    def _read_paced(self):
        try:
            self._paced_loop()
        except cv2.error as error:
            log.warning("Video read error: %s", error)
        finally:
            self._ended = True

    def _paced_loop(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0 or fps > 120:
            fps = 25

        start = time.perf_counter()
        index = 0

        while self._running:

            wait = start + index / fps - time.perf_counter()

            if wait > 0:
                time.sleep(min(wait, 0.01))
                continue

            ok, frame = self.cap.read()

            if not ok:
                break

            index += 1

            # Decoding slower than real time: skip frames to catch up
            behind = int((time.perf_counter() - (start + index / fps)) * fps)

            for _ in range(max(0, behind)):
                if not self.cap.grab():
                    break
                index += 1

            with self._lock:
                self._frame = frame
                self._seq += 1
                self._index = index

    # ---- cv2.VideoCapture-like API ----

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)

    @property
    def frame_index(self):
        """Position (in source frames) of the newest frame."""
        return self._index

    @property
    def ended(self):
        return self._ended

    def peek(self):
        """Newest frame WITHOUT consuming it. Do not modify the returned array."""
        with self._lock:
            if self._frame is None:
                return None, 0

            return self._frame, self._seq

    def read(self, timeout=3.0):
        """Wait for a NEW frame. Returns (False, None) at the end / on a stall."""
        deadline = time.time() + timeout

        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._seq != self._last_seq:
                    self._last_seq = self._seq
                    self.last_read_seq = self._seq
                    return True, self._frame.copy()

                if self._ended:
                    return False, None

            time.sleep(0.003)

        return False, None

    def release(self):
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=3.0)

        self.cap.release()