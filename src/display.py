"""
Smooth, aligned live preview for video files and cameras.

Problem: detection is slower than the video, so drawing the LATEST detections on
the NEWEST video frame leaves the boxes behind the riders (they show where
the riders were a moment ago).

Fix: the preview shows the video slightly DELAYED, and every shown frame gets the
detections that were made on (or next to) that same frame. The video plays at
its normal speed and the boxes stay glued to the riders. The delay adapts to
the time one detection takes on this computer.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import cv2
import numpy as np

log = logging.getLogger(__name__)


class OverlayDisplay:

    def __init__(self, cap, callback, size, fps, aligned=True,
                 min_delay=0.35, max_delay=2.0, clock=time.perf_counter):
        self.cap = cap
        self.callback = callback
        self.size = size                     # (width, height) of the preview
        self.fps = max(1.0, float(fps))
        self.aligned = aligned
        self.min_delay, self.max_delay = min_delay, max_delay
        self.clock = clock

        keep = int((max_delay + 0.5) * self.fps) + 2 if aligned else 2
        self.history = deque(maxlen=keep)    # (arrival time, seq, small frame)
        self.overlays = deque(maxlen=keep)   # (seq, mask, pixels)
        self.max_gap = max(3, int(self.fps * 0.8))

        self.delay = min_delay if aligned else 0.0
        self.target_delay = self.delay
        self._latency = None
        self._last_seq = -1
        self._shown_seq = -1
        self._last_step = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    # ---- called by the detection loop ----

    def push_overlay(self, seq, mask, pixels):
        with self._lock:
            self.overlays.append((seq, mask, pixels))

    def report_latency(self, seconds):
        """Seconds one frame took from arrival to finished detection."""
        self._latency = (seconds if self._latency is None
                         else 0.8 * self._latency + 0.2 * seconds)
        wanted = 1.3 * self._latency + 0.15
        self.target_delay = min(self.max_delay, max(self.min_delay, wanted))

    # ---- display side ----

    def _ingest(self, now):
        frame, seq = self.cap.peek()

        if frame is None or seq == self._last_seq:
            return

        self._last_seq = seq
        small = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)

        with self._lock:
            self.history.append((now, seq, small))

    def _ease_delay(self, now):
        if self._last_step is not None:
            limit = 0.15 * (now - self._last_step)      # at most 15 % speed change
            change = self.target_delay - self.delay
            self.delay += max(-limit, min(limit, change))

        self._last_step = now

    def _pick(self, now):
        with self._lock:
            if not self.history:
                return None

            if not self.aligned:
                return self.history[-1]

            wanted = now - self.delay
            chosen = None

            for entry in self.history:
                if entry[0] <= wanted:
                    chosen = entry
                else:
                    break

            return chosen

    def _overlay_for(self, seq):
        with self._lock:
            if not self.overlays:
                return None

            if not self.aligned:
                return self.overlays[-1]

            best = min(self.overlays, key=lambda o: abs(o[0] - seq))

        return best if abs(best[0] - seq) <= self.max_gap else None

    def compose(self, entry):
        _, seq, small = entry
        image = small.copy()
        overlay = self._overlay_for(seq)

        if overlay is not None:
            _, mask, pixels = overlay
            np.copyto(image, pixels, where=mask)

        return image

    def step(self, now=None):
        """Show the next frame if one is due. Returns True when a frame was shown."""
        now = self.clock() if now is None else now

        self._ingest(now)
        self._ease_delay(now)
        entry = self._pick(now)

        if entry is None or entry[1] == self._shown_seq:
            return False

        self._shown_seq = entry[1]
        self.callback(self.compose(entry))
        return True

    # ---- thread ----

    def _loop(self):
        while not self._stop.is_set():
            try:
                shown = self.step()
            except Exception as error:      # never let the preview kill the run
                log.warning("Display error: %s", error)
                shown = False
                time.sleep(0.05)

            if not shown:
                time.sleep(0.004)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)