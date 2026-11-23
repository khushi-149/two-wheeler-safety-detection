"""
Number plate detection + OCR for tracked motorcycles.

Two modes:
  1. Detector mode (recommended): a YOLO plate detector at
     models/plate_detector.pt finds the plate inside the motorcycle box,
     then EasyOCR reads the cropped plate.
  2. OCR-only fallback (no plate model): EasyOCR scans the lower half of the
     motorcycle box and only accepts text that matches the Indian plate format.
"""

import os
import queue
import re
import threading

import cv2

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Indian format: 2 letters, 2 digits, 1-3 letters, 4 digits  (e.g. HR26DK8337)
PLATE_REGEX = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$")

# Common OCR confusions, applied by character position
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
            "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S",
             "8": "B", "6": "G", "7": "T"}


def _fix(chars, mapping):
    return "".join(mapping.get(c, c) for c in chars)


def normalize_plate(raw):
    """Clean OCR text and fix position-based confusions. Returns (text, is_valid)."""
    text = re.sub(r"[^A-Z0-9]", "", raw.upper())

    if not 9 <= len(text) <= 11:
        return text, False

    series_len = len(text) - 8
    fixed = (
        _fix(text[:2], TO_LETTER)
        + _fix(text[2:4], TO_DIGIT)
        + _fix(text[4:4 + series_len], TO_LETTER)
        + _fix(text[-4:], TO_DIGIT)
    )

    return fixed, bool(PLATE_REGEX.match(fixed))


def _preprocess(crop):
    h, w = crop.shape[:2]
    scale = min(4.0, max(1.0, 200.0 / max(w, 1)))

    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _assemble(results):
    """Join EasyOCR fragments. Handles two-line plates (common on two-wheelers)."""
    items = []

    for bbox, text, conf in results:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        items.append({
            "text": text,
            "conf": float(conf),
            "yc": sum(ys) / len(ys),
            "h": max(ys) - min(ys),
            "x": min(xs),
        })

    if not items:
        return "", 0.0

    avg_h = sum(i["h"] for i in items) / len(items)
    items.sort(key=lambda i: i["yc"])

    rows = [[items[0]]]
    for item in items[1:]:
        if abs(item["yc"] - rows[-1][-1]["yc"]) < 0.6 * avg_h:
            rows[-1].append(item)
        else:
            rows.append([item])

    text = "".join(
        "".join(i["text"] for i in sorted(row, key=lambda i: i["x"]))
        for row in rows
    )

    total_chars = sum(len(i["text"]) for i in items) or 1
    conf = sum(i["conf"] * len(i["text"]) for i in items) / total_chars

    return text, conf


class PlateReader:

    def __init__(self, plate_model_path="models/plate_detector.pt",
                 use_gpu=None, plate_conf=0.30):

        import easyocr  # imported here so the rest of the project works without it

        if use_gpu is None:
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except ImportError:
                use_gpu = False

        print(f"Loading EasyOCR (GPU={use_gpu}). The first run downloads "
              "its models, which can take a few minutes...")

        self.ocr = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        self.plate_conf = plate_conf
        self.plate_model = None

        if os.path.exists(plate_model_path):
            from ultralytics import YOLO
            self.plate_model = YOLO(plate_model_path)
            print(f"Plate detector loaded: {plate_model_path}")
        else:
            print(
                f"Warning: {plate_model_path} not found. "
                "Using OCR-only fallback (lower accuracy)."
            )

    # ---------------------------------------------------------------

    def read(self, frame, moto_box):
        """
        Try to read the plate of one motorcycle.
        Returns dict {text, conf, valid, box, crop} or None.
        box is in full-frame coordinates (or None in fallback mode).
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, moto_box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 40 or y2 - y1 < 40:
            return None

        region = frame[y1:y2, x1:x2]

        if self.plate_model is not None:
            return self._read_with_detector(region, x1, y1)

        return self._read_ocr_only(region)

    # ---------------------------------------------------------------

    def _read_with_detector(self, region, ox, oy):

        result = self.plate_model(
            region, conf=self.plate_conf, verbose=False
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None

        best = max(result.boxes, key=lambda b: float(b.conf[0]))
        px1, py1, px2, py2 = map(int, best.xyxy[0].cpu().tolist())

        # small padding helps OCR
        pad_x = int(0.05 * (px2 - px1))
        pad_y = int(0.10 * (py2 - py1))
        rh, rw = region.shape[:2]

        px1, py1 = max(0, px1 - pad_x), max(0, py1 - pad_y)
        px2, py2 = min(rw, px2 + pad_x), min(rh, py2 + pad_y)

        crop = region[py1:py2, px1:px2]

        if crop.size == 0:
            return None

        ocr_result = self._ocr(crop)

        if ocr_result is None:
            return None

        ocr_result["box"] = (ox + px1, oy + py1, ox + px2, oy + py2)
        ocr_result["crop"] = crop

        return ocr_result

    def _read_ocr_only(self, region):

        rh = region.shape[0]
        lower = region[int(rh * 0.5):, :]

        if lower.size == 0:
            return None

        ocr_result = self._ocr(lower)

        # Without a detector we only trust fully valid plates
        if ocr_result is None or not ocr_result["valid"]:
            return None

        ocr_result["box"] = None
        ocr_result["crop"] = lower

        return ocr_result

    def _ocr(self, crop):

        gray = _preprocess(crop)

        results = self.ocr.readtext(
            gray, allowlist=ALLOWLIST, detail=1, paragraph=False
        )

        raw_text, conf = _assemble(results)
        text, valid = normalize_plate(raw_text)

        if len(text) < 6:
            return None

        return {"text": text, "conf": conf, "valid": valid}


class PlateTracker:
    """Keeps the best plate reading per motorcycle track ID."""

    def __init__(self, max_attempts=10, interval=8, good_conf=0.60):
        self.max_attempts = max_attempts
        self.interval = interval
        self.good_conf = good_conf
        self.attempts = {}
        self.plates = {}  # track_id -> result dict

    def should_try(self, track_id, frame_idx):
        if self.attempts.get(track_id, 0) >= self.max_attempts:
            return False

        if frame_idx % self.interval != 0:
            return False

        best = self.plates.get(track_id)

        if best and best["valid"] and best["conf"] >= self.good_conf:
            return False  # already have a confident reading

        return True

    def record_attempt(self, track_id):
        self.attempts[track_id] = self.attempts.get(track_id, 0) + 1

    def update(self, track_id, result):
        """Store result if better. Returns True if it improved the record."""
        current = self.plates.get(track_id)

        new_score = (result["valid"], result["conf"])
        old_score = (current["valid"], current["conf"]) if current else None

        if old_score is None or new_score > old_score:
            self.plates[track_id] = result
            return True

        return False

    def best(self, track_id):
        return self.plates.get(track_id)


class AsyncPlateReader:
    """
    Runs PlateReader in a background thread so slow OCR never freezes the
    video loop. Submit a job, keep processing frames, poll for results later.
    """

    def __init__(self, reader, max_queue=2):
        self.reader = reader
        self.jobs = queue.Queue(maxsize=max_queue)
        self.results = queue.Queue()
        self.pending = set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def is_pending(self, track_id):
        return track_id in self.pending

    def submit(self, track_id, frame, moto_box):
        """Queue a read for one motorcycle. Returns False if busy/invalid."""
        if track_id in self.pending:
            return False

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, moto_box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 40 or y2 - y1 < 40:
            return False

        region = frame[y1:y2, x1:x2].copy()

        try:
            self.jobs.put_nowait((track_id, region, (x1, y1)))
        except queue.Full:
            return False

        self.pending.add(track_id)
        return True

    def poll(self):
        """Return finished reads as a list of (track_id, result_or_None)."""
        done = []

        while True:
            try:
                track_id, result = self.results.get_nowait()
            except queue.Empty:
                break

            self.pending.discard(track_id)
            done.append((track_id, result))

        return done

    def close(self, timeout=10.0):
        """
        Stop the worker. Reads that have not started yet are dropped; the
        read in progress is allowed to finish (up to `timeout` seconds) so
        its result can still be collected with poll().
        """
        while True:
            try:
                job = self.jobs.get_nowait()
            except queue.Empty:
                break

            if job is not None:
                self.pending.discard(job[0])

        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass

        if self.pending and self._thread.is_alive():
            print("Finishing the plate read in progress...")

        self._thread.join(timeout=timeout)

    def _loop(self):
        while True:
            job = self.jobs.get()

            if job is None:
                break

            track_id, region, (ox, oy) = job

            try:
                rh, rw = region.shape[:2]
                result = self.reader.read(region, (0, 0, rw, rh))

                if result is not None and result["box"] is not None:
                    bx1, by1, bx2, by2 = result["box"]
                    result["box"] = (bx1 + ox, by1 + oy, bx2 + ox, by2 + oy)
            except Exception as error:  # never let the worker thread die
                print(f"Plate OCR error: {error}")
                result = None

            self.results.put((track_id, result))