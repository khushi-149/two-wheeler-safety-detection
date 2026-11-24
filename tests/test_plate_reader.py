import time

import numpy as np
import pytest

from plate_reader import AsyncPlateReader, PlateTracker, normalize_plate


@pytest.mark.parametrize("raw, expected, valid", [
    ("HR26DK8337", "HR26DK8337", True),
    ("hr 26 dk 8337", "HR26DK8337", True),          # case and spaces
    ("HR26DK833T", "HR26DK8337", True),             # letter read where a digit belongs
    ("H026DK8337", "HO26DK8337", True),             # zero read where a letter belongs
    ("MH12A1234", "MH12A1234", True),               # one-letter series
    ("MH12ABC1234", "MH12ABC1234", True),           # three-letter series
])
def test_normalize_valid_plates(raw, expected, valid):
    text, ok = normalize_plate(raw)
    assert text == expected and ok is valid


@pytest.mark.parametrize("raw", ["", "AB", "12345678901234", "HELLOWORLD"])
def test_normalize_rejects_garbage(raw):
    assert normalize_plate(raw)[1] is False


def good(conf=0.9, valid=True, text="HR26DK8337"):
    return {"text": text, "conf": conf, "valid": valid}


def test_tracker_keeps_best_reading():
    t = PlateTracker()
    assert t.update(1, good(0.4, valid=False)) is True
    assert t.update(1, good(0.3, valid=True)) is True      # valid beats invalid
    assert t.update(1, good(0.2, valid=True)) is False     # lower confidence
    assert t.best(1)["conf"] == 0.3


def test_tracker_stops_after_confident_read_and_max_attempts():
    t = PlateTracker(max_attempts=2, interval=1)
    assert t.should_try(1, 10)

    t.update(1, good(0.95))
    assert not t.should_try(1, 11)                          # already confident

    t.record_attempt(2)
    t.record_attempt(2)
    assert not t.should_try(2, 12)                          # attempts used up


def test_tracker_respects_interval():
    t = PlateTracker(interval=8)
    assert t.should_try(1, 16) and not t.should_try(1, 17)


class FakeReader:
    def __init__(self, result=None, error=None, delay=0.0):
        self.result, self.error, self.delay = result, error, delay

    def read(self, frame, box):
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


def wait_for(worker, seconds=3.0):
    end = time.time() + seconds
    while time.time() < end:
        done = worker.poll()
        if done:
            return done
        time.sleep(0.01)
    return []


FRAME = np.zeros((200, 300, 3), np.uint8)


def test_async_reader_returns_result():
    worker = AsyncPlateReader(FakeReader({"box": None, "text": "X", "conf": 1.0, "valid": True}))
    try:
        assert worker.submit(7, FRAME, (10, 10, 150, 150)) is True
        assert worker.is_pending(7)
        track_id, result = wait_for(worker)[0]
        assert track_id == 7 and result["text"] == "X"
        assert not worker.is_pending(7)
    finally:
        worker.close()


def test_async_reader_survives_reader_exception():
    worker = AsyncPlateReader(FakeReader(error=ValueError("boom")))
    try:
        worker.submit(1, FRAME, (10, 10, 150, 150))
        assert wait_for(worker) == [(1, None)]
        worker.submit(2, FRAME, (10, 10, 150, 150))          # thread is still alive
        assert wait_for(worker) == [(2, None)]
    finally:
        worker.close()


def test_async_reader_rejects_tiny_boxes_and_duplicates():
    worker = AsyncPlateReader(FakeReader(None, delay=0.3))
    try:
        assert worker.submit(1, FRAME, (0, 0, 10, 10)) is False      # too small
        assert worker.submit(2, FRAME, (10, 10, 150, 150)) is True
        assert worker.submit(2, FRAME, (10, 10, 150, 150)) is False  # already pending
    finally:
        worker.close()


def test_async_reader_close_is_prompt():
    worker = AsyncPlateReader(FakeReader(None))
    started = time.time()
    worker.close()
    assert time.time() - started < 2


# ------------------------------------------------ plate search / OCR variants

from fakes import Box, Boxes, Result      # noqa: E402
from plate_reader import (PlateReader, plausible_plate_shape,   # noqa: E402
                          search_region)


def test_search_region_extends_below_the_bike_and_stays_in_frame():
    x1, y1, x2, y2 = search_region((1000, 1000, 3), (100, 100, 300, 400))
    assert (x1, y1, x2) == (80, 100, 320) and y2 == 460

    assert search_region((420, 310, 3), (100, 100, 300, 400)) == (80, 100, 310, 420)


def test_implausible_plate_shapes_are_rejected():
    assert plausible_plate_shape(120, 30) and plausible_plate_shape(60, 45)
    assert not plausible_plate_shape(10, 10)            # speck
    assert not plausible_plate_shape(300, 20)           # 15:1 stripe
    assert not plausible_plate_shape(30, 100)           # tall sliver


class FakePlateModel:
    def __init__(self, boxes):
        self.boxes, self.kwargs = boxes, None

    def __call__(self, region, **kwargs):
        self.kwargs = kwargs
        return [Result(Boxes(self.boxes))]


class ScriptedOCR:
    """Returns the next scripted EasyOCR answer per call; remembers how often it ran."""

    def __init__(self, texts):
        self.texts, self.calls = list(texts), 0

    def readtext(self, image, **kwargs):
        text, conf = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return [([[0, 0], [100, 0], [100, 30], [0, 30]], text, conf)] if text else []


def make_reader(model, ocr):
    reader = PlateReader.__new__(PlateReader)           # skip EasyOCR / YOLO loading
    reader.ocr, reader.plate_model, reader.plate_conf = ocr, model, 0.3
    return reader


PLATE_FRAME = np.zeros((600, 800, 3), np.uint8)


def test_second_ocr_variant_is_used_when_the_first_fails():
    model = FakePlateModel([Box([100, 300, 220, 340], 0.9, 0)])
    ocr = ScriptedOCR([("", 0.0), ("HR26DK8337", 0.8)])

    result = make_reader(model, ocr).read(PLATE_FRAME, (250, 200, 500, 520))

    assert result["text"] == "HR26DK8337" and result["valid"] and ocr.calls == 2


def test_best_candidate_wins_and_junk_boxes_are_skipped():
    model = FakePlateModel([
        Box([10, 10, 20, 20], 0.95, 0),                  # speck: skipped despite top score
        Box([100, 300, 220, 340], 0.60, 0),
    ])
    ocr = ScriptedOCR([("MH12A1234", 0.9)])

    result = make_reader(model, ocr).read(PLATE_FRAME, (250, 200, 500, 520))

    assert result["text"] == "MH12A1234"
    assert result["box"][0] >= 225 + 90                        # full-frame coordinates, not the speck


def test_detector_runs_at_a_size_that_fits_the_search_area():
    model = FakePlateModel([])
    big = np.zeros((2000, 3000, 3), np.uint8)

    make_reader(model, ScriptedOCR([("", 0)])).read(big, (500, 300, 2500, 1800))

    assert model.kwargs["imgsz"] == 1280                  # capped