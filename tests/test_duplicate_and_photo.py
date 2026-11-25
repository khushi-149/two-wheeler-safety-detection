"""Duplicate-person merge and the photo motorcycle retry."""

from assignment import assign_people_to_bikes, merge_duplicate_people, same_person
from fakes import Box, Boxes, Result
from rider_tracking import PhotoMotorcycleDetector

BIKE = {"box": [200, 250, 400, 420], "conf": 0.9, "track_id": 1}


def person(box, head, hip, conf):
    return {"box": box, "head": head, "hip": hip, "conf": conf}


RIDER_A = person([215, 100, 285, 400], (250, 120), (250, 360), 0.90)
RIDER_B = person([300, 105, 370, 405], (335, 125), (335, 365), 0.85)


def count(people):
    counts, _, _ = assign_people_to_bikes(merge_duplicate_people(people), [BIKE])
    return counts[0]


def test_duplicate_of_one_rider_is_merged():
    for shift in (8, 15, 22):
        dup = person([215 + shift, 100, 285 + shift, 400],
                     (250 + shift // 2, 121), (250 + shift // 2, 360), 0.45)
        assert count([RIDER_A, RIDER_B, dup]) == 2, shift


def test_two_real_close_riders_are_kept():
    front = person([215, 100, 285, 400], (250, 120), (250, 360), 0.9)
    back = person([250, 130, 320, 400], (285, 150), (285, 360), 0.8)
    assert not same_person(front, back)
    assert count([front, back]) == 2


def test_three_real_riders_still_counted():
    third = person([380, 110, 450, 405], (415, 130), (390, 365), 0.8)
    assert len(merge_duplicate_people([RIDER_A, RIDER_B, third])) == 3


def test_higher_confidence_person_is_kept():
    dup = person([223, 100, 293, 400], (254, 121), (254, 360), 0.45)
    kept = merge_duplicate_people([dup, RIDER_A])
    assert kept == [RIDER_A]


class ScriptedDetector:
    """Finds a motorcycle only once conf <= 0.2, or imgsz is large."""

    def __init__(self, min_conf=None, needs_imgsz=False):
        self.min_conf, self.needs_imgsz = min_conf, needs_imgsz
        self.calls = []

    def __call__(self, frame, **kwargs):
        self.calls.append(kwargs)
        ok = (
            (self.min_conf is None or kwargs["conf"] <= self.min_conf)
            and (not self.needs_imgsz or "imgsz" in kwargs)
        )
        boxes = [Box([10, 10, 100, 100], 0.3, 3)] if ok else []
        return [Result(Boxes(boxes))]


def test_photo_detector_normal_confidence_first():
    model = ScriptedDetector()
    bikes = PhotoMotorcycleDetector(model, confidence=0.25).update(object())
    assert len(model.calls) == 1 and bikes[0]["track_id"] == 1


def test_photo_detector_retries_with_lower_confidence():
    model = ScriptedDetector(min_conf=0.10)
    detector = PhotoMotorcycleDetector(model, confidence=0.25)
    assert len(detector.update(object())) == 1
    assert detector.used_attempt == 1


def test_photo_detector_retries_with_larger_image():
    model = ScriptedDetector(min_conf=0.10, needs_imgsz=True)
    detector = PhotoMotorcycleDetector(model, confidence=0.25)
    assert len(detector.update(object())) == 1
    assert detector.used_attempt == 2


def test_photo_detector_gives_up_when_nothing_found():
    model = ScriptedDetector(min_conf=0.01)
    assert PhotoMotorcycleDetector(model).update(object()) == []
    assert len(model.calls) == 3