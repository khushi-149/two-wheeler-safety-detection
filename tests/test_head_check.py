import numpy as np
import pytest

from fakes import Box, Boxes, Result
from head_check import head_crop_detections, verify_decision

FRAME = np.zeros((480, 640, 3), np.uint8)
PERSON = {"box": [100, 100, 200, 400], "head": (150, 130)}     # head-crop radius = 42


class CropModel:
    """Returns one box per given (x, y, w, h) fraction of the crop it receives."""

    def __init__(self, items):
        self.items, self.calls, self.crop_shapes = items, 0, []

    def __call__(self, crop, **kwargs):
        self.calls += 1
        h, w = crop.shape[:2]
        self.crop_shapes.append((h, w))
        boxes = [Box([w * fx, h * fy, w * (fx + fw), h * (fy + fh)], conf, cls)
                 for (fx, fy, fw, fh, conf, cls) in self.items]
        return [Result(Boxes(boxes))]


def test_crop_result_is_mapped_back_to_frame_coordinates():
    model = CropModel([(0.3, 0.3, 0.4, 0.4, 0.6, 1)])          # box in the middle of the crop

    found = head_crop_detections(model, FRAME, PERSON, conf=0.1, imgsz=320)

    assert len(found) == 1 and found[0]["cls"] == 1 and found[0]["conf"] == pytest.approx(0.6)
    x1, y1, x2, y2 = found[0]["box"]
    assert ((x1 + x2) / 2, (y1 + y2) / 2) == pytest.approx((150, 130), abs=1.5)
    assert x2 - x1 == pytest.approx(0.4 * 84, abs=1.5)         # 84 px crop -> real size, not upscaled size


def test_small_crops_are_upscaled_before_inference():
    model = CropModel([])
    head_crop_detections(model, FRAME, PERSON, conf=0.1, imgsz=320)

    assert min(model.crop_shapes[0]) >= 224


def test_neighbours_helmet_inside_the_crop_is_ignored():
    model = CropModel([(0.0, 0.0, 0.1, 0.1, 0.9, 0)])           # far corner of the crop
    assert head_crop_detections(model, FRAME, PERSON, conf=0.1, imgsz=320) == []


def test_head_outside_frame_gives_no_crop():
    person = {"box": [0, 0, 4, 4], "head": (-50, -50)}
    model = CropModel([])
    assert head_crop_detections(model, FRAME, person, conf=0.1, imgsz=320) == []
    assert model.calls == 0


def det(cls, conf):
    return {"box": [0, 0, 1, 1], "cls": cls, "conf": conf}


def test_verify_overrides_only_when_without_clearly_wins():
    assert verify_decision([det(1, 0.6), det(0, 0.3)], 0, 1, 0.30)["cls"] == 1
    assert verify_decision([det(1, 0.5), det(0, 0.7)], 0, 1, 0.30) is None      # helmet still stronger
    assert verify_decision([det(1, 0.2)], 0, 1, 0.30) is None                   # not sure enough
    assert verify_decision([], 0, 1, 0.30) is None