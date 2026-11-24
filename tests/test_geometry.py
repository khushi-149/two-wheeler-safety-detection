import pytest

from geometry import box_area, box_center, ioa, iou, point_in_box


def test_area_and_center():
    assert box_area((0, 0, 10, 5)) == 50
    assert box_area((10, 10, 0, 0)) == 0          # inverted box
    assert box_center((0, 0, 10, 20)) == (5, 10)


def test_point_in_box_includes_edges():
    assert point_in_box((5, 5), (0, 0, 10, 10))
    assert point_in_box((10, 10), (0, 0, 10, 10))
    assert not point_in_box((11, 5), (0, 0, 10, 10))


def test_iou_values():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou_degenerate_boxes_do_not_divide_by_zero():
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_ioa_uses_smaller_box():
    assert ioa((0, 0, 100, 100), (10, 10, 20, 20)) == pytest.approx(1.0)
    assert ioa((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0