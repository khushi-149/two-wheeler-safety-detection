from rider_association import (
    associate_riders, count_riders, dedupe_motorcycles, dedupe_people,
    detect_triple_riding, head_from_keypoints, head_point,
)


def person(x1, y1, x2, y2, conf=0.9, head=None):
    return {"box": [x1, y1, x2, y2], "conf": conf, "head": head}


def test_head_from_keypoints_averages_confident_points():
    pts = [[10, 10], [12, 8], [8, 8], [0, 0], [0, 0]]
    conf = [0.9, 0.9, 0.9, 0.1, 0.1]
    assert head_from_keypoints(pts, conf) == (10.0, 26 / 3)


def test_head_from_keypoints_none_when_unreliable():
    assert head_from_keypoints(None) is None
    assert head_from_keypoints([[1, 1]] * 5, [0.1] * 5) is None


def test_head_from_keypoints_ignores_non_finite():
    assert head_from_keypoints([[float("nan"), 1]], [0.9]) is None


def test_head_point_falls_back_to_box_top():
    x, y = head_point(person(0, 0, 100, 200))
    assert x == 50 and y == 16


def test_dedupe_keeps_most_confident():
    kept = dedupe_motorcycles([
        {"box": [0, 0, 100, 100], "conf": 0.5},
        {"box": [2, 2, 101, 101], "conf": 0.9},
    ])
    assert len(kept) == 1 and kept[0]["conf"] == 0.9


def test_dedupe_people_keeps_two_overlapping_riders():
    a = person(0, 0, 100, 200, 0.9)
    b = person(30, 0, 130, 200, 0.8)          # IoU well below 0.85
    assert len(dedupe_people([a, b])) == 2


def test_dedupe_people_removes_near_identical():
    a = person(0, 0, 100, 200, 0.9)
    b = person(1, 1, 100, 200, 0.8)
    assert len(dedupe_people([a, b])) == 1


def test_associate_and_triple_riding():
    moto = {"box": [100, 300, 300, 450], "conf": 0.9}
    riders = [person(110 + 30 * i, 100, 200 + 30 * i, 400) for i in range(3)]

    assoc = associate_riders(riders, [moto])
    counts = count_riders(assoc)

    assert counts == {0: 3}
    assert detect_triple_riding(counts) == {0: "TRIPLE RIDING"}


def test_two_riders_are_ok():
    assert detect_triple_riding({0: 2, 1: 0}) == {0: "OK", 1: "OK"}


def test_pedestrian_far_from_bike_not_associated():
    moto = {"box": [100, 300, 300, 450], "conf": 0.9}
    walker = person(600, 100, 700, 400)
    assert count_riders(associate_riders([walker], [moto])) == {0: 0}