import numpy as np

from drawing import DARK, WHITE, LabelPlacer, Style


def overlap(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def test_text_size_follows_the_picture_but_is_capped():
    video, photo, small = Style(1280, 720), Style(2252, 4000), Style(320, 240)

    assert 0.6 <= video.label_scale <= 0.8
    assert photo.label_scale < 3.0                  # was ~3.9 for a 4000 px photo
    assert small.label_scale >= 0.45                # never unreadably small


def test_chips_never_overlap_and_stay_inside_the_picture():
    frame = np.zeros((600, 800, 3), np.uint8)
    placer = LabelPlacer(frame.shape, Style(800, 600))

    rects = [placer.draw(frame, "NO HELMET", 700, 40, anchor="above") for _ in range(5)]
    rects += [placer.draw(frame, "TRIPLE RIDING", 790, 595, anchor="below") for _ in range(3)]

    for i, a in enumerate(rects):
        assert a[0] >= 0 and a[1] >= 0 and a[2] <= 800 and a[3] <= 600
        for b in rects[i + 1:]:
            assert not overlap(a, b)


def test_stack_puts_chips_in_a_column():
    frame = np.zeros((600, 800, 3), np.uint8)
    placer = LabelPlacer(frame.shape, Style(800, 600))

    rects = placer.stack(frame, [("#2 3 riders", WHITE, DARK),
                                 ("TRIPLE RIDING", WHITE, DARK),
                                 ("NO HELMET", WHITE, DARK)], 100, 300, anchor="below")

    assert [r[1] for r in rects] == sorted(r[1] for r in rects)     # top to bottom
    assert all(rects[i][3] <= rects[i + 1][1] + 8 for i in range(2))  # stacked tightly


def test_chip_actually_paints_pixels():
    frame = np.zeros((300, 400, 3), np.uint8)
    x1, y1, x2, y2 = LabelPlacer(frame.shape, Style(400, 300)).draw(
        frame, "HELLO", 50, 100, anchor="below", bg=(0, 0, 170))

    assert frame[y1:y2, x1:x2].sum() > 0