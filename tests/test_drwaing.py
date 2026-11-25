"""Labels must fit inside the picture and must not sit on top of each other."""

import numpy as np

from drawing import Style, draw_label


def blank(w, h):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_long_label_is_shrunk_to_fit_width():
    frame = blank(400, 600)
    placed = []
    draw_label(frame, Style(600, 400), "ID 1 | RIDERS: 4 | TRIPLE RIDING | NO HELMET",
               350, 100, placed=placed)
    x1, _, x2, _ = placed[0]
    assert 0 <= x1 and x2 <= 400


def test_labels_at_same_spot_do_not_overlap():
    frame = blank(800, 800)
    style = Style(800, 800)
    placed = []

    for _ in range(4):
        draw_label(frame, style, "NO HELMET", 300, 300, placed=placed)

    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])


def test_multiline_label_taller_than_single_line():
    frame = blank(800, 800)
    placed = []
    draw_label(frame, Style(800, 800), "A", 10, 200, placed=placed)
    draw_label(frame, Style(800, 800), "A\nB", 400, 200, placed=placed)
    assert (placed[1][3] - placed[1][1]) > (placed[0][3] - placed[0][1])


def test_style_size_is_capped_for_huge_photos():
    assert Style(8000, 6000).label_scale <= 0.6 * 4.0 + 1e-9


def test_bottom_anchored_label_stays_inside_box_and_header_is_built():
    import numpy as np
    from drawing import Style, draw_label, make_header

    frame = np.full((600, 800, 3), 120, np.uint8)
    style = Style(600, 800)
    placed = []

    draw_label(frame, style, "ID 1\nRIDERS: 3", 20, 580, placed=placed, anchor="bottom")

    x1, y1, x2, y2 = placed[0]
    assert y2 <= 580 + 1 and y1 >= 0

    header = make_header(800, style, ["NO HELMET VIOLATION", "TRIPLE RIDING DETECTED"])
    assert header.shape[1] == 800 and header.shape[0] > 0
    assert make_header(800, style, []) is None


def test_head_tags_stay_next_to_their_own_head_and_do_not_overlap():
    import numpy as np
    from drawing import Style, draw_tag

    frame = np.full((900, 1000, 3), 120, np.uint8)
    style = Style(900, 1000)
    heads = [(560, 280, 680, 410), (525, 410, 625, 520)]
    placed = list(heads)

    for head in heads:
        draw_tag(frame, style, "NO HELMET", head, placed=placed)

    tags = placed[2:]
    assert len(tags) == 2

    for tag, head in zip(tags, heads):
        assert abs(tag[1] - head[1]) < 120 and abs(tag[0] - head[0]) < 120

    # no tag covers either head, and the tags do not cover each other
    for tag in tags:
        for head in heads:
            assert not (tag[0] < head[2] and head[0] < tag[2]
                        and tag[1] < head[3] and head[1] < tag[3])

    a, b = tags
    assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])