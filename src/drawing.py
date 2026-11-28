"""Annotation helpers: colours, resolution-aware sizes and text labels."""

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
BLUE = (255, 120, 0)
DARK = (25, 25, 25)


class Style:
    """Annotation sizes that scale with the picture size (smaller side, capped)."""

    def __init__(self, height, width=None):
        side = min(height, width) if width else height
        s = min(4.0, max(0.6, side / 600))
        self.label_scale = 0.6 * s
        self.label_thickness = max(1, int(round(1.5 * s)))
        self.warning_scale = 0.8 * s
        self.warning_thickness = max(2, int(round(2 * s)))
        self.box = max(2, int(3 * s))
        self.thin = max(1, int(2 * s))
        self.pad = max(4, int(5 * s))
        self.warning_line = int(52 * s)
        self.warning_start = int(45 * s)


# -------------------- Drawing helpers --------------------

def _overlaps(rect, others):
    x1, y1, x2, y2 = rect
    return any(x1 < ox2 and ox1 < x2 and y1 < oy2 and oy1 < y2
               for ox1, oy1, ox2, oy2 in others)


def draw_label(frame, style, text, x, y, color=WHITE, bg=DARK,
               font_scale=None, thickness=None, placed=None, anchor="top"):
    """
    Draw a text label with a background box.

    * ``text`` may contain newlines (one line per row).
    * The font shrinks if the text is wider than the picture, so it is never cut off.
    * ``anchor="top"``: the label sits just above ``y``. ``anchor="bottom"``:
      the label's bottom edge is at ``y`` (used to tuck a label inside a box).
    * ``placed`` is a list of rectangles already used on this frame; the label
      moves up (then down) until it no longer covers one, and adds its own
      rectangle to the list.
    """
    font_scale = font_scale or style.label_scale
    thickness = thickness or style.label_thickness
    pad = style.pad
    frame_h, frame_w = frame.shape[:2]
    lines = str(text).split("\n")

    def measure(scale):
        sizes = [cv2.getTextSize(line, FONT, scale, thickness) for line in lines]
        width = max(size[0][0] for size in sizes)
        line_h = max(size[0][1] for size in sizes)
        return width, line_h, sizes[0][1]

    tw, th, baseline = measure(font_scale)
    room = frame_w - 2 * pad - 4

    if tw > room > 0:
        font_scale = max(0.3, font_scale * room / tw)
        tw, th, baseline = measure(font_scale)

    row = th + baseline + pad // 2
    box_w = tw + 2 * pad
    box_h = row * len(lines) + pad

    x = int(max(0, min(x, frame_w - box_w)))
    if anchor == "bottom":
        top = int(y) - box_h
    else:
        top = int(y) - th - pad      # text baseline at y

    def clamp(value):
        return int(max(0, min(value, frame_h - box_h)))

    top = clamp(top)
    step = box_h + 2

    if placed is not None:
        candidates = [clamp(top - k * step) for k in range(0, 8)]
        candidates += [clamp(top + k * step) for k in range(1, 8)]

        for candidate in candidates:
            if not _overlaps((x, candidate, x + box_w, candidate + box_h), placed):
                top = candidate
                break

        placed.append((x, top, x + box_w, top + box_h))

    cv2.rectangle(frame, (x, top), (x + box_w, top + box_h), bg, -1)

    for number, line in enumerate(lines):
        base_y = top + pad // 2 + th + number * row
        cv2.putText(frame, line, (x + pad, base_y), FONT, font_scale,
                    color, thickness, cv2.LINE_AA)


def draw_tag(frame, style, text, box, color=WHITE, bg=DARK, font_scale=None,
             placed=None, avoid=None):
    """
    Draw a short label attached to ``box`` (a head, say): directly above it if
    that spot is free, otherwise directly below, to the right, to the left, or
    inside its top edge. ``avoid`` holds rectangles that must stay uncovered
    (other faces); ``placed`` holds labels already drawn.
    """
    font_scale = font_scale or style.label_scale * 0.8
    thickness = style.label_thickness
    pad = max(3, style.pad - 1)
    frame_h, frame_w = frame.shape[:2]

    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    w = tw + 2 * pad
    h = th + baseline + pad

    x1, y1, x2, y2 = (int(v) for v in box)
    gap = 2

    def clamp(x, y):
        return (int(max(0, min(x, frame_w - w))), int(max(0, min(y, frame_h - h))))

    candidates = [
        clamp(x1, y1 - h - gap),        # above, left-aligned
        clamp(x2 - w, y1 - h - gap),    # above, right-aligned
        clamp(x1, y2 + gap),            # below
        clamp(x2 - w, y2 + gap),
        clamp(x2 + gap, y1),            # right
        clamp(x1 - w - gap, y1),        # left
        clamp(x1, y1 + gap),            # inside the top edge
    ]
    own = (x1, y1, x2, y2)
    blocked = [tuple(r) for r in (placed or []) if tuple(r) != own]
    blocked += [tuple(a) for a in (avoid or []) if tuple(a) != own]

    pos = candidates[0]

    for cx, cy in candidates:
        if not _overlaps((cx, cy, cx + w, cy + h), blocked):
            pos = (cx, cy)
            break

    x, y = pos
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    cv2.putText(frame, text, (x + pad, y + pad // 2 + th), FONT, font_scale,
                color, thickness, cv2.LINE_AA)

    if placed is not None:
        placed.append((x, y, x + w, y + h))


def draw_warning(frame, style, text, y, placed=None):
    draw_label(frame, style, text, 10, y, WHITE, bg=(0, 0, 200),
               font_scale=style.warning_scale,
               thickness=style.warning_thickness, placed=placed)


def make_header(width, style, texts, colors=None):
    """
    A solid banner strip (for photos) holding the violation warnings, so they
    sit above the picture instead of covering it. Returns a BGR image
    ``width`` wide, or ``None`` when there is nothing to show.
    """
    import numpy as np

    if not texts:
        return None

    scale = style.warning_scale
    thick = style.warning_thickness
    pad = style.pad * 2
    sizes = [cv2.getTextSize(t, FONT, scale, thick) for t in texts]

    text_h = max(size[0][1] for size in sizes)
    row_h = text_h + sizes[0][1] + pad
    bar = np.full((row_h * len(texts), width, 3), DARK, dtype=np.uint8)

    for number, (text, size) in enumerate(zip(texts, sizes)):
        colour = (colors[number] if colors else (0, 0, 200))
        top = number * row_h
        cv2.rectangle(bar, (0, top), (width, top + row_h - 2), colour, -1)

        fit = scale
        if size[0][0] > width - 2 * pad:
            fit = max(0.3, scale * (width - 2 * pad) / size[0][0])

        cv2.putText(bar, text, (pad, top + pad // 2 + text_h + 2), FONT, fit,
                    WHITE, thick, cv2.LINE_AA)

    return bar