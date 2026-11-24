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
    """Annotation sizes that scale with the video height."""

    def __init__(self, height):
        s = max(0.6, height / 720)
        self.label_scale = 0.7 * s
        self.label_thickness = max(2, int(2 * s))
        self.warning_scale = 1.0 * s
        self.warning_thickness = max(2, int(3 * s))
        self.box = max(2, int(3 * s))
        self.thin = max(1, int(2 * s))
        self.pad = max(6, int(6 * s))
        self.warning_line = int(65 * s)
        self.warning_start = int(55 * s)


# -------------------- Drawing helpers --------------------

def draw_label(frame, style, text, x, y, color=WHITE, bg=DARK,
               font_scale=None, thickness=None):

    font_scale = font_scale or style.label_scale
    thickness = thickness or style.label_thickness
    pad = style.pad

    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)

    x = int(max(0, min(x, frame.shape[1] - tw - 2 * pad)))
    y = int(max(th + 2 * pad, y))

    cv2.rectangle(frame, (x, y - th - pad),
                  (x + tw + 2 * pad, y + baseline + pad // 2), bg, -1)

    cv2.putText(frame, text, (x + pad, y), FONT, font_scale,
                color, thickness, cv2.LINE_AA)


def draw_warning(frame, style, text, y):
    draw_label(frame, style, text, 20, y, WHITE, bg=(0, 0, 200),
               font_scale=style.warning_scale,
               thickness=style.warning_thickness)