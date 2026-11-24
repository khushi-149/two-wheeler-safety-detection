"""Annotation helpers: colours, resolution-aware sizes and non-overlapping labels."""

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
BLUE = (255, 120, 0)
DARK = (25, 25, 25)
DARK_RED = (0, 0, 170)
AMBER = (0, 140, 255)


class Style:
    """
    Annotation sizes that scale with the picture.

    Text is sized as a share of the LONGER picture side (about 1.2 %), so a
    1280 px video frame and a 4000 px photo both get text of the same relative
    size, instead of the 4000 px photo getting enormous labels.
    """

    def __init__(self, width, height=None):
        longest = max(width, height or width)
        s = min(3.2, max(0.6, longest / 1000.0))

        self.label_scale = max(0.45, longest * 0.012 / 22.0)
        self.label_thickness = max(1, int(round(1.6 * s)))
        self.tag_scale = self.label_scale * 0.85
        self.warning_scale = self.label_scale * 1.25
        self.warning_thickness = max(2, int(round(2.2 * s)))
        self.box = max(2, int(round(2.4 * s)))
        self.thin = max(1, int(round(1.6 * s)))
        self.pad = max(3, int(round(4 * s)))
        self.gap = max(3, int(round(5 * s)))


def _text_box(style, text, scale, thickness):
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    return tw + 2 * style.pad, th + baseline + 2 * style.pad, th, baseline


class LabelPlacer:
    """
    Draws text chips and remembers where they went, so labels never cover
    each other and never leave the picture (no more cut-off "ELMET").
    """

    def __init__(self, frame_shape, style):
        self.h, self.w = frame_shape[:2]
        self.style = style
        self.taken = []                      # drawn rectangles (x1, y1, x2, y2)

    def _free(self, rect):
        x1, y1, x2, y2 = rect

        if x1 < 0 or y1 < 0 or x2 > self.w or y2 > self.h:
            return False

        return all(x2 <= a or x1 >= c or y2 <= b or y1 >= d
                   for a, b, c, d in self.taken)

    def draw(self, frame, text, x, y, anchor="above", color=WHITE, bg=DARK,
             scale=None, thickness=None, height_ref=None):
        """
        Draw ``text`` as a chip next to the point (x, y).

        anchor="above": the chip's bottom-left sits just above y.
        anchor="below": the chip's top-left sits just below y.
        Tries the preferred side first, then the other side, then slides
        sideways / further away until the chip is free. If nothing is free
        the preferred spot (clamped into the picture) is used.
        Returns the chip rectangle.
        """
        st = self.style
        scale = scale or st.label_scale
        thickness = thickness or st.label_thickness
        cw, ch, th, baseline = _text_box(st, text, scale, thickness)
        cw = min(cw, self.w)

        x = int(max(0, min(x, self.w - cw)))
        order = (-1, 1) if anchor == "above" else (1, -1)
        step = ch + st.gap

        chosen = None

        for rounds in range(0, 6):
            for direction in order:
                if direction < 0:
                    y1 = int(y - st.gap - ch - rounds * step)
                else:
                    y1 = int(y + st.gap + rounds * step)

                y1 = max(0, min(y1, self.h - ch))
                rect = (x, y1, x + cw, y1 + ch)

                if self._free(rect):
                    chosen = rect
                    break

            if chosen:
                break

        if chosen is None:
            y1 = int(max(0, min(y - st.gap - ch if anchor == "above" else y + st.gap,
                                self.h - ch)))
            chosen = (x, y1, x + cw, y1 + ch)

        x1, y1, x2, y2 = chosen
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
        cv2.putText(frame, text, (x1 + st.pad, y1 + st.pad + th), FONT, scale,
                    color, thickness, cv2.LINE_AA)

        self.taken.append(chosen)
        return chosen

    def stack(self, frame, lines, x, y, anchor="below", scale=None):
        """
        Draw several chips in a column that starts at (x, y).
        ``lines`` is a list of (text, text_colour, background_colour).
        """
        rects = []
        cursor = y

        for text, color, bg in lines:
            rect = self.draw(frame, text, x, cursor, anchor=anchor,
                             color=color, bg=bg, scale=scale)
            rects.append(rect)
            # next chip starts right below this one (or above, when stacking upwards)
            cursor = rect[3] - self.style.gap + 1 if anchor == "below" else rect[1] + self.style.gap - 1

        return rects