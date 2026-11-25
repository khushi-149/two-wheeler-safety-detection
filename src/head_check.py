"""
Second look at a rider's head with the helmet model on a zoomed crop.

* ``head_crop_detections`` finds heads the main helmet pass missed.
* ``verify_decision`` decides whether a doubtful "With Helmet" should be
  overridden by the zoomed-in result.
"""

from __future__ import annotations

import math

import cv2

from rider_association import head_point


def head_crop_detections(helmet_model, frame, person, conf, imgsz, around=None):
    """
    Run the helmet model on a small crop around the person's head (upscaled
    so it is not tiny). Returns detections in FULL-FRAME coordinates:
    ``[{"box", "conf", "cls"}, ...]``, only those centred on this head.
    """
    px1, py1, px2, py2 = person["box"]
    width, height = px2 - px1, py2 - py1

    if around is not None:
        # look at THIS head (a detected head box), not at the person's own head
        # keypoint: with merged people the two can be different heads
        ax1, ay1, ax2, ay2 = around
        hx, hy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
        radius = max(ax2 - ax1, ay2 - ay1, 20) * 0.75
    else:
        hx, hy = head_point(person)
        radius = max(width * 0.40, height * 0.14, 20)

    img_h, img_w = frame.shape[:2]

    cx1, cy1 = int(max(0, hx - radius)), int(max(0, hy - radius))
    cx2, cy2 = int(min(img_w, hx + radius)), int(min(img_h, hy + radius))

    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return []

    crop = frame[cy1:cy2, cx1:cx2]
    scale = max(1.0, 224 / min(crop.shape[:2]))

    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    result = helmet_model(crop, conf=conf, imgsz=imgsz, verbose=False)[0]

    if result.boxes is None:
        return []

    found = []

    for b in result.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().tolist()
        box = [x1 / scale + cx1, y1 / scale + cy1,
               x2 / scale + cx1, y2 / scale + cy1]

        # ignore a neighbour's helmet that happens to be inside the crop
        centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

        if math.hypot(centre[0] - hx, centre[1] - hy) <= radius * 0.6:
            found.append({"box": box, "conf": float(b.conf[0]), "cls": int(b.cls[0])})

    return found


def verify_decision(detections, with_class, without_class, without_min):
    """
    From zoomed head-crop detections, return a "without helmet" detection if
    it clearly beats any "with helmet" one, else ``None`` (keep the original).
    """
    withs = [d for d in detections if d["cls"] == with_class]
    withouts = [d for d in detections if d["cls"] == without_class]

    best_with = max((d["conf"] for d in withs), default=0.0)
    best_without = max(withouts, key=lambda d: d["conf"], default=None)

    if (best_without is not None
            and best_without["conf"] >= without_min
            and best_without["conf"] > best_with):
        return best_without

    return None


def verify_with_decision(detections, with_class, without_class, with_min):
    """
    The opposite check: from zoomed head-crop detections, return a "with
    helmet" detection if it clearly beats any "without helmet" one (a dark helmet
    first called "Without Helmet"), else ``None`` (keep the original).
    """
    withs = [d for d in detections if d["cls"] == with_class]
    withouts = [d for d in detections if d["cls"] == without_class]

    best_without = max((d["conf"] for d in withouts), default=0.0)
    best_with = max(withs, key=lambda d: d["conf"], default=None)

    if (best_with is not None
            and best_with["conf"] >= with_min
            and best_with["conf"] > best_without):
        return best_with

    return None