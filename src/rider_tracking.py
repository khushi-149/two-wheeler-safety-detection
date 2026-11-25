"""Motorcycle tracking using Ultralytics ByteTrack."""

from __future__ import annotations

from settings import (
    COCO_MOTORCYCLE_CLASS,
    MOTORCYCLE_CONF,
    PHOTO_IMGSZ,
    PHOTO_MOTORCYCLE_RETRY_CONF,
)


class MotorcycleTracker:
    """Track motorcycles across consecutive video frames."""

    TRACKER_CONFIG = "bytetrack.yaml"

    def __init__(self, model, confidence: float = 0.25):
        self.model = model
        self.confidence = confidence

    def update(self, frame) -> list[dict]:
        """
        Process one frame and return the tracked motorcycles.

        Each item is ``{"box", "conf", "class_id", "track_id"}``. Detections
        the tracker has not assigned an id to yet are skipped.
        """
        result = self.model.track(
            source=frame,
            persist=True,
            classes=[COCO_MOTORCYCLE_CLASS],
            conf=self.confidence,
            tracker=self.TRACKER_CONFIG,
            verbose=False,
        )[0]

        boxes = result.boxes

        if boxes is None or boxes.id is None:
            return []

        return [
            {
                "box": box,
                "conf": float(conf),
                "class_id": int(class_id),
                "track_id": int(track_id),
            }
            for box, conf, class_id, track_id in zip(
                boxes.xyxy.cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.cls.cpu().tolist(),
                boxes.id.int().cpu().tolist(),
            )
        ]


class PhotoMotorcycleDetector:
    """
    Find motorcycles in a single photo (no tracking needed).

    The video tracker only sees detections above ``confidence``, so a bike the
    model is unsure about (close-ups, odd angles) is silently missed and the
    photo ends with no riders and no helmets. For a photo we can afford to try
    harder: first the normal confidence, then a lower one, then a lower one at
    a larger image size. The first attempt that finds a bike wins.

    Same ``update(frame)`` interface as :class:`MotorcycleTracker`; track ids
    are simply 1, 2, 3 ... in order of confidence.
    """

    def __init__(self, model, confidence: float = MOTORCYCLE_CONF,
                 retry_confidence: float = PHOTO_MOTORCYCLE_RETRY_CONF,
                 large_imgsz: int = PHOTO_IMGSZ):
        self.model = model
        self.attempts = [
            {"conf": confidence},
            {"conf": retry_confidence},
            {"conf": retry_confidence, "imgsz": large_imgsz},
        ]
        self.used_attempt = None  # index of the attempt that found bikes (for debugging)

    def update(self, frame) -> list[dict]:
        for number, extra in enumerate(self.attempts):
            result = self.model(
                frame,
                classes=[COCO_MOTORCYCLE_CLASS],
                verbose=False,
                **extra,
            )[0]

            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            found = sorted(
                zip(
                    boxes.xyxy.cpu().tolist(),
                    boxes.conf.cpu().tolist(),
                    boxes.cls.cpu().tolist(),
                ),
                key=lambda item: item[1],
                reverse=True,
            )

            self.used_attempt = number

            return [
                {
                    "box": box,
                    "conf": float(conf),
                    "class_id": int(class_id),
                    "track_id": index,
                }
                for index, (box, conf, class_id) in enumerate(found, start=1)
            ]

        return []