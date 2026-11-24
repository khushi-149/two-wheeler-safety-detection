"""Motorcycle tracking using Ultralytics ByteTrack."""

from __future__ import annotations

from settings import COCO_MOTORCYCLE_CLASS


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