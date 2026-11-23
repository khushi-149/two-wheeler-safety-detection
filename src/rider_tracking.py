"""Motorcycle tracking using Ultralytics ByteTrack."""

class MotorcycleTracker:
    """Track motorcycles across consecutive video frames."""

    def __init__(self, model, confidence=0.25):
        self.model = model
        self.confidence = confidence
        self.tracker_config = "bytetrack.yaml"

    def update(self, frame):
        """
        Process one frame and return tracked motorcycles.

        Each result contains:
        box, conf, class_id and track_id.
        """

        result = self.model.track(
            source=frame,
            persist=True,
            classes=[3],
            conf=self.confidence,
            tracker=self.tracker_config,
            verbose=False,
        )[0]

        motorcycles = []

        if result.boxes is None or result.boxes.id is None:
            return motorcycles

        boxes = result.boxes

        for box, confidence, class_id, track_id in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
            boxes.id.int().cpu().tolist(),
        ):
            motorcycles.append({
                "box": box,
                "conf": float(confidence),
                "class_id": int(class_id),
                "track_id": int(track_id),
            })

        return motorcycles
