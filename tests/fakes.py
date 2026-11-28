"""Minimal stand-ins for the Ultralytics result objects the pipeline touches."""

from __future__ import annotations


class T:
    """Behaves like the tiny part of a torch tensor the code uses."""

    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def tolist(self):
        return self.data

    def int(self):
        return T([int(v) for v in self.data])

    def __getitem__(self, index):
        item = self.data[index]
        return T(item) if isinstance(item, (list, tuple)) else item

    def __len__(self):
        return len(self.data)


class Box:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = T([list(xyxy)])
        self.conf = T([float(conf)])
        self.cls = T([int(cls)])


class Boxes:
    """Iterable of Box that ALSO exposes batched tensors (as tracker results do)."""

    def __init__(self, boxes, ids=None):
        self._boxes = boxes
        self.xyxy = T([b.xyxy.data[0] for b in boxes])
        self.conf = T([b.conf.data[0] for b in boxes])
        self.cls = T([b.cls.data[0] for b in boxes])
        self.id = None if ids is None else T(list(ids))

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)


class Keypoints:
    def __init__(self, xy, conf):
        self.xy = T(xy)
        self.conf = T(conf)


class Result:
    def __init__(self, boxes=None, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


def person_result(people):
    """people: list of dicts {box, head, hip}; keypoints in the same coordinates."""
    boxes, xy, conf = [], [], []

    for p in people:
        boxes.append(Box(p["box"], 0.9, 0))
        points = [[0.0, 0.0]] * 17
        confs = [0.0] * 17

        if p.get("head"):
            points = list(points)
            points[0] = list(p["head"])
            confs[0] = 0.9

        if p.get("hip"):
            points = list(points)
            points[11] = list(p["hip"])
            points[12] = list(p["hip"])
            confs[11] = confs[12] = 0.9

        xy.append(points)
        conf.append(confs)

    return Result(Boxes(boxes), Keypoints(xy, conf))


def helmet_result(heads):
    """heads: list of (xyxy, cls)."""
    return Result(Boxes([Box(b, 0.8, c) for b, c in heads]))


class FakeTraffic:
    """Stands in for the YOLO traffic model: .track() returns scripted motorcycles."""

    def __init__(self, script):
        self.script = script  # callable(frame_number) -> list of (track_id, xyxy)
        self.calls = 0

    def track(self, **kwargs):
        self.calls += 1
        items = self.script(self.calls)
        boxes = [Box(b, 0.9, 3) for _, b in items]
        ids = [i for i, _ in items] if items else None
        return [Result(Boxes(boxes, ids if items else None))]

    def __call__(self, source=None, **kwargs):
        """Plain detection (used for photos): same script, remembers the kwargs."""
        self.calls += 1
        self.last_kwargs = kwargs
        items = self.script(self.calls)
        boxes = [Box(b, 0.9, 3) for _, b in items]
        return [Result(Boxes(boxes))]


class FakeHelmetModel:
    names = {0: "With Helmet", 1: "Without Helmet"}

    def __call__(self, source, **kwargs):
        crops = source if isinstance(source, list) else [source]
        return [Result(Boxes([])) for _ in crops]


class FakePoseModel:
    def __call__(self, source, **kwargs):
        crops = source if isinstance(source, list) else [source]
        return [Result(Boxes([]), Keypoints([], [])) for _ in crops]