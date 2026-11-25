"""
End-to-end tests of ViolationPipeline.run() with scripted stand-in models and
real image / video files: tracking, rider counting, smoothing, drawing, evidence
files and the database are all exercised for real.
"""

import cv2
import numpy as np
import pytest

import pipeline as pl
from fakes import FakeHelmetModel, FakeTraffic, helmet_result, person_result

BIKE = (200, 250, 400, 420)


def make_riders(n):
    """n people sitting on BIKE (hips inside the bike box)."""
    out = []
    for i in range(n):
        cx = 230 + 70 * i
        out.append({"box": [cx - 35, 100, cx + 35, 400],
                    "head": (cx, 120), "hip": (cx, 360)})
    return out


def heads_for(riders, no_helmet_index=None):
    heads = []
    for i, r in enumerate(riders):
        cx, cy = r["head"]
        cls = 1 if i == no_helmet_index else 0
        heads.append(([cx - 20, cy - 25, cx + 20, cy + 15], cls))
    return heads


@pytest.fixture()
def make_pipeline(tmp_path, monkeypatch):
    weights = tmp_path / "helmet.pt"
    weights.write_bytes(b"fake")

    def build(riders, no_helmet_index=None, bikes=lambda n: [(1, BIKE)]):
        traffic = FakeTraffic(bikes)

        def fake_yolo(path):
            return traffic

        monkeypatch.setattr(pl, "YOLO", fake_yolo)
        p = pl.ViolationPipeline(
            pose_path="pose.pt", helmet_path=str(weights), traffic_path="t.pt",
            use_plates=False, db_path=tmp_path / "v.db",
            evidence_folder=tmp_path / "ev", video_folder=tmp_path / "vid",
            plate_folder=tmp_path / "pl", image_folder=tmp_path / "img",
        )
        # replace the model-heavy step with scripted detections (frame coordinates)
        monkeypatch.setattr(
            p, "_analyze_crops",
            lambda frame, bikes_: (riders, [
                {"box": b, "cls": c, "conf": 0.9} for b, c in heads_for(riders, no_helmet_index)
            ]),
        )
        return p

    return build


def write_photo(tmp_path, name="photo.jpg"):
    path = tmp_path / name
    cv2.imwrite(str(path), np.full((480, 640, 3), 90, np.uint8))
    return str(path)


def write_video(tmp_path, frames=12, name="clip.mp4"):
    path = tmp_path / name
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    for _ in range(frames):
        writer.write(np.full((480, 640, 3), 90, np.uint8))
    writer.release()
    return str(path)


# ------------------------------------------------------------------ photos

def test_photo_with_three_riders_is_flagged_triple(make_pipeline, tmp_path):
    """Regression: a single photo used to be capped at 2 riders and never flagged."""
    p = make_pipeline(make_riders(3))

    stats = p.run(write_photo(tmp_path))

    assert stats["motorcycles"] == 1
    assert stats["triple_riding"] == 1
    assert stats["helmet_violations"] == 0
    assert stats["output_image"] and cv2.imread(stats["output_image"]) is not None
    assert [r["violation_type"] for r in p.db.fetch()] == ["triple_riding"]


def test_photo_with_no_helmet_rider(make_pipeline, tmp_path):
    p = make_pipeline(make_riders(2), no_helmet_index=1)

    stats = p.run(write_photo(tmp_path))

    assert stats["helmet_violations"] == 1 and stats["triple_riding"] == 0
    assert stats["evidence_saved"] == 1
    assert cv2.imread(p.db.fetch()[0]["evidence_path"]) is not None


def test_photo_with_two_helmeted_riders_has_no_violation(make_pipeline, tmp_path):
    p = make_pipeline(make_riders(2))

    stats = p.run(write_photo(tmp_path))

    assert stats["helmet_violations"] == 0 and stats["triple_riding"] == 0
    assert stats["evidence_saved"] == 0 and p.db.fetch() == []


def test_photo_without_motorcycles(make_pipeline, tmp_path):
    p = make_pipeline(make_riders(3), bikes=lambda n: [])

    stats = p.run(write_photo(tmp_path))

    assert stats["motorcycles"] == 0 and stats["evidence_saved"] == 0


# ------------------------------------------------------------------ videos

def test_video_triple_riding_needs_agreement_then_is_logged_once(make_pipeline, tmp_path):
    p = make_pipeline(make_riders(3))

    stats = p.run(write_video(tmp_path, frames=12))

    assert stats["triple_riding"] == 1
    assert len(p.db.fetch()) == 1                         # one row per bike + type
    assert stats["output_video"]
    cap = cv2.VideoCapture(stats["output_video"])
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12   # every frame written
    cap.release()


def test_frame_stride_analyses_every_nth_frame_but_reports_source_position(make_pipeline, tmp_path):
    p = make_pipeline(make_riders(3))

    stats = p.run(write_video(tmp_path, frames=30), frame_stride=3)

    assert stats["frames_analysed"] == 10                 # 3x fewer frames analysed
    assert stats["frame"] == 30                           # progress is in source frames
    assert stats["triple_riding"] == 1                    # still finds the violation


def test_video_too_short_for_agreement_does_not_flag_triple(make_pipeline, tmp_path):
    p = make_pipeline(make_riders(3))

    stats = p.run(write_video(tmp_path, frames=2))

    assert stats["triple_riding"] == 0                    # smoothing: single glitch protection


def test_two_bikes_tracked_independently(make_pipeline, tmp_path):
    other = (420, 250, 600, 420)
    p = make_pipeline(make_riders(1), bikes=lambda n: [(1, BIKE), (2, other)])

    stats = p.run(write_video(tmp_path, frames=6))

    assert stats["motorcycles"] == 2


def test_missing_video_raises_clear_error(make_pipeline, tmp_path):
    p = make_pipeline(make_riders(1))

    with pytest.raises(RuntimeError, match="Could not open source"):
        p.run(str(tmp_path / "nope.mp4"))


def test_error_message_hides_stream_password(make_pipeline):
    p = make_pipeline(make_riders(1))

    with pytest.raises(RuntimeError) as err:
        p.run("rtsp://admin:topsecret@127.0.0.1:1/none", live=True)

    assert "topsecret" not in str(err.value)


def test_stop_callback_ends_run_early(make_pipeline, tmp_path):
    p = make_pipeline(make_riders(1))
    seen = {"n": 0}

    def should_stop():
        seen["n"] += 1
        return seen["n"] > 3

    stats = p.run(write_video(tmp_path, frames=30), should_stop=should_stop)

    assert stats["frame"] == 3


def test_frame_callback_receives_stats(make_pipeline, tmp_path):
    p = make_pipeline(make_riders(2))
    calls = []

    p.run(write_video(tmp_path, frames=6), frame_callback=lambda f, s: calls.append(s),
          callback_every=2)

    assert calls and {"motorcycles", "helmet_violations", "triple_riding"} <= set(calls[0])


def test_missing_helmet_weights_gives_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "YOLO", lambda path: FakeHelmetModel())

    with pytest.raises(FileNotFoundError, match="helmet_detector.pt"):
        pl.ViolationPipeline(helmet_path=str(tmp_path / "missing.pt"),
                             db_path=tmp_path / "v.db", use_plates=False,
                             evidence_folder=tmp_path, video_folder=tmp_path,
                             plate_folder=tmp_path, image_folder=tmp_path)


# ------------------------------------------------- real _analyze_crops path

def test_crop_coordinates_are_mapped_back_to_the_frame(make_pipeline, tmp_path, monkeypatch):
    """Models see a crop; detections must come back in FULL-FRAME coordinates."""
    p = make_pipeline([])
    monkeypatch.undo()                                   # use the real _analyze_crops
    monkeypatch.setattr(pl, "YOLO", lambda path: FakeTraffic(lambda n: []))

    frame = np.zeros((480, 640, 3), np.uint8)
    bike = {"box": list(BIKE), "track_id": 1, "conf": 0.9}
    rx1, ry1, _, _ = pl.bike_region(BIKE, 640, 480)

    class Pose:
        def __call__(self, crops, **kw):
            return [person_result([{"box": [10, 20, 60, 120], "head": (30, 30), "hip": (35, 100)}])]

    class Helmet:
        def __call__(self, crops, **kw):
            return [helmet_result([([20, 20, 40, 45], 1)])]

    p.pose_model, p.helmet_model = Pose(), Helmet()

    people, heads = p._analyze_crops(frame, [bike])

    assert people[0]["box"] == [10 + rx1, 20 + ry1, 60 + rx1, 120 + ry1]
    assert people[0]["hip"] == (35 + rx1, 100 + ry1)
    assert heads[0]["box"] == [20 + rx1, 20 + ry1, 40 + rx1, 45 + ry1]
    assert heads[0]["cls"] == 1


# ------------------------------------------- head re-check, mannequins, photo size

from fakes import Box, Boxes, Result      # noqa: E402


class ZoomHelmet:
    """Stands in for the helmet model on a head crop: one box in the middle of the crop."""

    def __init__(self, cls, conf):
        self.cls, self.conf, self.calls = cls, conf, 0

    def __call__(self, crop, **kwargs):
        self.calls += 1
        h, w = crop.shape[:2]
        return [Result(Boxes([Box([w * 0.3, h * 0.3, w * 0.7, h * 0.7], self.conf, self.cls)]))]


def script_heads(p, monkeypatch, riders, head_specs):
    """head_specs: list of (rider_index, cls, conf) -> only those riders get a head box."""
    heads = []
    for index, cls, conf in head_specs:
        cx, cy = riders[index]["head"]
        heads.append({"box": [cx - 20, cy - 25, cx + 20, cy + 15], "cls": cls, "conf": conf})
    monkeypatch.setattr(p, "_analyze_crops", lambda frame, bikes: (riders, heads))


def test_rider_without_head_box_is_recovered_from_a_zoomed_crop(make_pipeline, tmp_path, monkeypatch):
    riders = make_riders(2)
    p = make_pipeline(riders)
    script_heads(p, monkeypatch, riders, [(0, 0, 0.9)])            # rider 2 has no head box
    p.helmet_model = ZoomHelmet(cls=1, conf=0.6)                   # zoomed look: "without helmet"

    stats = p.run(write_photo(tmp_path))

    assert stats["helmet_violations"] == 1
    assert p.helmet_model.calls == 1                               # only the rider with no head


def test_unfound_head_is_not_a_violation_by_default(make_pipeline, tmp_path, monkeypatch):
    riders = make_riders(2)
    p = make_pipeline(riders)
    script_heads(p, monkeypatch, riders, [(0, 0, 0.9)])
    p.helmet_model = ZoomHelmet(cls=1, conf=0.01)                  # below RECOVERY_CONF -> nothing found
    monkeypatch.setattr(p.helmet_model, "__call__", lambda *a, **k: [Result(Boxes([]))], raising=False)

    class Nothing:
        def __call__(self, crop, **kwargs):
            return [Result(Boxes([]))]

    p.helmet_model = Nothing()

    assert p.run(write_photo(tmp_path))["helmet_violations"] == 0


def test_infer_no_helmet_setting_flags_unfound_heads(make_pipeline, tmp_path, monkeypatch):
    import settings
    monkeypatch.setattr(settings, "INFER_NO_HELMET", True)

    riders = make_riders(2)
    p = make_pipeline(riders)
    script_heads(p, monkeypatch, riders, [(0, 0, 0.9)])

    class Nothing:
        def __call__(self, crop, **kwargs):
            return [Result(Boxes([]))]

    p.helmet_model = Nothing()

    assert p.run(write_photo(tmp_path))["helmet_violations"] == 1


def test_doubtful_with_helmet_is_overridden_by_the_zoomed_look(make_pipeline, tmp_path, monkeypatch):
    riders = make_riders(1)
    p = make_pipeline(riders)
    script_heads(p, monkeypatch, riders, [(0, 0, 0.50)])           # "with helmet", only 0.50 sure
    p.helmet_model = ZoomHelmet(cls=1, conf=0.70)                  # zoomed look: clearly without

    assert p.run(write_photo(tmp_path))["helmet_violations"] == 1


def test_doubtful_with_helmet_is_kept_when_zoomed_look_agrees(make_pipeline, tmp_path, monkeypatch):
    riders = make_riders(1)
    p = make_pipeline(riders)
    script_heads(p, monkeypatch, riders, [(0, 0, 0.50)])
    p.helmet_model = ZoomHelmet(cls=0, conf=0.90)

    assert p.run(write_photo(tmp_path))["helmet_violations"] == 0


def test_confident_with_helmet_is_not_rechecked(make_pipeline, tmp_path, monkeypatch):
    riders = make_riders(1)
    p = make_pipeline(riders)
    script_heads(p, monkeypatch, riders, [(0, 0, 0.90)])
    p.helmet_model = ZoomHelmet(cls=1, conf=0.90)

    assert p.run(write_photo(tmp_path))["helmet_violations"] == 0
    assert p.helmet_model.calls == 0                                # no wasted inference


def test_mannequins_behind_a_parked_scooter_never_become_a_violation(make_pipeline, tmp_path, monkeypatch):
    scooter = (170, 400, 317, 510)
    mannequins = [{"box": [x - 40, 282, x + 40, 425], "head": (x, 300), "hip": (x, 395), "conf": 0.9}
                  for x in (180, 215, 250, 285)]
    p = make_pipeline(mannequins, bikes=lambda n: [(1, scooter)])
    script_heads(p, monkeypatch, mannequins, [(i, 0, 0.9) for i in range(4)])

    stats = p.run(write_photo(tmp_path))

    assert stats["triple_riding"] == 0 and stats["helmet_violations"] == 0
    assert p.db.fetch() == []


def test_photos_use_a_larger_analysis_size_but_videos_do_not(make_pipeline):
    import settings
    p = make_pipeline(make_riders(1))
    big = [np.zeros((2000, 1250, 3), np.uint8)]
    medium = [np.zeros((700, 500, 3), np.uint8)]
    small = [np.zeros((300, 200, 3), np.uint8)]

    p.photo_mode = False
    assert p._crop_imgsz(big) == settings.CROP_IMGSZ                # video: fixed, fast

    p.photo_mode = True
    assert p._crop_imgsz(big) == settings.CROP_IMGSZ_MAX            # capped
    assert p._crop_imgsz(medium) == 704                             # own size, multiple of 32
    assert p._crop_imgsz(small) == settings.CROP_IMGSZ              # never below the minimum


class FixedPlateReader:
    def __init__(self):
        self.calls = 0

    def read(self, frame, box):
        self.calls += 1
        return {"text": "HR26DK8337", "conf": 0.9, "valid": True,
                "box": (1, 1, 5, 5), "crop": np.zeros((10, 30, 3), np.uint8)}


def test_photo_violator_plate_is_read_and_logged(make_pipeline, tmp_path, monkeypatch):
    riders = make_riders(1)
    p = make_pipeline(riders)
    script_heads(p, monkeypatch, riders, [(0, 1, 0.9)])          # no helmet
    p.plate_reader = FixedPlateReader()

    stats = p.run(write_photo(tmp_path))

    assert stats["helmet_violations"] == 1
    assert list(stats["plates"].values()) == ["HR26DK8337"]
    assert [r["plate_number"] for r in p.db.fetch()] == ["HR26DK8337"]


def test_photo_without_violation_reads_plate_by_default(make_pipeline, tmp_path, monkeypatch):
    riders = make_riders(1)
    p = make_pipeline(riders)
    script_heads(p, monkeypatch, riders, [(0, 0, 0.9)])
    p.plate_reader = FixedPlateReader()

    p.run(write_photo(tmp_path))

    assert p.plate_reader.calls == 1


def test_photo_without_violation_skips_plate_when_switched_off(make_pipeline, tmp_path, monkeypatch):
    import settings

    monkeypatch.setattr(settings, "READ_PLATES_IN_PHOTOS", False)
    riders = make_riders(1)
    p = make_pipeline(riders)
    script_heads(p, monkeypatch, riders, [(0, 0, 0.9)])
    p.plate_reader = FixedPlateReader()

    p.run(write_photo(tmp_path))

    assert p.plate_reader.calls == 0


def test_rider_with_visible_face_and_no_head_found_is_flagged(make_pipeline, tmp_path, monkeypatch):
    riders = make_riders(2)
    riders[1]["face"] = 3                                          # nose + both eyes clearly seen
    p = make_pipeline(riders)
    script_heads(p, monkeypatch, riders, [(0, 0, 0.9)])            # rider 2: no head box found

    class Nothing:
        def __call__(self, crop, **kwargs):
            return [Result(Boxes([]))]

    p.helmet_model = Nothing()

    assert p.run(write_photo(tmp_path))["helmet_violations"] == 1


def test_rider_seen_from_behind_with_no_head_found_is_not_flagged(make_pipeline, tmp_path, monkeypatch):
    riders = make_riders(2)
    riders[1]["face"] = 0                                          # back of the head only
    p = make_pipeline(riders)
    script_heads(p, monkeypatch, riders, [(0, 0, 0.9)])

    class Nothing:
        def __call__(self, crop, **kwargs):
            return [Result(Boxes([]))]

    p.helmet_model = Nothing()

    assert p.run(write_photo(tmp_path))["helmet_violations"] == 0


def test_hidden_pillion_head_gets_the_zoomed_second_look(make_pipeline, tmp_path, monkeypatch):
    riders = make_riders(1)
    p = make_pipeline(riders)
    own = {"box": [210, 95, 250, 135], "cls": 0, "conf": 0.9}
    pillion = {"box": [260, 110, 300, 150], "cls": 0, "conf": 0.50}   # "with helmet", only 0.50 sure
    monkeypatch.setattr(p, "_analyze_crops", lambda frame, bikes: (riders, [own, pillion]))
    p.helmet_model = ZoomHelmet(cls=1, conf=0.70)                  # zoomed look: clearly without

    assert p.run(write_photo(tmp_path))["helmet_violations"] == 1



def test_weak_with_helmet_that_nothing_confirms_counts_as_no_helmet(make_pipeline, tmp_path, monkeypatch):
    # a yellow cap / a baby: the helmet model says "With Helmet" at only 0.30
    riders = make_riders(1)
    p = make_pipeline(riders)
    script_heads(p, monkeypatch, riders, [(0, 0, 0.30)])

    class Nothing:
        def __call__(self, crop, **kwargs):
            return [Result(Boxes([]))]

    p.helmet_model = Nothing()                                     # zoomed look: nothing at all

    assert p.run(write_photo(tmp_path))["helmet_violations"] == 1


def test_fairly_sure_with_helmet_is_kept(make_pipeline, tmp_path, monkeypatch):
    riders = make_riders(1)
    p = make_pipeline(riders)
    script_heads(p, monkeypatch, riders, [(0, 0, 0.60)])           # a real helmet scored 0.58-0.85
    p.helmet_model = ZoomHelmet(cls=0, conf=0.39)

    assert p.run(write_photo(tmp_path))["helmet_violations"] == 0


def test_weak_with_helmet_is_kept_when_the_zoomed_look_confirms_it(make_pipeline, tmp_path, monkeypatch):
    riders = make_riders(1)
    p = make_pipeline(riders)
    script_heads(p, monkeypatch, riders, [(0, 0, 0.47)])
    p.helmet_model = ZoomHelmet(cls=0, conf=0.70)                  # zoomed look: clearly a helmet

    assert p.run(write_photo(tmp_path))["helmet_violations"] == 0