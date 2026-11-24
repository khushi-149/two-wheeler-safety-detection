import pytest

import pipeline


@pytest.mark.parametrize("source, expected", [
    (0, "camera 0"),
    ("rtsp://admin:s3cret@192.168.1.10:554/stream1", "rtsp://192.168.1.10:554"),
    ("http://user:p%40ss@10.0.0.5:8080/video", "http://10.0.0.5:8080"),
    ("http://10.0.0.5:8080/video", "http://10.0.0.5:8080"),
    ("data/videos/clip.mp4", "data/videos/clip.mp4"),
])
def test_redact_source_never_leaks_credentials(source, expected):
    text = pipeline.redact_source(source)
    assert text == expected
    assert "s3cret" not in text and "p%40ss" not in text


def test_helmet_classes_read_from_model_names():
    assert pipeline.resolve_helmet_classes({0: "With Helmet", 1: "Without Helmet"}) == (0, 1)
    # a retrained model with the opposite order must still work
    assert pipeline.resolve_helmet_classes({0: "Without Helmet", 1: "With Helmet"}) == (1, 0)
    assert pipeline.resolve_helmet_classes(["helmet", "no_helmet"]) == (0, 1)


def test_helmet_classes_fall_back_to_defaults():
    assert pipeline.resolve_helmet_classes(None) == (0, 1)
    assert pipeline.resolve_helmet_classes({0: "a", 1: "b"}) == (0, 1)
    assert pipeline.resolve_helmet_classes({0: "helmet"}) == (0, 1)