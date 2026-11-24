import settings


def test_paths_are_anchored_to_project_root():
    assert settings.PROJECT_ROOT.name != "src"
    assert (settings.PROJECT_ROOT / "src" / "settings.py").exists()
    assert settings.resolve_path("outputs/x.db") == str(settings.PROJECT_ROOT / "outputs" / "x.db")


def test_resolve_path_keeps_absolute(tmp_path):
    assert settings.resolve_path(tmp_path) == str(tmp_path)


def test_model_path_prefers_models_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)
    (tmp_path / "a.pt").write_bytes(b"x")

    assert settings.model_path("a.pt") == str(tmp_path / "a.pt")


def test_model_path_unknown_name_is_returned_for_auto_download(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)

    assert settings.model_path("yolo11n.pt") == "yolo11n.pt"


def test_extension_lists_are_lowercase_dotted():
    for ext in settings.IMAGE_EXTENSIONS + settings.VIDEO_EXTENSIONS:
        assert ext.startswith(".") and ext == ext.lower()