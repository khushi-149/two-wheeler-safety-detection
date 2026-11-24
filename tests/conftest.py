"""Shared test setup: import path, and a stand-in for ultralytics when it is not installed."""

import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:  # real package present (developer machine): tests still never load weights
    import ultralytics  # noqa: F401
except ImportError:  # CI / minimal environment
    stub = types.ModuleType("ultralytics")

    class YOLO:  # replaced per-test through monkeypatch
        def __init__(self, *args, **kwargs):
            raise RuntimeError("tests must patch pipeline.YOLO")

    stub.YOLO = YOLO
    sys.modules["ultralytics"] = stub