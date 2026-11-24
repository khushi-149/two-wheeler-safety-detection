import pytest

from violation_db import ViolationDB


@pytest.fixture()
def db(tmp_path):
    return ViolationDB(tmp_path / "sub" / "v.db")     # creates the folder too


def log(db, run="r1", track=1, vtype="no_helmet", **kw):
    return db.log_violation(run, "clip.mp4", track, vtype, 1.5, "e.jpg", **kw)


def test_insert_and_duplicate_ignored(db):
    assert log(db) is True
    assert log(db) is False                      # same run/track/type
    assert log(db, vtype="triple_riding") is True
    assert len(db.fetch()) == 2


def test_fetch_filters_and_order(db):
    log(db, run="a", track=1)
    log(db, run="b", track=2, vtype="triple_riding")

    assert [r["run_id"] for r in db.fetch()] == ["b", "a"]      # newest first
    assert len(db.fetch(run_id="a")) == 1
    assert len(db.fetch(violation_type="triple_riding")) == 1
    assert db.run_ids() == ["b", "a"]


def test_update_plate_applies_to_whole_track(db):
    log(db, track=5)
    log(db, track=5, vtype="triple_riding")
    log(db, track=6)

    db.update_plate("r1", 5, "HR26DK8337", 0.91, "p.jpg")

    rows = {(r["track_id"], r["violation_type"]): r for r in db.fetch()}
    assert rows[(5, "no_helmet")]["plate_number"] == "HR26DK8337"
    assert rows[(5, "triple_riding")]["plate_conf"] == pytest.approx(0.91)
    assert rows[(6, "no_helmet")]["plate_number"] is None


def test_summary(db):
    log(db, track=1)
    log(db, track=2, vtype="triple_riding")
    db.update_plate("r1", 2, "MH12AB1234", 0.8)

    assert db.summary() == {"no_helmet": 1, "triple_riding": 1, "with_plate": 1}
    assert db.summary(run_id="other") == {"no_helmet": 0, "triple_riding": 0, "with_plate": 0}


def test_every_connection_is_closed(db, monkeypatch):
    """Regression: 'with sqlite3.connect()' commits but never closes the connection."""
    import sqlite3

    import violation_db

    opened, closed = [], []
    real_connect = sqlite3.connect

    class Tracked(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

        def close(self):
            closed.append(self)
            super().close()

    monkeypatch.setattr(
        violation_db.sqlite3, "connect",
        lambda *a, **k: real_connect(*a, factory=Tracked, **k),
    )

    for i in range(5):
        log(db, track=i)
        db.fetch()
        db.summary()
        db.run_ids()
        db.update_plate("r1", i, "HR26DK8337", 0.9)

    assert len(opened) > 0
    assert len(opened) == len(closed)


def test_failed_write_rolls_back_and_still_closes(db):
    import sqlite3

    with pytest.raises(sqlite3.Error):
        with db._connect() as conn:
            conn.execute("INSERT INTO violations (run_id) VALUES ('x')")   # NOT NULL fails

    assert db.fetch() == []


def test_survives_special_characters(db):
    assert log(db, run="r'; DROP TABLE violations;--") is True
    assert len(db.fetch()) == 1