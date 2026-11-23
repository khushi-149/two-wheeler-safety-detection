"""SQLite violation log: one record per (run, motorcycle track, violation type)."""

import os
import sqlite3
from datetime import datetime


class ViolationDB:

    def __init__(self, path="outputs/violations.db"):
        self.path = path
        folder = os.path.dirname(path)

        if folder:
            os.makedirs(folder, exist_ok=True)

        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    video_name TEXT,
                    track_id INTEGER NOT NULL,
                    violation_type TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    video_time REAL,
                    evidence_path TEXT,
                    plate_number TEXT,
                    plate_conf REAL,
                    plate_image_path TEXT,
                    UNIQUE (run_id, track_id, violation_type)
                )
                """
            )

    # ---------------------------------------------------------------

    def log_violation(self, run_id, video_name, track_id, violation_type,
                      video_time, evidence_path,
                      plate_number=None, plate_conf=None,
                      plate_image_path=None):
        """Insert a record. Returns True if new, False if already logged."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO violations
                (run_id, video_name, track_id, violation_type, detected_at,
                 video_time, evidence_path, plate_number, plate_conf,
                 plate_image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, video_name, int(track_id), violation_type,
                    datetime.now().isoformat(timespec="seconds"),
                    float(video_time), evidence_path,
                    plate_number, plate_conf, plate_image_path,
                ),
            )
            return cursor.rowcount > 0

    def update_plate(self, run_id, track_id, plate_number, plate_conf,
                     plate_image_path=None):
        """Attach (or improve) the plate on all records of one track in a run."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE violations
                SET plate_number = ?, plate_conf = ?, plate_image_path = ?
                WHERE run_id = ? AND track_id = ?
                """,
                (plate_number, float(plate_conf), plate_image_path,
                 run_id, int(track_id)),
            )

    # ---------------------------------------------------------------

    def fetch(self, run_id=None, violation_type=None):
        query = "SELECT * FROM violations WHERE 1=1"
        params = []

        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)

        if violation_type:
            query += " AND violation_type = ?"
            params.append(violation_type)

        query += " ORDER BY id DESC"

        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query, params)]

    def summary(self, run_id=None):
        query = "SELECT violation_type, COUNT(*) AS n FROM violations"
        params = []

        if run_id:
            query += " WHERE run_id = ?"
            params.append(run_id)

        query += " GROUP BY violation_type"

        with self._connect() as conn:
            counts = {row["violation_type"]: row["n"]
                      for row in conn.execute(query, params)}

            plate_query = (
                "SELECT COUNT(*) AS n FROM violations "
                "WHERE plate_number IS NOT NULL"
            )
            plate_params = []

            if run_id:
                plate_query += " AND run_id = ?"
                plate_params.append(run_id)

            plates = conn.execute(plate_query, plate_params).fetchone()["n"]

        return {
            "no_helmet": counts.get("no_helmet", 0),
            "triple_riding": counts.get("triple_riding", 0),
            "with_plate": plates,
        }

    def run_ids(self):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT run_id FROM violations ORDER BY run_id DESC"
            )
            return [row["run_id"] for row in rows]