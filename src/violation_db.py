"""SQLite violation log: one record per (run, motorcycle track, violation type)."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime
from typing import Iterator, Optional

from settings import DB_PATH


class ViolationDB:
    """Thin wrapper around a SQLite file. Safe to use from several threads."""

    def __init__(self, path: str | os.PathLike = DB_PATH):
        self.path = str(path)
        folder = os.path.dirname(self.path)

        if folder:
            os.makedirs(folder, exist_ok=True)

        self._init_schema()

    # ---------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """
        Open a connection, commit on success / roll back on error, and ALWAYS
        close it (``with sqlite3.connect()`` alone only commits, it does not
        close, which leaks file handles and locks the file on Windows).
        """
        with closing(sqlite3.connect(self.path, timeout=10)) as conn:
            conn.row_factory = sqlite3.Row

            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _init_schema(self) -> None:
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_violations_run "
                "ON violations (run_id)"
            )

    # ---------------------------------------------------------------

    def log_violation(
        self,
        run_id: str,
        video_name: str,
        track_id: int,
        violation_type: str,
        video_time: float,
        evidence_path: Optional[str],
        plate_number: Optional[str] = None,
        plate_conf: Optional[float] = None,
        plate_image_path: Optional[str] = None,
    ) -> bool:
        """Insert a record. Returns True if it is new, False if it was already logged."""
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
                    plate_number,
                    None if plate_conf is None else float(plate_conf),
                    plate_image_path,
                ),
            )
            return cursor.rowcount > 0

    def update_plate(
        self,
        run_id: str,
        track_id: int,
        plate_number: str,
        plate_conf: float,
        plate_image_path: Optional[str] = None,
    ) -> None:
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

    def fetch(
        self,
        run_id: Optional[str] = None,
        violation_type: Optional[str] = None,
    ) -> list[dict]:
        """All matching records, newest first."""
        query = "SELECT * FROM violations WHERE 1=1"
        params: list = []

        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)

        if violation_type:
            query += " AND violation_type = ?"
            params.append(violation_type)

        query += " ORDER BY id DESC"

        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query, params)]

    def summary(self, run_id: Optional[str] = None) -> dict[str, int]:
        """Counts per violation type, plus how many records have a plate."""
        where = " WHERE run_id = ?" if run_id else ""
        params = [run_id] if run_id else []

        with self._connect() as conn:
            counts = {
                row["violation_type"]: row["n"]
                for row in conn.execute(
                    "SELECT violation_type, COUNT(*) AS n FROM violations"
                    f"{where} GROUP BY violation_type",
                    params,
                )
            }

            plate_where = (
                " WHERE plate_number IS NOT NULL"
                + (" AND run_id = ?" if run_id else "")
            )
            plates = conn.execute(
                f"SELECT COUNT(*) AS n FROM violations{plate_where}", params
            ).fetchone()["n"]

        return {
            "no_helmet": counts.get("no_helmet", 0),
            "triple_riding": counts.get("triple_riding", 0),
            "with_plate": plates,
        }

    def run_ids(self) -> list[str]:
        """All run ids that have at least one record, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT run_id FROM violations ORDER BY run_id DESC"
            )
            return [row["run_id"] for row in rows]

    # ---------------------------------------------------------------

    def runs_overview(self) -> list[dict]:
        """One row per run (newest first): source, time and what was found."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id,
                       MAX(video_name) AS source,
                       MIN(detected_at) AS started,
                       SUM(violation_type = 'no_helmet') AS no_helmet,
                       SUM(violation_type = 'triple_riding') AS triple_riding,
                       COUNT(DISTINCT CASE WHEN plate_number IS NOT NULL
                                           THEN track_id END) AS plates,
                       COUNT(*) AS records
                FROM violations
                GROUP BY run_id
                ORDER BY run_id DESC
                """
            )
            return [dict(row) for row in rows]

    def delete_runs(self, run_ids, remove_files: bool = True) -> int:
        """
        Delete every record of the given runs, and (by default) the evidence and
        plate pictures they point to. Returns how many records were removed.
        """
        run_ids = list(run_ids)

        if not run_ids:
            return 0

        marks = ",".join("?" for _ in run_ids)

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT evidence_path, plate_image_path FROM violations "
                f"WHERE run_id IN ({marks})", run_ids,
            ).fetchall()
            removed = conn.execute(
                f"DELETE FROM violations WHERE run_id IN ({marks})", run_ids
            ).rowcount

        if remove_files:
            self._remove_files(
                path for row in rows
                for path in (row["evidence_path"], row["plate_image_path"])
            )

        return removed

    def delete_all(self, remove_files: bool = True) -> int:
        """Delete every record (and their pictures). Returns how many."""
        return self.delete_runs(self.run_ids(), remove_files=remove_files)

    @staticmethod
    def _remove_files(paths) -> None:
        for path in {p for p in paths if p}:
            try:
                os.remove(path)
            except OSError:
                pass        # already gone, or in use: not worth failing for