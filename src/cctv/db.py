"""SQLite event indexing and retrieval layer."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from cctv.events import CCTVEvent

logger = logging.getLogger(__name__)

NO_EVIDENCE_MESSAGE = "I couldn't verify this from the available footage."


class EventDatabase:
    """Manages SQLite storage and querying for CCTV events."""

    def __init__(self, db_path: Path | str = "data/db/cctv_events.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    zone_name TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    start_frame INTEGER NOT NULL,
                    end_frame INTEGER NOT NULL,
                    video_path TEXT NOT NULL,
                    det_conf_mean REAL NOT NULL,
                    det_conf_min REAL NOT NULL,
                    det_conf_max REAL NOT NULL,
                    extra_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_class ON events(class_name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(start_time, end_time);")

    def insert_event(self, event: CCTVEvent) -> int:
        """Insert a single CCTVEvent into the database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (
                    camera_id, event_type, track_id, class_name, zone_name,
                    start_time, end_time, start_frame, end_frame, video_path,
                    det_conf_mean, det_conf_min, det_conf_max, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.camera_id,
                    event.event_type,
                    event.track_id,
                    event.class_name,
                    event.zone_name,
                    event.start_time,
                    event.end_time,
                    event.start_frame,
                    event.end_frame,
                    event.video_path,
                    event.det_conf_mean,
                    event.det_conf_min,
                    event.det_conf_max,
                    json.dumps(event.extra_data),
                ),
            )
            return cursor.lastrowid or -1

    def insert_events(self, events: list[CCTVEvent]) -> int:
        """Batch insert multiple CCTVEvent instances."""
        if not events:
            return 0
        with self._get_connection() as conn:
            cursor = conn.executemany(
                """
                INSERT INTO events (
                    camera_id, event_type, track_id, class_name, zone_name,
                    start_time, end_time, start_frame, end_frame, video_path,
                    det_conf_mean, det_conf_min, det_conf_max, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.camera_id,
                        e.event_type,
                        e.track_id,
                        e.class_name,
                        e.zone_name,
                        e.start_time,
                        e.end_time,
                        e.start_frame,
                        e.end_frame,
                        e.video_path,
                        e.det_conf_mean,
                        e.det_conf_min,
                        e.det_conf_max,
                        json.dumps(e.extra_data),
                    )
                    for e in events
                ],
            )
            return cursor.rowcount

    def search_events(
        self,
        camera_id: str | None = None,
        event_type: str | None = None,
        class_name: str | None = None,
        zone_name: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        min_det_conf: float | None = None,
    ) -> list[dict[str, Any]]:
        """Query events matching filters. Returns exact rows found in DB (Rule 2)."""
        query = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []

        if camera_id is not None:
            query += " AND camera_id = ?"
            params.append(camera_id)
        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type)
        if class_name is not None:
            query += " AND class_name = ?"
            params.append(class_name)
        if zone_name is not None:
            query += " AND zone_name = ?"
            params.append(zone_name)
        if start_time is not None:
            query += " AND end_time >= ?"
            params.append(start_time)
        if end_time is not None:
            query += " AND start_time <= ?"
            params.append(end_time)
        if min_det_conf is not None:
            query += " AND det_conf_mean >= ?"
            params.append(min_det_conf)

        query += " ORDER BY start_time ASC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
