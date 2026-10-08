"""Tests for query engine and verification of Rule 2 and Rule 3."""

import tempfile
from pathlib import Path

from cctv.db import NO_EVIDENCE_MESSAGE, EventDatabase
from cctv.events import CCTVEvent
from cctv.query.builder import EventQuery, EventQueryEngine


def test_no_evidence_message_exact():
    """Verify that when no evidence exists, the exact required string is returned."""
    # ignore_cleanup_errors=True avoids Windows WinError 32 when SQLite
    # WAL/SHM files are briefly held after the connection is released.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        db = EventDatabase(db_path=db_path)
        engine = EventQueryEngine(db)

        query = EventQuery(camera_id="cam_999")
        res = engine.execute(query)

        assert not res.found
        assert res.message == "I couldn't verify this from the available footage."
        assert res.message == NO_EVIDENCE_MESSAGE
        assert len(res.events) == 0


def test_evidence_presence_and_fields():
    """Verify that existing events are returned with required Rule 3 fields."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        db = EventDatabase(db_path=db_path)

        ev = CCTVEvent(
            camera_id="cam_entrance",
            event_type="zone_entry",
            track_id=1,
            class_name="person",
            zone_name="doorway",
            start_time=2.5,
            end_time=4.0,
            start_frame=50,
            end_frame=80,
            video_path="data/raw/entrance.mp4",
            det_conf_mean=0.88,
            det_conf_min=0.82,
            det_conf_max=0.94,
        )
        db.insert_event(ev)

        engine = EventQueryEngine(db)
        query = EventQuery(camera_id="cam_entrance")
        res = engine.execute(query)

        assert res.found
        assert len(res.events) == 1
        item = res.events[0]

        # Verify Rule 3 required fields
        assert item["camera_id"] == "cam_entrance"
        assert item["start_time"] == 2.5
        assert item["end_time"] == 4.0
        assert item["start_frame"] == 50
        assert item["end_frame"] == 80
        assert item["video_path"] == "data/raw/entrance.mp4"
        assert item["det_conf_mean"] == 0.88
