"""Smoke test verifying imports and basic package initialization."""

import cctv


def test_package_version():
    """Verify cctv package is importable and has a version string."""
    assert cctv.__version__ == "0.1.0"


def test_smoke_import_modules():
    """Verify pure logic modules can be imported."""
    from cctv import clips, db, detect_track, events, video_io, zones
    from cctv.query import builder

    assert hasattr(zones, "Zone")
    assert hasattr(events, "CCTVEvent")
    assert hasattr(db, "EventDatabase")
    assert hasattr(builder, "EventQueryEngine")
