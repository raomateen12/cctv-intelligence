"""Unit tests for spatial zones pure logic, point-in-polygon containment, and scaling."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cctv.zones import (
    Zone,
    draw_zones_on_frame,
    footpoint_of_box,
    load_zone_config,
    load_zones,
    parse_zones_from_config,
    save_zones,
    scale_zones,
    zone_of_bbox,
    zone_of_point,
)


# ── Preserved existing tests ──────────────────────────────────────────────────


def test_zone_contains_point():
    polygon = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    zone = Zone(name="box", polygon=polygon)

    assert zone.contains_point((50.0, 50.0))
    assert zone.contains_point((0.0, 0.0))  # vertex
    assert not zone.contains_point((150.0, 50.0))  # outside


def test_zone_contains_bbox_bottom_center():
    polygon = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    zone = Zone(name="box", polygon=polygon)

    # Box bottom-center is at (50, 90) -> inside
    bbox_inside = (40.0, 20.0, 60.0, 90.0)
    assert zone.contains_bbox(bbox_inside, anchor="bottom_center")

    # Box bottom-center is at (50, 120) -> outside
    bbox_outside = (40.0, 20.0, 60.0, 120.0)
    assert not zone.contains_bbox(bbox_outside, anchor="bottom_center")


def test_parse_zones_from_config():
    config_data = [
        {
            "name": "doorway",
            "polygon": [[10, 10], [50, 10], [50, 50], [10, 50]],
            "type": "entry_exit",
        }
    ]
    zones = parse_zones_from_config(config_data)
    assert len(zones) == 1
    assert zones[0].name == "doorway"
    assert zones[0].zone_type == "entry_exit"


# ── Extended tests: points inside, outside, edge, overlapping, scaling ───────


def test_zone_of_point_inside_and_outside():
    """Points strictly inside return the zone name; points outside return empty."""
    z = Zone(name="walkway", polygon=((10.0, 10.0), (100.0, 10.0), (100.0, 80.0), (10.0, 80.0)))
    zones = [z]

    # Inside
    assert zone_of_point(zones, 50.0, 50.0) == ["walkway"]
    assert zone_of_point(zones, 20.0, 70.0) == ["walkway"]

    # Outside
    assert zone_of_point(zones, 5.0, 50.0) == []
    assert zone_of_point(zones, 105.0, 50.0) == []
    assert zone_of_point(zones, 50.0, 5.0) == []
    assert zone_of_point(zones, 50.0, 90.0) == []


def test_zone_of_point_on_edge_and_vertex():
    """Points directly on the edge or vertex are inside (cv2.pointPolygonTest >= 0)."""
    polygon = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    z = Zone(name="loading_bay", polygon=polygon)
    zones = [z]

    # Edges
    assert zone_of_point(zones, 50.0, 0.0) == ["loading_bay"]    # Top edge
    assert zone_of_point(zones, 100.0, 50.0) == ["loading_bay"]  # Right edge
    assert zone_of_point(zones, 50.0, 100.0) == ["loading_bay"]  # Bottom edge
    assert zone_of_point(zones, 0.0, 50.0) == ["loading_bay"]    # Left edge

    # Vertices
    assert zone_of_point(zones, 0.0, 0.0) == ["loading_bay"]
    assert zone_of_point(zones, 100.0, 0.0) == ["loading_bay"]
    assert zone_of_point(zones, 100.0, 100.0) == ["loading_bay"]
    assert zone_of_point(zones, 0.0, 100.0) == ["loading_bay"]


def test_zone_of_point_overlapping_zones():
    """A point can be inside multiple overlapping zones simultaneously."""
    zone_a = Zone(name="Zone A", polygon=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)))
    zone_b = Zone(name="Zone B", polygon=((50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0)))
    zones = [zone_a, zone_b]

    # Overlap region: [50, 100] in x and y
    overlap_matches = zone_of_point(zones, 75.0, 75.0)
    assert overlap_matches == ["Zone A", "Zone B"]

    # Only in Zone A
    assert zone_of_point(zones, 25.0, 25.0) == ["Zone A"]

    # Only in Zone B
    assert zone_of_point(zones, 125.0, 125.0) == ["Zone B"]

    # Outside both
    assert zone_of_point(zones, 200.0, 200.0) == []


def test_footpoint_of_box_and_zone_of_bbox():
    """Footpoint is defined as bottom-center: ((x1 + x2) / 2, y2)."""
    # Box: x1=100, y1=200, x2=300, y2=500
    fx, fy = footpoint_of_box((100.0, 200.0, 300.0, 500.0))
    assert math.isclose(fx, 200.0)
    assert math.isclose(fy, 500.0)

    # Invalid box length raises ValueError
    with pytest.raises(ValueError, match="at least 4 coordinates"):
        footpoint_of_box((10.0, 20.0))

    # Test zone_of_bbox using bottom-center
    zone = Zone(name="danger_zone", polygon=((150.0, 450.0), (250.0, 450.0), (250.0, 550.0), (150.0, 550.0)))
    # Bottom center is (200, 500) -> inside danger_zone
    assert zone_of_bbox([zone], (100.0, 200.0, 300.0, 500.0)) == ["danger_zone"]

    # Bottom center is (200, 400) -> outside danger_zone
    assert zone_of_bbox([zone], (100.0, 200.0, 300.0, 400.0)) == []


def test_resolution_scaling_math():
    """Test scaling polygon coordinates from original resolution to target resolution."""
    orig_w, orig_h = 3840, 2160
    target_w, target_h = 1920, 1080  # 0.5x in both dimensions

    zone = Zone(
        name="storage",
        polygon=((100.0, 200.0), (500.0, 200.0), (500.0, 600.0), (100.0, 600.0)),
        zone_type="restricted",
    )

    scaled_zones = scale_zones([zone], orig_w, orig_h, target_w, target_h)
    assert len(scaled_zones) == 1
    sz = scaled_zones[0]

    assert sz.name == "storage"
    assert sz.zone_type == "restricted"
    expected_poly = ((50.0, 100.0), (250.0, 100.0), (250.0, 300.0), (50.0, 300.0))
    assert sz.polygon == expected_poly

    # Point at (150, 200) in target resolution should be inside
    assert sz.contains_point((150.0, 200.0))
    # Point at (300, 200) should be outside
    assert not sz.contains_point((300.0, 200.0))

    # Anisotropic scaling (different X and Y scales)
    scaled_aniso = scale_zones([zone], 1000, 500, 2000, 1500)  # sx=2.0, sy=3.0
    p0 = scaled_aniso[0].polygon[0]
    assert math.isclose(p0[0], 100.0 * 2.0)
    assert math.isclose(p0[1], 200.0 * 3.0)


def test_resolution_scaling_validation():
    """Non-positive dimensions or scale factors raise ValueError."""
    zone = Zone(name="test", polygon=((0, 0), (10, 0), (10, 10)))
    with pytest.raises(ValueError, match="Resolutions must be positive"):
        scale_zones([zone], orig_width=0, orig_height=100, target_width=100, target_height=100)

    with pytest.raises(ValueError, match="Scale factors must be positive"):
        zone.scale(-1.0, 1.0)


def test_save_and_load_zones_roundtrip(tmp_path: Path):
    """Test saving to JSON and loading with/without scaling."""
    out_file = tmp_path / "zones_testcam.json"
    zones = [
        Zone(name="Zone A", polygon=((100.0, 100.0), (400.0, 100.0), (400.0, 400.0), (100.0, 400.0)), zone_type="normal"),
        Zone(name="Zone B", polygon=((300.0, 300.0), (600.0, 300.0), (600.0, 600.0), (300.0, 600.0)), zone_type="restricted"),
    ]

    saved_path = save_zones("testcam", frame_width=1920, frame_height=1080, zones=zones, out_path=out_file)
    assert saved_path.exists()

    # Load without scaling (native resolution)
    loaded_cfg = load_zone_config(out_file)
    assert loaded_cfg.camera_id == "testcam"
    assert loaded_cfg.frame_width == 1920
    assert loaded_cfg.frame_height == 1080
    assert len(loaded_cfg.zones) == 2
    assert loaded_cfg.zones[0].name == "Zone A"
    assert loaded_cfg.zones[0].zone_type == "normal"
    assert loaded_cfg.zones[1].name == "Zone B"
    assert loaded_cfg.zones[1].zone_type == "restricted"

    # Load with scaling to target resolution 960x540 (0.5x)
    scaled_zones = load_zones(out_file, target_size=(960, 540))
    assert len(scaled_zones) == 2
    assert scaled_zones[0].polygon[0] == (50.0, 50.0)
    assert scaled_zones[1].polygon[1] == (300.0, 150.0)


def test_draw_zones_on_frame():
    """Verify draw_zones_on_frame renders visual markings on a synthetic frame."""
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    zones = [
        Zone(name="Norm", polygon=((50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0)), zone_type="normal"),
        Zone(name="Restr", polygon=((200.0, 200.0), (300.0, 200.0), (300.0, 300.0), (200.0, 300.0)), zone_type="restricted"),
    ]

    annotated = draw_zones_on_frame(frame, zones, alpha=0.3)
    assert annotated.shape == frame.shape
    # Ensure drawing modified the image (non-zero pixels present)
    assert np.count_nonzero(annotated) > 0
