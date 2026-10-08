"""Unit tests for event detection logic using synthetic track data.

All tests use purely in-memory data (no files, no video, no model weights).
Covers:
  - Normal observed enter/exit with ZONE_VISIT
  - Flicker at polygon edge (must produce exactly ONE enter/exit pair)
  - Track ending inside zone (open ZONE_VISIT and NO ZONE_EXIT)
  - Track starting inside zone (ZONE_FIRST_SEEN_INSIDE and NO ZONE_ENTER)
  - Succession hand-off pattern (#3 -> #9 style, IoU 0.49 in one frame flags both)
  - Long-coexisting tracks with peak IoU 0.21 not flagged as handoff
  - False split pattern (track < 2.0 s overlapping longer track flags short track)
  - Restricted zone observed entry (RESTRICTED_ENTRY)
  - Restricted zone starting inside (RESTRICTED_FIRST_SEEN_INSIDE)
  - first_seen_at_video_start boolean semantics (True at start, False mid-video)
  - Overlap flagging for concurrent tracks in zone
  - Gap flagging for temporal track dropouts
  - Noise track suppressed (shorter than min_track_len_s)
  - ZONE_OCCUPIED merging of adjacent intervals
"""

from __future__ import annotations

import math

import pytest

from cctv.events import (
    PERSON_APPEARED,
    PERSON_LEFT_VIEW,
    RESTRICTED_ENTRY,
    RESTRICTED_FIRST_SEEN_INSIDE,
    ZONE_ENTER,
    ZONE_EXIT,
    ZONE_FIRST_SEEN_INSIDE,
    ZONE_OCCUPIED,
    ZONE_VISIT,
    CCTVEvent,
    EventConfig,
    TrackRow,
    _iou,
    detect_events,
)
from cctv.zones import Zone

# ── Helpers ───────────────────────────────────────────────────────────────────

FPS = 10.0  # synthetic video at 10 fps → 1 frame = 0.1 s

# A simple 100×100 box zone in the centre of a 300×300 frame
ZONE = Zone(
    name="TestZone",
    polygon=((100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)),
    zone_type="normal",
)
RESTRICTED_ZONE = Zone(
    name="DangerZone",
    polygon=((100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)),
    zone_type="restricted",
)

DEFAULT_CFG = EventConfig(
    min_enter_s=1.0,
    exit_grace_s=2.0,
    min_track_len_s=1.0,
    occupied_merge_gap_s=2.0,
    overlap_iou_thresh=0.3,
    overlap_fraction_thresh=0.2,
    gap_s_thresh=1.0,
    handoff_iou_thresh=0.2,
)


def _row(frame: int, track_id: int, x1: float, y1: float, x2: float, y2: float,
         conf: float = 0.8) -> TrackRow:
    """Make a TrackRow at 10 fps."""
    return TrackRow(
        frame_idx=frame,
        time_s=round(frame / FPS, 4),
        track_id=track_id,
        cls=0,
        conf=conf,
        x1=x1, y1=y1, x2=x2, y2=y2,
    )


def _run(rows: list[TrackRow], zones: list[Zone], cfg: EventConfig = DEFAULT_CFG) -> list[CCTVEvent]:
    return detect_events(
        rows=rows,
        zones=zones,
        camera_id="test_cam",
        video_path="data/raw/test.mp4",
        video_fps=FPS,
        cfg=cfg,
    )


def _filter(events: list[CCTVEvent], etype: str) -> list[CCTVEvent]:
    return [e for e in events if e.event_type == etype]


# ── IoU helper unit test ──────────────────────────────────────────────────────


def test_iou_basics():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)
    # 50% overlap
    iou = _iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert 0.3 < iou < 0.4


# ── Noise track is suppressed ─────────────────────────────────────────────────


def test_noise_track_suppressed():
    """A track with span < min_track_len_s (1.0 s) is dropped completely."""
    # 5 frames at 10 fps = 0.5 s span → noise
    rows = [_row(f, 1, 50.0, 50.0, 80.0, 80.0) for f in range(0, 5)]
    events = _run(rows, [ZONE])
    # No PERSON_APPEARED/LEFT because track is noise
    assert not _filter(events, PERSON_APPEARED)
    assert not _filter(events, PERSON_LEFT_VIEW)


# ── Observed enter + exit ─────────────────────────────────────────────────────


def test_normal_enter_exit():
    """Track enters zone for > min_enter_s, then leaves for > exit_grace_s.
    Observed outside->inside yields ZONE_ENTER.
    Observed inside->outside yields ZONE_EXIT and closed ZONE_VISIT.
    """
    # Frames 0-19: outside zone (footpoint at (65, 170) → outside 100–200 range)
    # Frames 20-39: inside zone (footpoint at (140, 170))
    # Frames 40-79: outside zone (far away)
    rows = (
        [_row(f, 1, 30.0, 130.0, 100.0, 170.0) for f in range(0, 20)]   # outside
        + [_row(f, 1, 110.0, 120.0, 170.0, 170.0) for f in range(20, 40)]  # inside; fp=(140,170)
        + [_row(f, 1, 30.0, 130.0, 100.0, 170.0) for f in range(40, 80)]   # outside
    )
    events = _run(rows, [ZONE])

    enters = _filter(events, ZONE_ENTER)
    exits = _filter(events, ZONE_EXIT)
    visits = _filter(events, ZONE_VISIT)

    assert len(enters) == 1, f"Expected 1 ZONE_ENTER, got {len(enters)}"
    assert len(exits) == 1, f"Expected 1 ZONE_EXIT, got {len(exits)}"
    assert len(visits) == 1, f"Expected 1 ZONE_VISIT, got {len(visits)}"

    # ZONE_ENTER: point event at entry time
    assert math.isclose(enters[0].start_time_s, 2.0, abs_tol=1e-3)
    assert enters[0].starts_inside is False
    assert enters[0].open_ended is False

    # ZONE_EXIT: point event at last in-zone frame, carries duration and exit_confirmed_time_s
    assert math.isclose(exits[0].start_time_s, 3.9, abs_tol=1e-3)
    assert exits[0].duration_s > 0
    assert exits[0].exit_confirmed_time_s is not None
    assert exits[0].exit_confirmed_time_s >= 5.9  # 3.9s + 2.0s grace
    assert exits[0].duration_is_lower_bound is False

    # ZONE_VISIT: continuous stay from entry to exit
    assert math.isclose(visits[0].start_time_s, 2.0, abs_tol=1e-3)
    assert math.isclose(visits[0].end_time_s, 3.9, abs_tol=1e-3)
    assert visits[0].starts_inside is False
    assert visits[0].open_ended is False
    assert visits[0].duration_is_lower_bound is False

    # PERSON events exist
    assert len(_filter(events, PERSON_APPEARED)) == 1
    assert len(_filter(events, PERSON_LEFT_VIEW)) == 1


# ── Flicker at polygon edge ───────────────────────────────────────────────────


def test_flicker_produces_single_enter_exit():
    """A track that wavers in/out very briefly at the zone edge should produce
    exactly ONE enter and ONE exit (debounce prevents spurious events).
    """
    in_box = (110.0, 120.0, 170.0, 170.0)   # fp = (140, 170) → inside zone
    out_box = (30.0, 130.0, 100.0, 170.0)   # fp = (65, 170) → outside zone

    rows: list[TrackRow] = []
    # Observed outside start
    for f in range(0, 10):
        rows.append(_row(f, 1, *out_box))
    # Enter zone
    for f in range(10, 30):
        rows.append(_row(f, 1, *in_box))
    # Flicker out briefly (1-2 frames < exit_grace_s)
    rows.append(_row(30, 1, *out_box))
    rows.append(_row(31, 1, *in_box))
    rows.append(_row(32, 1, *out_box))
    rows.append(_row(33, 1, *in_box))
    # More inside
    for f in range(34, 45):
        rows.append(_row(f, 1, *in_box))
    # Properly outside > exit_grace_s
    for f in range(45, 90):
        rows.append(_row(f, 1, *out_box))

    events = _run(rows, [ZONE])
    enters = _filter(events, ZONE_ENTER)
    exits = _filter(events, ZONE_EXIT)
    visits = _filter(events, ZONE_VISIT)

    assert len(enters) == 1, f"Flicker should produce 1 ZONE_ENTER, got {len(enters)}"
    assert len(exits) == 1, f"Flicker should produce 1 ZONE_EXIT, got {len(exits)}"
    assert len(visits) == 1, f"Flicker should produce 1 ZONE_VISIT, got {len(visits)}"


# ── Track ending inside zone (open ZONE_VISIT and NO ZONE_EXIT) ───────────────


def test_track_ending_inside_zone_is_open_ended():
    """A track whose last frame is inside the zone must produce open_ended=True ZONE_VISIT
    and NO ZONE_EXIT (person never left the zone).
    """
    # Track enters zone at frame 10 (t=1.0s), stays until end of track (frame 30, t=3.0s)
    rows = (
        [_row(f, 1, 30.0, 130.0, 100.0, 170.0) for f in range(0, 10)]      # outside
        + [_row(f, 1, 110.0, 120.0, 170.0, 170.0) for f in range(10, 31)]  # inside
    )
    events = _run(rows, [ZONE])

    zone_events = [e for e in events if e.zone == "TestZone"]
    enters = _filter(zone_events, ZONE_ENTER)
    exits = _filter(zone_events, ZONE_EXIT)
    visits = _filter(zone_events, ZONE_VISIT)

    assert len(enters) == 1, f"Expected 1 ZONE_ENTER, got {len(enters)}"
    assert enters[0].open_ended is True
    assert len(exits) == 0, f"Expected NO ZONE_EXIT when track ends inside, got {len(exits)}"

    assert len(visits) == 1, f"Expected 1 ZONE_VISIT, got {len(visits)}"
    assert visits[0].open_ended is True
    assert visits[0].duration_is_lower_bound is True


# ── Track starting inside zone (ZONE_FIRST_SEEN_INSIDE and NO ZONE_ENTER) ──────


def test_track_starting_inside_zone():
    """First appearance is already inside the zone:
    Emits ZONE_FIRST_SEEN_INSIDE and NO ZONE_ENTER (true entry time unknown).
    When the track later leaves, ZONE_EXIT is emitted with duration_is_lower_bound=True.
    first_seen_at_video_start is True because track starts at frame 0.
    """
    rows = (
        [_row(f, 1, 110.0, 120.0, 170.0, 170.0) for f in range(0, 20)]  # inside from start
        + [_row(f, 1, 30.0, 130.0, 100.0, 170.0) for f in range(20, 60)]  # outside
    )
    events = _run(rows, [ZONE])

    presents = _filter(events, ZONE_FIRST_SEEN_INSIDE)
    enters = _filter(events, ZONE_ENTER)
    exits = _filter(events, ZONE_EXIT)
    visits = _filter(events, ZONE_VISIT)

    assert len(presents) == 1, f"Expected 1 ZONE_FIRST_SEEN_INSIDE, got {len(presents)}"
    assert presents[0].starts_inside is True
    assert presents[0].first_seen_at_video_start is True
    assert math.isclose(presents[0].start_time_s, 0.0, abs_tol=1e-3)

    assert len(enters) == 0, f"Track starting inside must NOT emit ZONE_ENTER, got {len(enters)}"

    assert len(exits) == 1, f"Expected 1 ZONE_EXIT, got {len(exits)}"
    assert exits[0].starts_inside is True
    assert exits[0].duration_is_lower_bound is True

    assert len(visits) == 1
    assert visits[0].starts_inside is True
    assert visits[0].duration_is_lower_bound is True


def test_first_seen_inside_video_start_flag():
    """Requirement 3(iii):
    A track present at the first frame gets first_seen_at_video_start=True.
    A track first seen mid-video inside a zone gets ZONE_FIRST_SEEN_INSIDE with
    first_seen_at_video_start=False.
    """
    rows = (
        # Track 1: frames 0 to 30, starts at frame 0 inside zone
        [_row(f, 1, 110.0, 120.0, 170.0, 170.0) for f in range(0, 31)]
        # Track 2: frames 40 to 70, first seen mid-video inside zone
        + [_row(f, 2, 120.0, 120.0, 180.0, 170.0) for f in range(40, 71)]
    )
    events = _run(rows, [ZONE])

    t1_presents = [e for e in events if e.track_id == 1 and e.event_type == ZONE_FIRST_SEEN_INSIDE]
    t2_presents = [e for e in events if e.track_id == 2 and e.event_type == ZONE_FIRST_SEEN_INSIDE]

    assert len(t1_presents) == 1
    assert t1_presents[0].starts_inside is True
    assert t1_presents[0].first_seen_at_video_start is True

    assert len(t2_presents) == 1
    assert t2_presents[0].starts_inside is True
    assert t2_presents[0].first_seen_at_video_start is False


# ── Long-coexisting adjacent tracks NOT flagged as hand-off ───────────────────


def test_long_coexisting_adjacent_tracks_not_flagged_as_handoff():
    """Requirement 3(i):
    Two long-coexisting adjacent people with a peak IoU of 0.21 must NOT be flagged as hand-off.
    Coexistence time is well above max_coexist_s (5.0 s > 1.5 s).
    Rule (c) must not flag them.
    """
    # Box A: (100, 100, 160, 160), area 3600
    # Box B with peak IoU 0.21: (139.17, 100, 199.17, 160)
    # inter_w = 20.83, inter_area = 1249.8, union = 5950.2, IoU = 0.21004
    box_a = (100.0, 100.0, 160.0, 160.0)
    box_b_peak = (139.17, 100.0, 199.17, 160.0)
    box_b_away = (220.0, 100.0, 280.0, 160.0)

    rows: list[TrackRow] = []
    # Both tracks exist for frames 0 to 50 (5.0s at 10 fps)
    for f in range(0, 51):
        rows.append(_row(f, 1, *box_a))
        b = box_b_peak if f == 10 else box_b_away
        rows.append(_row(f, 2, *b))

    cfg = EventConfig(max_coexist_s=1.5, handoff_iou_thresh=0.2)
    events = _run(rows, [ZONE], cfg=cfg)

    t1_events = [e for e in events if e.track_id == 1]
    t2_events = [e for e in events if e.track_id == 2]

    assert len(t1_events) > 0
    assert len(t2_events) > 0

    assert not any("handoff" in e.uncertainty_reasons for e in t1_events), (
        f"Track 1 should not have handoff flag, got: {[e.uncertainty_reasons for e in t1_events]}"
    )
    assert not any("handoff" in e.uncertainty_reasons for e in t2_events), (
        f"Track 2 should not have handoff flag, got: {[e.uncertainty_reasons for e in t2_events]}"
    )


# ── Succession hand-off flags both tracks (#3 -> #9 style) ───────────────────


def test_handoff_succession_flags_both_tracks():
    """Requirement 3(ii):
    #3 -> #9 style succession: Track A ends while Track B starts, with coexistence
    <= max_coexist_s (e.g. 1.2s <= 1.5s) and peak IoU 0.49 in one frame: MUST flag both.
    """
    # Track 3: frames 0 to 48 (0.0s to 4.8s)
    # Track 9: frames 36 to 80 (3.6s to 8.0s)
    # Coexistence: 4.8 - 3.6 = 1.2s <= max_coexist_s (1.5s)
    # B starts within [A.end_time - max_coexist_s, A.end_time + gap_s_thresh] = [3.3, 5.8]
    # At frame 36, IoU is 0.49:
    # box_a = (100, 100, 160, 160), box_b = (120.54, 100, 180.54, 160)
    # inter_w = 39.46, inter_area = 2367.6, union = 4832.4, IoU = 0.490
    box_a = (100.0, 100.0, 160.0, 160.0)
    box_b_peak = (120.54, 100.0, 180.54, 160.0)
    box_b_away = (135.0, 100.0, 195.0, 160.0)

    rows: list[TrackRow] = []
    for f in range(0, 49):
        rows.append(_row(f, 3, *box_a))
    for f in range(36, 81):
        b = box_b_peak if f == 36 else box_b_away
        rows.append(_row(f, 9, *b))

    cfg = EventConfig(handoff_iou_thresh=0.2, max_coexist_s=1.5)
    events = _run(rows, [ZONE], cfg=cfg)

    t3_events = [e for e in events if e.track_id == 3]
    t9_events = [e for e in events if e.track_id == 9]

    assert len(t3_events) > 0
    assert len(t9_events) > 0

    assert all(e.identity_uncertain for e in t3_events), "All Track 3 events must be flagged"
    assert all(e.identity_uncertain for e in t9_events), "All Track 9 events must be flagged"

    assert any("handoff candidate with #9" in e.uncertainty_reasons for e in t3_events)
    assert any("handoff candidate with #3" in e.uncertainty_reasons for e in t9_events)


# ── False split pattern flags short track ─────────────────────────────────────


def test_false_split_flags_short_track():
    """A track shorter than 2.0 s that overlaps a longer track must be flagged
    identity_uncertain=True with a possible false split reason.
    """
    # Track 1: duration 5.0 s (frames 0 to 50)
    # Track 2: duration 1.2 s (frames 10 to 22), < 2.0 s
    # Boxes overlap at frame 15
    box_1 = (110.0, 110.0, 170.0, 170.0)
    box_2 = (115.0, 115.0, 175.0, 175.0)

    rows: list[TrackRow] = []
    for f in range(0, 51):
        rows.append(_row(f, 1, *box_1))
    for f in range(10, 23):
        rows.append(_row(f, 2, *box_2))

    events = _run(rows, [ZONE])

    t2_events = [e for e in events if e.track_id == 2]
    assert len(t2_events) > 0
    assert all(e.identity_uncertain for e in t2_events)
    assert any("possible false split with #1" in e.uncertainty_reasons for e in t2_events)


# ── Restricted zone: observed entry vs starting inside ────────────────────────


def test_restricted_zone_emits_restricted_entry():
    """An observed outside->inside entry into a restricted zone fires RESTRICTED_ENTRY."""
    rows = (
        [_row(f, 1, 30.0, 130.0, 100.0, 170.0) for f in range(0, 10)]      # outside
        + [_row(f, 1, 110.0, 120.0, 170.0, 170.0) for f in range(10, 30)]  # inside
        + [_row(f, 1, 30.0, 130.0, 100.0, 170.0) for f in range(30, 60)]   # outside
    )
    events = _run(rows, [RESTRICTED_ZONE])

    enters = _filter(events, ZONE_ENTER)
    restricted_entry = _filter(events, RESTRICTED_ENTRY)
    restricted_start = _filter(events, RESTRICTED_FIRST_SEEN_INSIDE)

    assert len(enters) == 1
    assert len(restricted_entry) == 1, f"Expected 1 RESTRICTED_ENTRY, got {len(restricted_entry)}"
    assert len(restricted_start) == 0
    assert restricted_entry[0].zone == "DangerZone"


def test_restricted_zone_starts_inside_emits_restricted_first_seen_inside():
    """A track already inside a restricted zone at its first frame emits RESTRICTED_FIRST_SEEN_INSIDE."""
    rows = (
        [_row(f, 1, 110.0, 120.0, 170.0, 170.0) for f in range(0, 25)]  # inside from start
        + [_row(f, 1, 30.0, 130.0, 100.0, 170.0) for f in range(25, 60)]  # outside
    )
    events = _run(rows, [RESTRICTED_ZONE])

    presents = _filter(events, ZONE_FIRST_SEEN_INSIDE)
    enters = _filter(events, ZONE_ENTER)
    restricted_entry = _filter(events, RESTRICTED_ENTRY)
    restricted_start = _filter(events, RESTRICTED_FIRST_SEEN_INSIDE)

    assert len(presents) == 1
    assert len(enters) == 0
    assert len(restricted_start) == 1, f"Expected 1 RESTRICTED_FIRST_SEEN_INSIDE, got {len(restricted_start)}"
    assert restricted_start[0].first_seen_at_video_start is True
    assert len(restricted_entry) == 0
    assert restricted_start[0].zone == "DangerZone"


# ── Two people overlapping in the same zone ───────────────────────────────────


def test_two_people_overlapping_in_zone():
    """Two tracks overlap significantly during a zone event.
    The events should be identity_uncertain=True.
    """
    in_box_a = (110.0, 120.0, 170.0, 170.0)   # fp=(140, 170) inside
    in_box_b = (115.0, 122.0, 175.0, 172.0)   # almost identical, heavily overlapping

    rows = (
        [_row(f, 1, 30.0, 130.0, 100.0, 170.0) for f in range(0, 10)]
        + [_row(f, 2, 30.0, 130.0, 100.0, 170.0) for f in range(0, 10)]
        + [_row(f, 1, *in_box_a) for f in range(10, 50)]
        + [_row(f, 2, *in_box_b) for f in range(10, 50)]
        + [_row(f, 1, 30.0, 130.0, 100.0, 170.0) for f in range(50, 90)]
        + [_row(f, 2, 30.0, 130.0, 100.0, 170.0) for f in range(50, 90)]
    )
    events = _run(rows, [ZONE])

    enters = _filter(events, ZONE_ENTER)
    assert len(enters) == 2, f"Expected 2 ZONE_ENTER events (one per track), got {len(enters)}"
    uncertain_enters = [e for e in enters if e.identity_uncertain]
    assert len(uncertain_enters) > 0, "Overlapping tracks should trigger identity_uncertain"


# ── Non-overlapping tracks not flagged for box overlap ────────────────────────


def test_non_overlapping_tracks_not_flagged():
    """Two tracks in the same zone but far apart should NOT have 'overlap' in reasons."""
    # Track 1: left side of zone, Track 2: right side, no box overlap
    rows = (
        [_row(f, 1, 30.0, 130.0, 90.0, 170.0) for f in range(0, 10)]
        + [_row(f, 2, 30.0, 130.0, 90.0, 170.0) for f in range(0, 10)]
        + [_row(f, 1, 105.0, 120.0, 130.0, 170.0) for f in range(10, 50)]    # fp=(117, 170)
        + [_row(f, 2, 160.0, 120.0, 195.0, 170.0) for f in range(10, 50)]  # fp=(177, 170)
        + [_row(f, 1, 30.0, 130.0, 90.0, 170.0) for f in range(50, 90)]
        + [_row(f, 2, 30.0, 130.0, 90.0, 170.0) for f in range(50, 90)]
    )
    events = _run(rows, [ZONE])
    enters = _filter(events, ZONE_ENTER)
    assert len(enters) == 2
    for e in enters:
        assert "overlap" not in e.uncertainty_reasons


# ── Gap fragmentation flagging ────────────────────────────────────────────────


def test_gap_in_track_triggers_fragmentation_flag():
    """A track with a gap > gap_s_thresh inside the event's time span should be flagged."""
    in_box = (110.0, 120.0, 170.0, 170.0)   # fp = (140, 170) → inside
    out_box = (30.0, 130.0, 100.0, 170.0)   # outside zone

    rows = (
        [_row(f, 1, *in_box) for f in range(0, 20)]       # inside
        # GAP: frames 20-39 missing from CSV (track lost)
        + [_row(f, 1, *in_box) for f in range(40, 60)]    # inside again after gap
        + [_row(f, 1, *out_box) for f in range(60, 100)]  # outside
    )
    cfg = EventConfig(
        min_enter_s=0.5,
        exit_grace_s=0.5,
        min_track_len_s=1.0,
        occupied_merge_gap_s=2.0,
        overlap_iou_thresh=0.3,
        overlap_fraction_thresh=0.2,
        gap_s_thresh=0.5,
    )
    events = _run(rows, [ZONE], cfg=cfg)

    zone_events = [e for e in events if e.zone is not None and e.identity_uncertain]
    assert len(zone_events) > 0, "Gap in track should trigger identity_uncertain"
    assert any("gap" in e.uncertainty_reasons for e in zone_events)


# ── ZONE_OCCUPIED merging ─────────────────────────────────────────────────────


def test_zone_occupied_merges_close_intervals():
    """When two person intervals in a zone are close (< occupied_merge_gap_s),
    they should be merged into a single ZONE_OCCUPIED event.
    """
    in_box = (110.0, 120.0, 170.0, 170.0)
    out_box = (30.0, 130.0, 90.0, 170.0)

    rows = (
        [_row(f, 1, *in_box) for f in range(0, 20)]
        + [_row(f, 1, *out_box) for f in range(20, 80)]
        + [_row(f, 2, *out_box) for f in range(0, 30)]
        + [_row(f, 2, *in_box) for f in range(30, 60)]
        + [_row(f, 2, *out_box) for f in range(60, 100)]
    )
    events = _run(rows, [ZONE])
    occupied = _filter(events, ZONE_OCCUPIED)
    assert len(occupied) == 1, f"Expected 1 merged ZONE_OCCUPIED, got {len(occupied)}"
    assert occupied[0].zone == "TestZone"


def test_zone_occupied_not_merged_when_gap_large():
    """When gap between intervals exceeds occupied_merge_gap_s, they stay separate."""
    in_box = (110.0, 120.0, 170.0, 170.0)
    out_box = (30.0, 130.0, 90.0, 170.0)

    rows = (
        [_row(f, 1, *in_box) for f in range(0, 20)]
        + [_row(f, 1, *out_box) for f in range(20, 100)]
        + [_row(f, 2, *out_box) for f in range(0, 60)]
        + [_row(f, 2, *in_box) for f in range(60, 80)]
        + [_row(f, 2, *out_box) for f in range(80, 120)]
    )
    events = _run(rows, [ZONE])
    occupied = _filter(events, ZONE_OCCUPIED)
    assert len(occupied) == 2, f"Expected 2 separate ZONE_OCCUPIED, got {len(occupied)}"


# ── EventConfig from dict ─────────────────────────────────────────────────────


def test_event_config_from_dict():
    d = {
        "min_enter_s": 0.5,
        "exit_grace_s": 1.5,
        "min_track_len_s": 2.0,
        "occupied_merge_gap_s": 3.0,
        "overlap_iou_thresh": 0.4,
        "overlap_fraction_thresh": 0.3,
        "gap_s_thresh": 0.8,
        "handoff_iou_thresh": 0.25,
        "max_coexist_s": 1.8,
    }
    cfg = EventConfig.from_dict(d)
    assert math.isclose(cfg.min_enter_s, 0.5)
    assert math.isclose(cfg.exit_grace_s, 1.5)
    assert math.isclose(cfg.gap_s_thresh, 0.8)
    assert math.isclose(cfg.handoff_iou_thresh, 0.25)
    assert math.isclose(cfg.max_coexist_s, 1.8)


# ── CCTVEvent legacy fields stay in sync ──────────────────────────────────────


def test_cctvevent_legacy_aliases():
    ev = CCTVEvent(
        camera_id="cam",
        event_type=ZONE_ENTER,
        start_time_s=3.5,
        end_time_s=7.2,
        zone="ZA",
    )
    assert math.isclose(ev.start_time, 3.5)
    assert math.isclose(ev.end_time, 7.2)
    assert ev.zone_name == "ZA"
    assert math.isclose(ev.duration_s, 7.2 - 3.5, rel_tol=1e-5)


# ── Rule 13: make_events fails when zones file is missing ─────────────────────


def test_make_events_fails_when_zones_missing(tmp_path):
    """make_events must fail (exit non-zero) when zones file is missing, and never create a default."""
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from make_events import main

    fake_tracks = tmp_path / "tracks.csv"
    fake_tracks.write_text("frame_idx,time_s,track_id,cls,conf,x1,y1,x2,y2\n", encoding="utf-8")
    missing_zones = tmp_path / "nonexistent_zones.json"
    out_dir = tmp_path / "out"

    ret = main([
        "--tracks", str(fake_tracks),
        "--zones", str(missing_zones),
        "--out-dir", str(out_dir),
        "--camera-id", "test_cam",
        "--video", str(tmp_path / "fake.mp4"),
    ])
    assert ret != 0, "make_events should return non-zero exit code when zones file is missing"
    assert not missing_zones.exists(), "make_events must never auto-generate or create a missing zones file"
