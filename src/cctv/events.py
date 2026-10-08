"""Event detection logic: converts tracks.csv + zone configs into discrete events.

Rules observed
--------------
Rule 2 : Never fabricate. All events come from rows in tracks.csv. If a track is
         not in the CSV, it does not appear in output.
Rule 3 : Every event carries camera_id, start/end timestamp, start/end frame, and
         video_path as evidence pointers.  det_conf_mean is the arithmetic mean of
         per-frame detector confidence for the supporting frames — it is NOT the
         probability that the event is correct.
Rule 4 : type hints, pathlib, logging (not print), no magic numbers.
Rule 9 : called by scripts/make_events.py which also runs verification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)


# ─── Shared constants ─────────────────────────────────────────────────────────

# Event type strings — used as-is in the CSV.  Adding new types here is a
# deliberate interface extension and must be communicated (Rule 10).
PERSON_APPEARED = "PERSON_APPEARED"
PERSON_LEFT_VIEW = "PERSON_LEFT_VIEW"
ZONE_ENTER = "ZONE_ENTER"
ZONE_FIRST_SEEN_INSIDE = "ZONE_FIRST_SEEN_INSIDE"
ZONE_EXIT = "ZONE_EXIT"
ZONE_OCCUPIED = "ZONE_OCCUPIED"
ZONE_VISIT = "ZONE_VISIT"
RESTRICTED_ENTRY = "RESTRICTED_ENTRY"
RESTRICTED_FIRST_SEEN_INSIDE = "RESTRICTED_FIRST_SEEN_INSIDE"

# Backward-compatibility aliases
ZONE_PRESENT_AT_START = ZONE_FIRST_SEEN_INSIDE
RESTRICTED_PRESENT_AT_START = RESTRICTED_FIRST_SEEN_INSIDE


# ─── Event dataclass ──────────────────────────────────────────────────────────


@dataclass
class CCTVEvent:
    """A detected CCTV event (Rule 3).

    Attributes
    ----------
    event_id          : Unique sequential integer within a processing run.
    camera_id         : Camera identifier (from zones JSON or CLI flag).
    event_type        : One of PERSON_APPEARED, PERSON_LEFT_VIEW, ZONE_ENTER,
                        ZONE_FIRST_SEEN_INSIDE, ZONE_EXIT, ZONE_OCCUPIED,
                        ZONE_VISIT, RESTRICTED_ENTRY, RESTRICTED_FIRST_SEEN_INSIDE.
    zone              : Zone name (None for PERSON_APPEARED/PERSON_LEFT_VIEW
                        and in ZONE_OCCUPIED it is still set to the zone name).
    track_id          : Tracker-assigned ID (None only for ZONE_OCCUPIED).
    start_time_s      : Start timestamp in seconds from video start (original frame).
    end_time_s        : End timestamp in seconds from video start (original frame).
    start_frame       : Original frame index at event start (evidence pointer).
    end_frame         : Original frame index at event end (evidence pointer).
    duration_s        : Dwell / stay duration in seconds (for ZONE_EXIT and ZONE_VISIT).
    det_conf_mean     : Arithmetic mean of per-frame detector confidence for
                        supporting frames.  This is the mean detector score —
                        NOT the probability that the event is correct.
    video_path        : Path to source video (evidence pointer).
    open_ended        : True when the track was still inside the zone at the
                        last processed frame; the true end time is unknown.
    starts_inside     : True when the first in-zone frame is also the track's
                        first appearance — the true entry time is unknown.
    first_seen_at_video_start: True if the track's first processed frame is the
                        1st or 2nd processed frame of the video (person was in
                        view when recording began); False otherwise (mid-video).
    exit_confirmed_time_s: Timestamp in seconds when exit grace period expired.
    duration_is_lower_bound: True when the duration is a lower bound (e.g. visit
                        started inside or ended inside).
    identity_uncertain: True when either (a) significant box overlap with
                        another track suggests the ID might be wrong, (b) a
                        gap inside event span suggests fragmentation, (c) a
                        succession hand-off candidate is detected, or (d) a
                        short track false split is detected.
    uncertainty_reasons: Human-readable explanation. Empty string when False.

    Extra legacy fields kept for backward compatibility with CCTVEvent users
    in db.py and test_query.py:
    class_name, zone_name, start_time, end_time, extra_data,
    det_conf_min, det_conf_max.
    """

    # Primary schema
    event_id: int = 0
    camera_id: str = ""
    event_type: str = ""
    zone: str | None = None
    track_id: int | None = None
    start_time_s: float = 0.0
    end_time_s: float = 0.0
    start_frame: int = 0
    end_frame: int = 0
    duration_s: float = 0.0
    det_conf_mean: float = 0.0
    video_path: str = ""
    open_ended: bool = False
    starts_inside: bool = False
    first_seen_at_video_start: bool = False
    exit_confirmed_time_s: float | None = None
    duration_is_lower_bound: bool = False
    identity_uncertain: bool = False
    uncertainty_reasons: str = ""

    # Legacy fields (preserved so db.py / test_query.py still compile)
    class_name: str = "person"
    zone_name: str = ""
    start_time: float = 0.0   # alias for start_time_s
    end_time: float = 0.0     # alias for end_time_s
    det_conf_min: float = 0.0
    det_conf_max: float = 0.0
    extra_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Keep legacy aliases in sync
        if self.start_time == 0.0 and self.start_time_s != 0.0:
            self.start_time = self.start_time_s
        elif self.start_time_s == 0.0 and self.start_time != 0.0:
            self.start_time_s = self.start_time

        if self.end_time == 0.0 and self.end_time_s != 0.0:
            self.end_time = self.end_time_s
        elif self.end_time_s == 0.0 and self.end_time != 0.0:
            self.end_time_s = self.end_time

        if self.zone_name == "" and self.zone is not None:
            self.zone_name = self.zone
        elif self.zone is None and self.zone_name != "":
            self.zone = self.zone_name

        if self.duration_s == 0.0 and self.end_time_s > self.start_time_s:
            self.duration_s = round(self.end_time_s - self.start_time_s, 4)


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class EventConfig:
    """Detection thresholds for event generation.  All values are configurable
    via configs/events.yaml — no magic numbers in code (Rule 4).
    """
    min_enter_s: float = 1.0         # footpoint must be in zone this long before ZONE_ENTER fires
    exit_grace_s: float = 2.0        # footpoint must be OUTSIDE for this long before ZONE_EXIT fires
    min_track_len_s: float = 1.0     # tracks shorter than this are noise and are dropped
    occupied_merge_gap_s: float = 2.0  # merge ZONE_OCCUPIED intervals closer than this
    overlap_iou_thresh: float = 0.3  # IoU threshold for "overlap" uncertainty check
    overlap_fraction_thresh: float = 0.2  # fraction of supporting frames that must have overlapping box
    gap_s_thresh: float = 1.0        # gap inside event span that triggers fragmentation flag
    handoff_iou_thresh: float = 0.2  # IoU threshold for handoff candidate uncertainty check (rule c)
    max_coexist_s: float = 1.5       # max coexistence duration for succession handoff candidate (rule c)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EventConfig":
        return cls(
            min_enter_s=float(d.get("min_enter_s", 1.0)),
            exit_grace_s=float(d.get("exit_grace_s", 2.0)),
            min_track_len_s=float(d.get("min_track_len_s", 1.0)),
            occupied_merge_gap_s=float(d.get("occupied_merge_gap_s", 2.0)),
            overlap_iou_thresh=float(d.get("overlap_iou_thresh", 0.3)),
            overlap_fraction_thresh=float(d.get("overlap_fraction_thresh", 0.2)),
            gap_s_thresh=float(d.get("gap_s_thresh", 1.0)),
            handoff_iou_thresh=float(d.get("handoff_iou_thresh", 0.2)),
            max_coexist_s=float(d.get("max_coexist_s", 1.5)),
        )


# ─── Track row types ─────────────────────────────────────────────────────────


class TrackRow(NamedTuple):
    """One row from tracks.csv."""
    frame_idx: int
    time_s: float
    track_id: int
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


# ─── Geometry helpers ─────────────────────────────────────────────────────────


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union of two XYXY boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


# ─── Identity uncertainty detection ──────────────────────────────────────────


def compute_track_level_uncertainty(
    tracks: dict[int, list[TrackRow]],
    cfg: EventConfig,
) -> dict[int, list[str]]:
    """Compute track-level uncertainty reasons across all active tracks.

    Conditions checked:
    (c) handoff candidate: succession (not coexistence). Track B starts within
        [A.end_time - max_coexist_s, A.end_time + gap_s_thresh], total time
        during which A and B are both present is <= max_coexist_s, and their
        boxes overlap with IoU >= handoff_iou_thresh in at least one frame.
        Flags BOTH tracks with reason e.g. "handoff candidate with #9 (IoU 0.49)".
        Two tracks that coexist longer than max_coexist_s are NOT a hand-off;
        they are covered by rule (a) (overlap fraction) only.
    (d) false split: a track shorter than 2.0 s that overlaps a longer track;
        flags the shorter track with reason e.g. "possible false split with #9".
    """
    track_reasons: dict[int, list[str]] = {tid: [] for tid in tracks}

    tids = sorted(tracks.keys())
    for i, tid_x in enumerate(tids):
        trows_x = tracks[tid_x]
        x_start = trows_x[0].time_s
        x_end = trows_x[-1].time_s
        dur_x = x_end - x_start
        boxes_x = {r.frame_idx: (r.x1, r.y1, r.x2, r.y2) for r in trows_x}

        for tid_y in tids[i + 1:]:
            trows_y = tracks[tid_y]
            y_start = trows_y[0].time_s
            y_end = trows_y[-1].time_s
            dur_y = y_end - y_start
            boxes_y = {r.frame_idx: (r.x1, r.y1, r.x2, r.y2) for r in trows_y}

            common_frames = set(boxes_x.keys()) & set(boxes_y.keys())
            max_iou = 0.0
            for f in common_frames:
                val = _iou(boxes_x[f], boxes_y[f])
                if val > max_iou:
                    max_iou = val

            # Total coexistence time during which both tracks are present
            coexist_s = max(0.0, min(x_end, y_end) - max(x_start, y_start))

            # (c) Succession handoff: check X -> Y and Y -> X
            succ_xy = (x_end - cfg.max_coexist_s <= y_start <= x_end + cfg.gap_s_thresh)
            succ_yx = (y_end - cfg.max_coexist_s <= x_start <= y_end + cfg.gap_s_thresh)

            if (succ_xy or succ_yx) and (coexist_s <= cfg.max_coexist_s) and (max_iou >= cfg.handoff_iou_thresh):
                track_reasons[tid_x].append(f"handoff candidate with #{tid_y} (IoU {max_iou:.2f})")
                track_reasons[tid_y].append(f"handoff candidate with #{tid_x} (IoU {max_iou:.2f})")

            # (d) False split check: track < 2.0 s overlapping a longer track
            if len(common_frames) > 0 and max_iou > 0.0:
                if dur_x < 2.0 and dur_y > dur_x:
                    track_reasons[tid_x].append(f"possible false split with #{tid_y}")
                if dur_y < 2.0 and dur_x > dur_y:
                    track_reasons[tid_y].append(f"possible false split with #{tid_x}")

    return track_reasons


def check_identity_uncertainty(
    track_id: int,
    support_frames: list[TrackRow],
    all_rows_by_frame: dict[int, list[TrackRow]],
    track_reasons: dict[int, list[str]],
    cfg: EventConfig,
) -> tuple[bool, str]:
    """Return (identity_uncertain, uncertainty_reasons).

    Conditions checked:
    (a) overlap: in >= overlap_fraction_thresh fraction of supporting frames,
        this track's box has IoU >= overlap_iou_thresh with another track's box.
    (b) fragmentation / gap: the track has a gap > gap_s_thresh inside the
        event's supporting frames.
    (c) handoff candidate: precomputed in track_reasons (succession only).
    (d) false split: precomputed in track_reasons.
    """
    reasons: list[str] = list(track_reasons.get(track_id, []))

    if not support_frames:
        uncertain = len(reasons) > 0
        return uncertain, "; ".join(reasons)

    # ── (a) overlap check ───────────────────────────────────────────────────
    overlap_count = 0
    for row in support_frames:
        bbox = (row.x1, row.y1, row.x2, row.y2)
        others = all_rows_by_frame.get(row.frame_idx, [])
        for other in others:
            if other.track_id == track_id:
                continue
            other_bbox = (other.x1, other.y1, other.x2, other.y2)
            if _iou(bbox, other_bbox) >= cfg.overlap_iou_thresh:
                overlap_count += 1
                break

    overlap_frac = overlap_count / len(support_frames)
    if overlap_frac >= cfg.overlap_fraction_thresh:
        reasons.append(f"overlap {overlap_frac:.2f}")

    # ── (b) fragmentation check ──────────────────────────────────────────────
    times = sorted(r.time_s for r in support_frames)
    if len(times) >= 2:
        for i in range(len(times) - 1):
            gap = times[i + 1] - times[i]
            if gap > cfg.gap_s_thresh:
                reasons.append(f"gap {gap:.2f}s")
                break

    # Deduplicate reasons while preserving order
    unique_reasons: list[str] = []
    seen = set()
    for r in reasons:
        if r not in seen:
            seen.add(r)
            unique_reasons.append(r)

    uncertain = len(unique_reasons) > 0
    return uncertain, "; ".join(unique_reasons)


# ─── Main event detection ─────────────────────────────────────────────────────


def detect_events(
    rows: list[TrackRow],
    zones: list[Any],  # list[cctv.zones.Zone]  – avoid circular import
    camera_id: str,
    video_path: str | Path,
    video_fps: float,
    cfg: EventConfig,
) -> list[CCTVEvent]:
    """Convert a list of TrackRow objects into discrete CCTVEvents.

    Parameters
    ----------
    rows       : All rows from tracks.csv for this camera/video, sorted by time.
    zones      : Zone objects (from cctv.zones).
    camera_id  : Camera identifier string.
    video_path : Path to source video (stored in every event as evidence pointer).
    video_fps  : Source video FPS (used for stride-aware gap detection).
    cfg        : Detection thresholds.

    Returns
    -------
    Sorted list of CCTVEvent objects.
    """
    from cctv.zones import footpoint_of_box  # local import avoids circular

    video_path_str = str(video_path)
    events: list[CCTVEvent] = []
    event_counter = 0

    def next_id() -> int:
        nonlocal event_counter
        event_counter += 1
        return event_counter

    # ── Index rows ────────────────────────────────────────────────────────────
    if not rows:
        return []

    # Group by track_id
    tracks: dict[int, list[TrackRow]] = {}
    for r in rows:
        tracks.setdefault(r.track_id, []).append(r)
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r.frame_idx)

    # Filter noise tracks shorter than min_track_len_s
    valid_tracks: dict[int, list[TrackRow]] = {}
    for tid, trows in tracks.items():
        span_s = trows[-1].time_s - trows[0].time_s
        if span_s < cfg.min_track_len_s:
            logger.debug("Track %d dropped as noise (%.2f s < %.2f s)", tid, span_s, cfg.min_track_len_s)
        else:
            valid_tracks[tid] = trows

    # Group valid tracks by frame for overlap checks
    rows_by_frame: dict[int, list[TrackRow]] = {}
    for trows in valid_tracks.values():
        for r in trows:
            rows_by_frame.setdefault(r.frame_idx, []).append(r)

    # Video start frames: first or second processed frame of the video
    processed_video_frames = sorted({r.frame_idx for r in rows})
    if processed_video_frames and processed_video_frames[0] in (0, 1):
        video_start_frames = set(processed_video_frames[:2])
    else:
        video_start_frames = {0, 1}

    # Precompute track-level uncertainty reasons (rules c and d)
    track_reasons = compute_track_level_uncertainty(valid_tracks, cfg)

    # ── PERSON_APPEARED / PERSON_LEFT_VIEW ───────────────────────────────────
    for tid, trows in valid_tracks.items():
        confs = [r.conf for r in trows]
        mean_conf = sum(confs) / len(confs)
        first_seen_at_start = (trows[0].frame_idx in video_start_frames)

        unc_start, rsn_start = check_identity_uncertainty(
            tid, [trows[0]], rows_by_frame, track_reasons, cfg
        )
        appeared = CCTVEvent(
            event_id=next_id(),
            camera_id=camera_id,
            event_type=PERSON_APPEARED,
            zone=None,
            track_id=tid,
            start_time_s=round(trows[0].time_s, 4),
            end_time_s=round(trows[0].time_s, 4),
            start_frame=trows[0].frame_idx,
            end_frame=trows[0].frame_idx,
            duration_s=0.0,
            det_conf_mean=round(mean_conf, 4),
            video_path=video_path_str,
            first_seen_at_video_start=first_seen_at_start,
            identity_uncertain=unc_start,
            uncertainty_reasons=rsn_start,
        )
        events.append(appeared)

        unc_end, rsn_end = check_identity_uncertainty(
            tid, [trows[-1]], rows_by_frame, track_reasons, cfg
        )
        left = CCTVEvent(
            event_id=next_id(),
            camera_id=camera_id,
            event_type=PERSON_LEFT_VIEW,
            zone=None,
            track_id=tid,
            start_time_s=round(trows[-1].time_s, 4),
            end_time_s=round(trows[-1].time_s, 4),
            start_frame=trows[-1].frame_idx,
            end_frame=trows[-1].frame_idx,
            duration_s=0.0,
            det_conf_mean=round(mean_conf, 4),
            video_path=video_path_str,
            first_seen_at_video_start=first_seen_at_start,
            identity_uncertain=unc_end,
            uncertainty_reasons=rsn_end,
        )
        events.append(left)

    # ── Zone events ───────────────────────────────────────────────────────────
    zone_occupancy: dict[str, list[tuple[float, float]]] = {z.name: [] for z in zones}

    for tid, trows in valid_tracks.items():
        first_seen_at_start = (trows[0].frame_idx in video_start_frames)

        for zone in zones:
            in_zone_seq: list[tuple[TrackRow, bool]] = []
            for r in trows:
                fp = footpoint_of_box((r.x1, r.y1, r.x2, r.y2))
                in_z = zone.contains_point(fp)
                in_zone_seq.append((r, in_z))

            if not in_zone_seq:
                continue

            OUTSIDE = "outside"
            ENTERING = "entering"
            INSIDE = "inside"
            LEAVING = "leaving"

            state = OUTSIDE
            seen_outside = False
            enter_candidate: list[TrackRow] = []
            exit_candidate: list[TrackRow] = []
            current_dwell: list[TrackRow] = []
            confirmed_enter_row: TrackRow | None = None
            starts_inside = False

            for row, in_z in in_zone_seq:
                if state == OUTSIDE:
                    if in_z:
                        enter_candidate = [row]
                        state = ENTERING
                    else:
                        seen_outside = True

                elif state == ENTERING:
                    if in_z:
                        enter_candidate.append(row)
                        dur = enter_candidate[-1].time_s - enter_candidate[0].time_s
                        if dur >= cfg.min_enter_s:
                            confirmed_enter_row = enter_candidate[0]
                            current_dwell = list(enter_candidate)
                            starts_inside = not seen_outside
                            state = INSIDE
                    else:
                        enter_candidate = []
                        state = OUTSIDE
                        seen_outside = True

                elif state == INSIDE:
                    if in_z:
                        current_dwell.append(row)
                        exit_candidate = []
                    else:
                        exit_candidate = [row]
                        state = LEAVING

                elif state == LEAVING:
                    if in_z:
                        current_dwell.extend(exit_candidate)
                        current_dwell.append(row)
                        exit_candidate = []
                        state = INSIDE
                    else:
                        exit_candidate.append(row)
                        out_dur = exit_candidate[-1].time_s - exit_candidate[0].time_s
                        if out_dur >= cfg.exit_grace_s:
                            # ── Observed Exit Confirmed ──
                            last_inside_row = current_dwell[-1]
                            exit_confirmed_time_s = exit_candidate[-1].time_s
                            start_row = trows[0] if starts_inside else confirmed_enter_row
                            assert start_row is not None
                            dwell_dur = round(last_inside_row.time_s - start_row.time_s, 4)

                            support = current_dwell
                            uncertain, reasons = check_identity_uncertainty(
                                tid, support, rows_by_frame, track_reasons, cfg
                            )
                            confs = [r.conf for r in support]
                            mean_c = sum(confs) / len(confs) if confs else 0.0

                            # 1. ZONE_FIRST_SEEN_INSIDE or ZONE_ENTER
                            if starts_inside:
                                ev_start = CCTVEvent(
                                    event_id=next_id(),
                                    camera_id=camera_id,
                                    event_type=ZONE_FIRST_SEEN_INSIDE,
                                    zone=zone.name,
                                    track_id=tid,
                                    start_time_s=round(trows[0].time_s, 4),
                                    end_time_s=round(trows[0].time_s, 4),
                                    start_frame=trows[0].frame_idx,
                                    end_frame=trows[0].frame_idx,
                                    duration_s=0.0,
                                    det_conf_mean=round(mean_c, 4),
                                    video_path=video_path_str,
                                    starts_inside=True,
                                    first_seen_at_video_start=first_seen_at_start,
                                    open_ended=False,
                                    duration_is_lower_bound=False,
                                    identity_uncertain=uncertain,
                                    uncertainty_reasons=reasons,
                                )
                                events.append(ev_start)
                                if zone.zone_type.lower() == "restricted":
                                    ev_restr_start = CCTVEvent(
                                        event_id=next_id(),
                                        camera_id=camera_id,
                                        event_type=RESTRICTED_FIRST_SEEN_INSIDE,
                                        zone=zone.name,
                                        track_id=tid,
                                        start_time_s=round(trows[0].time_s, 4),
                                        end_time_s=round(trows[0].time_s, 4),
                                        start_frame=trows[0].frame_idx,
                                        end_frame=trows[0].frame_idx,
                                        duration_s=0.0,
                                        det_conf_mean=round(mean_c, 4),
                                        video_path=video_path_str,
                                        starts_inside=True,
                                        first_seen_at_video_start=first_seen_at_start,
                                        open_ended=False,
                                        duration_is_lower_bound=False,
                                        identity_uncertain=uncertain,
                                        uncertainty_reasons=reasons,
                                    )
                                    events.append(ev_restr_start)
                            else:
                                ev_enter = CCTVEvent(
                                    event_id=next_id(),
                                    camera_id=camera_id,
                                    event_type=ZONE_ENTER,
                                    zone=zone.name,
                                    track_id=tid,
                                    start_time_s=round(confirmed_enter_row.time_s, 4),
                                    end_time_s=round(confirmed_enter_row.time_s, 4),
                                    start_frame=confirmed_enter_row.frame_idx,
                                    end_frame=confirmed_enter_row.frame_idx,
                                    duration_s=0.0,
                                    det_conf_mean=round(mean_c, 4),
                                    video_path=video_path_str,
                                    starts_inside=False,
                                    first_seen_at_video_start=first_seen_at_start,
                                    open_ended=False,
                                    duration_is_lower_bound=False,
                                    identity_uncertain=uncertain,
                                    uncertainty_reasons=reasons,
                                )
                                events.append(ev_enter)
                                if zone.zone_type.lower() == "restricted":
                                    ev_restr = CCTVEvent(
                                        event_id=next_id(),
                                        camera_id=camera_id,
                                        event_type=RESTRICTED_ENTRY,
                                        zone=zone.name,
                                        track_id=tid,
                                        start_time_s=round(confirmed_enter_row.time_s, 4),
                                        end_time_s=round(confirmed_enter_row.time_s, 4),
                                        start_frame=confirmed_enter_row.frame_idx,
                                        end_frame=confirmed_enter_row.frame_idx,
                                        duration_s=0.0,
                                        det_conf_mean=round(mean_c, 4),
                                        video_path=video_path_str,
                                        starts_inside=False,
                                        first_seen_at_video_start=first_seen_at_start,
                                        open_ended=False,
                                        duration_is_lower_bound=False,
                                        identity_uncertain=uncertain,
                                        uncertainty_reasons=reasons,
                                    )
                                    events.append(ev_restr)

                            # 2. ZONE_EXIT (observed outside transition)
                            ev_exit = CCTVEvent(
                                event_id=next_id(),
                                camera_id=camera_id,
                                event_type=ZONE_EXIT,
                                zone=zone.name,
                                track_id=tid,
                                start_time_s=round(last_inside_row.time_s, 4),
                                end_time_s=round(last_inside_row.time_s, 4),
                                start_frame=last_inside_row.frame_idx,
                                end_frame=last_inside_row.frame_idx,
                                duration_s=dwell_dur,
                                det_conf_mean=round(mean_c, 4),
                                video_path=video_path_str,
                                starts_inside=starts_inside,
                                first_seen_at_video_start=first_seen_at_start,
                                open_ended=False,
                                exit_confirmed_time_s=round(exit_confirmed_time_s, 4),
                                duration_is_lower_bound=starts_inside,
                                identity_uncertain=uncertain,
                                uncertainty_reasons=reasons,
                            )
                            events.append(ev_exit)

                            # 3. ZONE_VISIT
                            ev_visit = CCTVEvent(
                                event_id=next_id(),
                                camera_id=camera_id,
                                event_type=ZONE_VISIT,
                                zone=zone.name,
                                track_id=tid,
                                start_time_s=round(start_row.time_s, 4),
                                end_time_s=round(last_inside_row.time_s, 4),
                                start_frame=start_row.frame_idx,
                                end_frame=last_inside_row.frame_idx,
                                duration_s=dwell_dur,
                                det_conf_mean=round(mean_c, 4),
                                video_path=video_path_str,
                                starts_inside=starts_inside,
                                first_seen_at_video_start=first_seen_at_start,
                                open_ended=False,
                                exit_confirmed_time_s=round(exit_confirmed_time_s, 4),
                                duration_is_lower_bound=starts_inside,
                                identity_uncertain=uncertain,
                                uncertainty_reasons=reasons,
                            )
                            events.append(ev_visit)

                            zone_occupancy[zone.name].append(
                                (start_row.time_s, last_inside_row.time_s)
                            )

                            state = OUTSIDE
                            seen_outside = True
                            enter_candidate = []
                            exit_candidate = []
                            current_dwell = []
                            confirmed_enter_row = None
                            starts_inside = False

            # End of track — check if track was still inside (open_ended)
            if state in (INSIDE, LEAVING) and current_dwell:
                last_inside_row = current_dwell[-1]
                start_row = trows[0] if starts_inside else confirmed_enter_row
                assert start_row is not None
                dwell_dur = round(last_inside_row.time_s - start_row.time_s, 4)

                support = current_dwell
                uncertain, reasons = check_identity_uncertainty(
                    tid, support, rows_by_frame, track_reasons, cfg
                )
                confs = [r.conf for r in support]
                mean_c = sum(confs) / len(confs) if confs else 0.0

                if starts_inside:
                    ev_start = CCTVEvent(
                        event_id=next_id(),
                        camera_id=camera_id,
                        event_type=ZONE_FIRST_SEEN_INSIDE,
                        zone=zone.name,
                        track_id=tid,
                        start_time_s=round(trows[0].time_s, 4),
                        end_time_s=round(trows[0].time_s, 4),
                        start_frame=trows[0].frame_idx,
                        end_frame=trows[0].frame_idx,
                        duration_s=0.0,
                        det_conf_mean=round(mean_c, 4),
                        video_path=video_path_str,
                        starts_inside=True,
                        first_seen_at_video_start=first_seen_at_start,
                        open_ended=True,
                        duration_is_lower_bound=True,
                        identity_uncertain=uncertain,
                        uncertainty_reasons=reasons,
                    )
                    events.append(ev_start)
                    if zone.zone_type.lower() == "restricted":
                        ev_restr_start = CCTVEvent(
                            event_id=next_id(),
                            camera_id=camera_id,
                            event_type=RESTRICTED_FIRST_SEEN_INSIDE,
                            zone=zone.name,
                            track_id=tid,
                            start_time_s=round(trows[0].time_s, 4),
                            end_time_s=round(trows[0].time_s, 4),
                            start_frame=trows[0].frame_idx,
                            end_frame=trows[0].frame_idx,
                            duration_s=0.0,
                            det_conf_mean=round(mean_c, 4),
                            video_path=video_path_str,
                            starts_inside=True,
                            first_seen_at_video_start=first_seen_at_start,
                            open_ended=True,
                            duration_is_lower_bound=True,
                            identity_uncertain=uncertain,
                            uncertainty_reasons=reasons,
                        )
                        events.append(ev_restr_start)
                else:
                    ev_enter = CCTVEvent(
                        event_id=next_id(),
                        camera_id=camera_id,
                        event_type=ZONE_ENTER,
                        zone=zone.name,
                        track_id=tid,
                        start_time_s=round(confirmed_enter_row.time_s, 4),
                        end_time_s=round(confirmed_enter_row.time_s, 4),
                        start_frame=confirmed_enter_row.frame_idx,
                        end_frame=confirmed_enter_row.frame_idx,
                        duration_s=0.0,
                        det_conf_mean=round(mean_c, 4),
                        video_path=video_path_str,
                        starts_inside=False,
                        first_seen_at_video_start=first_seen_at_start,
                        open_ended=True,
                        duration_is_lower_bound=True,
                        identity_uncertain=uncertain,
                        uncertainty_reasons=reasons,
                    )
                    events.append(ev_enter)
                    if zone.zone_type.lower() == "restricted":
                        ev_restr = CCTVEvent(
                            event_id=next_id(),
                            camera_id=camera_id,
                            event_type=RESTRICTED_ENTRY,
                            zone=zone.name,
                            track_id=tid,
                            start_time_s=round(confirmed_enter_row.time_s, 4),
                            end_time_s=round(confirmed_enter_row.time_s, 4),
                            start_frame=confirmed_enter_row.frame_idx,
                            end_frame=confirmed_enter_row.frame_idx,
                            duration_s=0.0,
                            det_conf_mean=round(mean_c, 4),
                            video_path=video_path_str,
                            starts_inside=False,
                            first_seen_at_video_start=first_seen_at_start,
                            open_ended=True,
                            duration_is_lower_bound=True,
                            identity_uncertain=uncertain,
                            uncertainty_reasons=reasons,
                        )
                        events.append(ev_restr)

                # Track ends inside -> do NOT emit ZONE_EXIT.
                # Emit ZONE_VISIT with open_ended=True:
                ev_visit = CCTVEvent(
                    event_id=next_id(),
                    camera_id=camera_id,
                    event_type=ZONE_VISIT,
                    zone=zone.name,
                    track_id=tid,
                    start_time_s=round(start_row.time_s, 4),
                    end_time_s=round(last_inside_row.time_s, 4),
                    start_frame=start_row.frame_idx,
                    end_frame=last_inside_row.frame_idx,
                    duration_s=dwell_dur,
                    det_conf_mean=round(mean_c, 4),
                    video_path=video_path_str,
                    starts_inside=starts_inside,
                    first_seen_at_video_start=first_seen_at_start,
                    open_ended=True,
                    duration_is_lower_bound=True,
                    identity_uncertain=uncertain,
                    uncertainty_reasons=reasons,
                )
                events.append(ev_visit)

                zone_occupancy[zone.name].append(
                    (start_row.time_s, last_inside_row.time_s)
                )

    # ── ZONE_OCCUPIED (merged) ────────────────────────────────────────────────
    for zone in zones:
        intervals = sorted(zone_occupancy[zone.name], key=lambda iv: iv[0])
        if not intervals:
            continue
        merged: list[tuple[float, float]] = []
        cur_start, cur_end = intervals[0]
        for s, e in intervals[1:]:
            if s - cur_end <= cfg.occupied_merge_gap_s:
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))

        for s, e in merged:
            zone_rows = [r for r in rows if s <= r.time_s <= e]
            sf = min(zone_rows, key=lambda r: abs(r.time_s - s)).frame_idx if zone_rows else 0
            ef = max(zone_rows, key=lambda r: abs(r.time_s - e)).frame_idx if zone_rows else 0
            mean_c = (
                sum(r.conf for r in zone_rows) / len(zone_rows) if zone_rows else 0.0
            )
            ev = CCTVEvent(
                event_id=next_id(),
                camera_id=camera_id,
                event_type=ZONE_OCCUPIED,
                zone=zone.name,
                track_id=None,
                start_time_s=round(s, 4),
                end_time_s=round(e, 4),
                start_frame=sf,
                end_frame=ef,
                duration_s=round(e - s, 4),
                det_conf_mean=round(mean_c, 4),
                video_path=video_path_str,
                first_seen_at_video_start=False,
            )
            events.append(ev)

    # Sort by start_time_s
    events.sort(key=lambda ev: (ev.start_time_s, ev.event_type, ev.track_id or -1))

    # Re-number event_ids sequentially after sorting
    for i, ev in enumerate(events, start=1):
        ev.event_id = i

    return events


# ─── CSV I/O ──────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "event_id",
    "camera_id",
    "event_type",
    "zone",
    "track_id",
    "start_time_s",
    "end_time_s",
    "start_frame",
    "end_frame",
    "duration_s",
    "det_conf_mean",
    "video_path",
    "open_ended",
    "starts_inside",
    "first_seen_at_video_start",
    "exit_confirmed_time_s",
    "duration_is_lower_bound",
    "identity_uncertain",
    "uncertainty_reasons",
]


def events_to_dicts(events: list[CCTVEvent]) -> list[dict[str, Any]]:
    """Convert CCTVEvent list to list of dicts with the canonical CSV columns."""
    rows = []
    for ev in events:
        rows.append({
            "event_id": ev.event_id,
            "camera_id": ev.camera_id,
            "event_type": ev.event_type,
            "zone": ev.zone if ev.zone is not None else "",
            "track_id": ev.track_id if ev.track_id is not None else "",
            "start_time_s": ev.start_time_s,
            "end_time_s": ev.end_time_s,
            "start_frame": ev.start_frame,
            "end_frame": ev.end_frame,
            "duration_s": ev.duration_s,
            "det_conf_mean": ev.det_conf_mean,
            "video_path": ev.video_path,
            "open_ended": ev.open_ended,
            "starts_inside": ev.starts_inside,
            "first_seen_at_video_start": ev.first_seen_at_video_start,
            "exit_confirmed_time_s": ev.exit_confirmed_time_s if ev.exit_confirmed_time_s is not None else "",
            "duration_is_lower_bound": ev.duration_is_lower_bound,
            "identity_uncertain": ev.identity_uncertain,
            "uncertainty_reasons": ev.uncertainty_reasons,
        })
    return rows
