#!/usr/bin/env python
"""Convert tracks.csv + zone config into events.csv.

Usage
-----
python scripts/make_events.py \\
    --tracks data/output/cam01_default/tracks.csv \\
    --zones  configs/zones_cam01.json \\
    --events-config configs/events.yaml \\
    --out-dir data/output/cam01_default \\
    --camera-id cam01 \\
    --video data/raw/cam01.mp4

Rules observed
--------------
Rule 2: No fabrication — all events come from rows in tracks.csv.
Rule 3: Every event carries camera_id, start/end timestamp, start/end frame, video_path.
        det_conf_mean = mean detector confidence of supporting frames, NOT event correctness.
Rule 4: type hints, pathlib, logging, no magic numbers (thresholds from events.yaml).
Rule 9: prints real counts after running.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

# ── Ensure src/ is on path ────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cctv.events import (  # noqa: E402
    CSV_COLUMNS,
    EventConfig,
    TrackRow,
    detect_events,
    events_to_dicts,
)
from cctv.zones import load_zones  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("make_events")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert tracks.csv + zone config into events.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tracks", required=True, type=Path,
                   help="Path to tracks.csv from run_tracking.py.")
    p.add_argument("--zones", required=True, type=Path,
                   help="Path to zone config JSON (e.g. configs/zones_cam01.json).")
    p.add_argument("--events-config", type=Path,
                   default=_REPO_ROOT / "configs" / "events.yaml",
                   help="Path to events.yaml with detection thresholds.")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Directory to write events.csv.")
    p.add_argument("--camera-id", required=True,
                   help="Camera identifier string (e.g. cam01).")
    p.add_argument("--video", required=True, type=Path,
                   help="Path to source video (stored as evidence pointer in events).")
    p.add_argument("--video-fps", type=float, default=None,
                   help="Override video FPS (read from run_stats.json if absent).")
    return p


def _load_run_stats(tracks_path: Path) -> dict[str, Any]:
    stats_path = tracks_path.parent / "run_stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            return json.load(f)
    return {}


def _load_tracks(tracks_path: Path) -> list[TrackRow]:
    rows: list[TrackRow] = []
    with open(tracks_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # cls may be stored as a class name string (e.g. 'person') or integer
            cls_raw = r["cls"]
            try:
                cls_int = int(cls_raw)
            except ValueError:
                cls_int = 0  # COCO person = class 0; class name stored instead of ID
            rows.append(TrackRow(
                frame_idx=int(r["frame_idx"]),
                time_s=float(r["time_s"]),
                track_id=int(r["track_id"]),
                cls=cls_int,
                conf=float(r["conf"]),
                x1=float(r["x1"]),
                y1=float(r["y1"]),
                x2=float(r["x2"]),
                y2=float(r["y2"]),
            ))
    return sorted(rows, key=lambda r: (r.frame_idx, r.track_id))


def _load_event_config(config_path: Path) -> EventConfig:
    if not config_path.exists():
        logger.warning("events.yaml not found at %s — using defaults", config_path)
        return EventConfig()
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    events_section = data.get("event_detection", data.get("events", {}))
    return EventConfig.from_dict(events_section)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Validate zones file (Rule 13: human-authored inputs are read-only; never invent defaults)
    if not args.zones.exists():
        msg = f"Zones file missing: {args.zones} (Rule 13: human-authored inputs are read-only and must never be auto-generated)"
        logger.error(msg)
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    try:
        zones_sha256 = compute_sha256(args.zones)
        zones = load_zones(args.zones)
    except Exception as exc:
        logger.error("Failed to load zones from %s: %s", args.zones, exc)
        print(f"ERROR: Failed to load zones from {args.zones}: {exc}", file=sys.stderr)
        return 1

    zone_names = [z.name for z in zones]
    print(f"Zones file  : {args.zones.resolve()}")
    print(f"Zones sha256: {zones_sha256}")
    print(f"Zone names  : {zone_names}")
    logger.info("Zones file: %s (sha256: %s)", args.zones, zones_sha256)
    logger.info("Loaded %d zones: %s", len(zones), zone_names)

    if not args.tracks.exists():
        logger.error("tracks.csv not found: %s", args.tracks)
        return 1
    tracks_sha256 = compute_sha256(args.tracks)

    if not args.video.exists():
        logger.error("Video file not found: %s", args.video)
        return 1

    # Determine FPS
    video_fps = args.video_fps
    stats = _load_run_stats(args.tracks)
    if video_fps is None:
        video_fps = float(stats.get("video_fps", 25.0))
        logger.info("Using video FPS from run_stats.json: %.2f", video_fps)

    # Load config
    event_cfg = _load_event_config(args.events_config)
    logger.info(
        "Event config: min_enter_s=%.1f exit_grace_s=%.1f min_track_len_s=%.1f "
        "occupied_merge_gap_s=%.1f overlap_iou=%.2f overlap_frac=%.2f gap_s=%.1f",
        event_cfg.min_enter_s,
        event_cfg.exit_grace_s,
        event_cfg.min_track_len_s,
        event_cfg.occupied_merge_gap_s,
        event_cfg.overlap_iou_thresh,
        event_cfg.overlap_fraction_thresh,
        event_cfg.gap_s_thresh,
    )

    # Load tracks
    try:
        track_rows = _load_tracks(args.tracks)
    except Exception as exc:
        logger.error("Failed to load tracks from %s: %s", args.tracks, exc)
        return 1
    logger.info("Loaded %d track rows from %s", len(track_rows), args.tracks)

    # Detect events
    logger.info("Detecting events for camera '%s' …", args.camera_id)
    event_list = detect_events(
        rows=track_rows,
        zones=zones,
        camera_id=args.camera_id,
        video_path=args.video,
        video_fps=video_fps,
        cfg=event_cfg,
    )
    logger.info("Detected %d events total", len(event_list))

    # Write events.csv and events_provenance.json (Rule 14)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "events.csv"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(events_to_dicts(event_list))

    logger.info("Saved events.csv to: %s", out_csv)

    # Record provenance (Rule 14)
    provenance = {
        "zones": {
            "path": str(args.zones.resolve()),
            "sha256": zones_sha256,
            "names": zone_names,
        },
        "tracks": {
            "path": str(args.tracks.resolve()),
            "sha256": tracks_sha256,
        },
        "events_config": {
            "path": str(args.events_config.resolve()) if args.events_config.exists() else str(args.events_config),
            "sha256": compute_sha256(args.events_config) if args.events_config.exists() else None,
            "settings": {
                "min_enter_s": event_cfg.min_enter_s,
                "exit_grace_s": event_cfg.exit_grace_s,
                "min_track_len_s": event_cfg.min_track_len_s,
                "occupied_merge_gap_s": event_cfg.occupied_merge_gap_s,
                "overlap_iou_thresh": event_cfg.overlap_iou_thresh,
                "overlap_fraction_thresh": event_cfg.overlap_fraction_thresh,
                "gap_s_thresh": event_cfg.gap_s_thresh,
                "handoff_iou_thresh": event_cfg.handoff_iou_thresh,
                "max_coexist_s": event_cfg.max_coexist_s,
            },
        },
        "tracker_settings": {
            k: stats[k]
            for k in ("tracker", "conf_thresh", "imgsz", "stride", "model_name", "video_fps")
            if k in stats
        },
    }
    prov_path = out_dir / "events_provenance.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    logger.info("Saved events_provenance.json to: %s", prov_path)

    # ── Print counts per event_type (Rule 9) ─────────────────────────────────
    from collections import Counter
    type_counts: Counter[str] = Counter(ev.event_type for ev in event_list)
    uncertain_count = sum(1 for ev in event_list if ev.identity_uncertain)

    print("\n=== Event counts ===")
    for etype, cnt in sorted(type_counts.items()):
        print(f"  {etype:<25} : {cnt}")
    print(f"\n  identity_uncertain=True  : {uncertain_count}")
    print(f"  total events             : {len(event_list)}")

    print("\n=== 10 sample events ===")
    sample = event_list[:10]
    for ev in sample:
        print(
            f"  [{ev.event_id:3d}] {ev.event_type:<20} zone={ev.zone or '-':<15} "
            f"tid={ev.track_id if ev.track_id is not None else '-':<4} "
            f"t={ev.start_time_s:.2f}s–{ev.end_time_s:.2f}s "
            f"uncertain={ev.identity_uncertain}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
