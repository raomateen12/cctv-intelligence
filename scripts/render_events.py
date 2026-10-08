#!/usr/bin/env python
"""Re-render annotated video with zone overlays and event banners.

For every processed frame draws:
  - All zones (coloured polygon outlines + semi-transparent fill)
  - Each tracked person's box + track_id
  - A footpoint dot (bottom-center of the box) coloured by which zone(s) it's in
  - Event banners (e.g. "ZONE_FIRST_SEEN_INSIDE Zone A #1", "ZONE_EXIT Zone A #2") that last for BANNER_DURATION_S

Output: events_overlay.mp4 (H.264, yuv420p)

Rules observed
--------------
Rule 2: No fabrication — draws only from tracks.csv and events.csv.
Rule 4: type hints, pathlib, logging, configurable parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── Ensure src/ is on path ────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cctv.events import TrackRow  # noqa: E402
from cctv.zones import ZONE_COLORS, Zone, draw_zones_on_frame, load_zones  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("render_events")

BANNER_DURATION_S = 2.0  # how long event banners stay on screen
FOOTPOINT_RADIUS = 6

# Styling as per Rule 13 & 14 / requirements:
# Neutral grey for normal tracks, vibrant orange when identity is uncertain
BOX_NORMAL_COLOR = (210, 210, 210)     # neutral silver/grey (BGR)
BOX_UNCERTAIN_COLOR = (0, 140, 255)    # orange (BGR)
DOT_OUTSIDE_COLOR = (128, 128, 128)    # grey when outside all zones (BGR)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render events_overlay.mp4 with zone outlines and event banners.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, type=Path,
                   help="Source video file.")
    p.add_argument("--tracks", required=True, type=Path,
                   help="tracks.csv from run_tracking.py.")
    p.add_argument("--zones", required=True, type=Path,
                   help="Zone config JSON (e.g. configs/zones_cam01.json).")
    p.add_argument("--events", required=True, type=Path,
                   help="events.csv from make_events.py.")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Directory to save events_overlay.mp4.")
    p.add_argument("--stride", type=int, default=1,
                   help="Process every Nth frame to match tracking stride.")
    p.add_argument("--banner-duration-s", type=float, default=BANNER_DURATION_S,
                   help="Duration in seconds for event banners to stay on screen.")
    return p


def _load_tracks(tracks_path: Path) -> dict[int, list[TrackRow]]:
    """Load tracks.csv grouped by frame_idx."""
    by_frame: dict[int, list[TrackRow]] = {}
    with open(tracks_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cls_raw = r.get("cls", 0)
            try:
                cls_int = int(cls_raw)
            except (ValueError, TypeError):
                cls_int = 0
            tr = TrackRow(
                frame_idx=int(r["frame_idx"]),
                time_s=float(r["time_s"]),
                track_id=int(r["track_id"]),
                cls=cls_int,
                conf=float(r["conf"]),
                x1=float(r["x1"]),
                y1=float(r["y1"]),
                x2=float(r["x2"]),
                y2=float(r["y2"]),
            )
            by_frame.setdefault(tr.frame_idx, []).append(tr)
    return by_frame


def _load_events(events_path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(events_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def _ffmpeg_encode(raw_avi: Path, out_mp4: Path) -> None:
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-i", str(raw_avi),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        str(out_mp4),
    ]
    logger.info("Re-encoding with ffmpeg …")
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning("ffmpeg stderr: %s", result.stderr.decode(errors="replace"))


def _find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return "ffmpeg"


def _draw_legend(canvas: np.ndarray, zones: list[Zone], zone_color_map: dict[str, tuple[int, int, int]]) -> None:
    pad = 14
    x0, y0 = 24, 24
    line_h = 32
    n_lines = 1 + len(zones) + 2  # header + zones + box line + dot line
    panel_w = 520
    panel_h = pad * 2 + n_lines * line_h

    # Semi-transparent dark background card
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 80, 80), 1, cv2.LINE_AA)

    cur_y = y0 + pad + 20
    # Header
    cv2.putText(canvas, "LEGEND", (x0 + pad, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cur_y += line_h

    # Zones with color swatches
    for z in zones:
        c = zone_color_map.get(z.name, (200, 200, 200))
        cv2.rectangle(canvas, (x0 + pad, cur_y - 16), (x0 + pad + 20, cur_y + 4), c, -1)
        cv2.rectangle(canvas, (x0 + pad, cur_y - 16), (x0 + pad + 20, cur_y + 4), (0, 0, 0), 1)
        z_type_suffix = f" ({z.zone_type})" if z.zone_type else ""
        cv2.putText(canvas, f"{z.name}{z_type_suffix}", (x0 + pad + 30, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        cur_y += line_h

    # Box color explanation
    cv2.putText(canvas, "Box: Grey = normal | Orange = uncertain (?)", (x0 + pad, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cur_y += line_h

    # Dot color explanation
    cv2.putText(canvas, "Dot: Zone colour = inside | Grey = outside", (x0 + pad, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Validate zones file (Rule 13: human-authored inputs are read-only)
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

    for p, name in [(args.video, "video"), (args.tracks, "tracks"), (args.events, "events")]:
        if not p.exists():
            logger.error("%s not found: %s", name, p)
            return 1

    # ── Load data ─────────────────────────────────────────────────────────────
    tracks_by_frame = _load_tracks(args.tracks)
    event_rows = _load_events(args.events)

    # Track uncertainty spans from events.csv
    uncertain_spans: dict[int, list[tuple[int, int]]] = {}
    for er in event_rows:
        is_unc = str(er.get("identity_uncertain", "")).strip().lower() in ("true", "1", "yes")
        tid_val = er.get("track_id")
        if is_unc and tid_val:
            try:
                tid = int(tid_val)
                sf = int(er.get("start_frame", 0))
                ef = int(er.get("end_frame", sf))
                uncertain_spans.setdefault(tid, []).append((sf, ef))
            except ValueError:
                pass

    def is_track_uncertain(track_id: int, frame: int) -> bool:
        for sf, ef in uncertain_spans.get(track_id, []):
            if sf <= frame <= ef:
                return True
        return False

    # Map events to firing frame (use start_frame) for banner display
    banner_events: dict[int, list[tuple[str, str, str]]] = {}
    for er in event_rows:
        sf = int(er["start_frame"])
        et = er["event_type"]
        zn = er.get("zone") or ""
        tid = str(er.get("track_id") or "")
        banner_events.setdefault(sf, []).append((et, zn, tid))

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Video: %dx%d @ %.1f fps, %d frames", w, h, fps, total_frames)

    # Load zones at video resolution
    zones_scaled = load_zones(args.zones, target_width=w, target_height=h)

    # Build zone color map
    zone_color_map: dict[str, tuple[int, int, int]] = {}
    for idx, z in enumerate(zones_scaled):
        if z.zone_type.lower() == "restricted":
            zone_color_map[z.name] = (40, 40, 230)  # Crimson red (BGR)
        else:
            zone_color_map[z.name] = ZONE_COLORS[idx % len(ZONE_COLORS)]

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "_events_raw.avi"
    out_mp4 = out_dir / "events_overlay.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (w, h))

    banner_timeout: list[tuple[float, str, tuple[int, int, int]]] = []  # (expires_at_t, text, color)

    current_detections: list[TrackRow] = []
    frame_idx = 0
    from cctv.zones import zone_of_point

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        if frame_idx in tracks_by_frame:
            current_detections = tracks_by_frame[frame_idx]

        canvas = draw_zones_on_frame(frame, zones_scaled, alpha=0.2, line_thickness=2)

        for tr in current_detections:
            unc = is_track_uncertain(tr.track_id, frame_idx)
            box_color = BOX_UNCERTAIN_COLOR if unc else BOX_NORMAL_COLOR
            x1, y1, x2, y2 = int(tr.x1), int(tr.y1), int(tr.x2), int(tr.y2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # Footpoint dot
            fp_x = int((x1 + x2) / 2)
            fp_y = int(y2)
            in_zones = zone_of_point(zones_scaled, float(fp_x), float(fp_y))

            if in_zones:
                dot_color = zone_color_map.get(in_zones[0], (0, 255, 255))
            else:
                dot_color = DOT_OUTSIDE_COLOR

            cv2.circle(canvas, (fp_x, fp_y), FOOTPOINT_RADIUS, dot_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (fp_x, fp_y), FOOTPOINT_RADIUS + 2, (0, 0, 0), 1, cv2.LINE_AA)

            # Label above box: #id + zone name(s) + (?) if uncertain
            label = f"#{tr.track_id}"
            if in_zones:
                label += " " + ",".join(in_zones)
            if unc:
                label += " (?)"

            cv2.putText(
                canvas, label, (x1, max(y1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA
            )

        # Legend panel in top-left corner
        _draw_legend(canvas, zones_scaled, zone_color_map)

        # Expire old banners
        banner_timeout = [(exp, txt, col) for (exp, txt, col) in banner_timeout if exp > t]

        # Add new banners
        if frame_idx in banner_events:
            for (et, zn, tid) in banner_events[frame_idx]:
                parts = [et]
                if zn:
                    parts.append(zn)
                if tid:
                    parts.append(f"#{tid}")
                banner_text = " ".join(parts)

                # Color-code banner by event category
                if "RESTRICTED" in et:
                    banner_color = (40, 40, 240)  # red/crimson alert
                elif "FIRST_SEEN_INSIDE" in et or "PRESENT_AT_START" in et:
                    banner_color = (255, 200, 50)  # yellow/gold status
                elif "ENTER" in et:
                    banner_color = (80, 230, 80)  # emerald green
                elif "EXIT" in et:
                    banner_color = (0, 140, 255)  # amber orange
                elif "VISIT" in et:
                    banner_color = (240, 180, 70)  # sky blue
                elif "APPEARED" in et:
                    banner_color = (210, 210, 210)  # light grey
                elif "LEFT_VIEW" in et:
                    banner_color = (150, 150, 150)  # dim grey
                else:
                    banner_color = (0, 255, 200)

                banner_timeout.append((t + args.banner_duration_s, banner_text, banner_color))

        # Draw banners (bottom-left, stacked upward)
        for bi, (_, text, color) in enumerate(banner_timeout[:7]):
            bx, by = 20, h - 80 - bi * 34
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(canvas, (bx - 2, by - th - 6), (bx + tw + 4, by + baseline), (15, 15, 15), -1)
            cv2.putText(canvas, text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # Time overlay (top right)
        ts_text = f"t={t:.2f}s  f={frame_idx}"
        cv2.putText(canvas, ts_text, (w - 240, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

        writer.write(canvas)
        frame_idx += 1

    writer.release()
    cap.release()
    logger.info("Wrote raw AVI: %s", raw_path)

    _ffmpeg_encode(raw_path, out_mp4)
    raw_path.unlink(missing_ok=True)
    logger.info("Saved events_overlay.mp4: %s", out_mp4)
    print(f"events_overlay.mp4 saved: {out_mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
