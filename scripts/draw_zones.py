#!/usr/bin/env python
"""Interactive GUI tool for defining spatial zones on video frames.

Opens the first frame of a video in an OpenCV window, allowing interactive
polygon drawing with mouse clicks. Downscales for display if larger than
1280x720, but always converts and saves points in ORIGINAL video resolution.

Controls
--------
  [Left Click] : Add a polygon point
  'n'          : Finish current polygon, prompt for zone name & type in terminal
  'u'          : Undo last point in the current polygon
  's'          : Save all zones to configs/zones_<camera_id>.json
  'q' / ESC    : Quit

Usage
-----
python scripts/draw_zones.py --video data/raw/cam01.mp4 --camera-id cam01
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── Ensure src/ is importable ─────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cctv.zones import (  # noqa: E402
    Zone,
    draw_zones_on_frame,
    load_zone_config,
    save_zones,
    scale_zones,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("draw_zones")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive tool to draw polygon zones on a video frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, type=Path,
                   help="Path to source video file (e.g. data/raw/cam01.mp4).")
    p.add_argument("--camera-id", default=None,
                   help="Camera ID (defaults to video filename stem, e.g. cam01).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path for zones JSON (defaults to configs/zones_<camera_id>.json).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_path: Path = args.video

    if not video_path.exists():
        logger.error("Video file does not exist: %s", video_path)
        return 1

    camera_id: str = args.camera_id or video_path.stem
    out_path: Path = args.out or (_REPO_ROOT / "configs" / f"zones_{camera_id}.json")

    # Read first frame of video
    cap = cv2.VideoCapture(str(video_path))
    ret, orig_frame = cap.read()
    cap.release()

    if not ret or orig_frame is None:
        logger.error("Failed to read first frame from: %s", video_path)
        return 1

    orig_h, orig_w = orig_frame.shape[:2]
    logger.info("Video resolution: %dx%d (HxW)", orig_h, orig_w)

    # Compute display scaling if frame exceeds 1280x720 (Rule 5 & requirements)
    max_disp_w, max_disp_h = 1280, 720
    scale = 1.0
    if orig_w > max_disp_w or orig_h > max_disp_h:
        scale = min(max_disp_w / orig_w, max_disp_h / orig_h)

    disp_w = int(round(orig_w * scale))
    disp_h = int(round(orig_h * scale))
    if scale < 1.0:
        logger.info(
            "Displaying downscaled frame at %dx%d (scale factor: %.4f). Coordinates will be mapped to original %dx%d.",
            disp_w,
            disp_h,
            scale,
            orig_w,
            orig_h,
        )
        disp_base = cv2.resize(orig_frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        disp_base = orig_frame.copy()

    # Load existing zones if present
    completed_zones: list[Zone] = []
    if out_path.exists():
        try:
            cfg = load_zone_config(out_path)
            completed_zones = list(cfg.zones)
            logger.info("Loaded %d existing zones from %s", len(completed_zones), out_path)
        except Exception as exc:
            logger.warning("Could not load existing zones from %s: %s", out_path, exc)

    current_polygon_orig: list[tuple[float, float]] = []
    state = {"needs_redraw": True}

    def to_orig(x_disp: int, y_disp: int) -> tuple[float, float]:
        """Convert display coordinates to original video pixel coordinates."""
        ox = float(np.clip(round(x_disp / scale), 0, orig_w - 1))
        oy = float(np.clip(round(y_disp / scale), 0, orig_h - 1))
        return (ox, oy)

    def to_disp(x_orig: float, y_orig: float) -> tuple[int, int]:
        """Convert original video pixel coordinates to display coordinates."""
        dx = int(round(x_orig * scale))
        dy = int(round(y_orig * scale))
        return (dx, dy)

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = to_orig(x, y)
            current_polygon_orig.append(pt)
            print(f"  Point #{len(current_polygon_orig)} added: original ({pt[0]:.1f}, {pt[1]:.1f})")
            state["needs_redraw"] = True

    win_name = f"Draw Zones - {camera_id} (Original: {orig_w}x{orig_h})"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, on_mouse)

    print("\n" + "=" * 70)
    print(f"Draw Zones Tool: Camera '{camera_id}' ({orig_w}x{orig_h})")
    print("=" * 70)
    print("Controls:")
    print("  [Left Click] : Add polygon vertex")
    print("  'n'          : Finish current polygon, set name and type in terminal")
    print("  'u'          : Undo last point in current polygon")
    print("  's'          : Save all zones to JSON")
    print("  'q' / ESC    : Quit")
    print("=" * 70 + "\n")

    while True:
        if state["needs_redraw"]:
            canvas = disp_base.copy()

            # Render completed zones scaled to display frame
            if completed_zones:
                disp_zones = scale_zones(completed_zones, orig_w, orig_h, disp_w, disp_h)
                canvas = draw_zones_on_frame(canvas, disp_zones, alpha=0.3, line_thickness=2)

            # Render currently in-progress polygon points and lines
            if current_polygon_orig:
                disp_pts = [to_disp(px, py) for px, py in current_polygon_orig]
                for pt in disp_pts:
                    cv2.circle(canvas, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, 6, (0, 0, 0), 1, cv2.LINE_AA)

                if len(disp_pts) > 1:
                    cv2.polylines(
                        canvas,
                        [np.array(disp_pts, dtype=np.int32)],
                        isClosed=False,
                        color=(0, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

            # Top status bar
            bar_h = 30
            cv2.rectangle(canvas, (0, 0), (disp_w, bar_h), (25, 25, 25), -1)
            status_text = (
                f"[Click]: Add pt | [n]: Finish | [u]: Undo | [s]: Save | [q]: Quit | "
                f"Zones: {len(completed_zones)} | In-prog: {len(current_polygon_orig)} pts"
            )
            cv2.putText(
                canvas,
                status_text,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(win_name, canvas)
            state["needs_redraw"] = False

        key = cv2.waitKey(30) & 0xFF

        # 'q' or ESC -> Quit
        if key in (ord("q"), ord("Q"), 27):
            print("\nExiting zone drawer.")
            break

        # 'u' -> Undo point
        elif key in (ord("u"), ord("U")):
            if current_polygon_orig:
                undone = current_polygon_orig.pop()
                print(f"Undid point: ({undone[0]:.1f}, {undone[1]:.1f})")
                state["needs_redraw"] = True
            else:
                print("No points in current polygon to undo.")

        # 'n' -> Finish polygon
        elif key in (ord("n"), ord("N")):
            if len(current_polygon_orig) < 3:
                print("\n[Warning] A polygon must have at least 3 points to complete. Add more points.")
                continue

            print(f"\n--- Finish Zone #{len(completed_zones) + 1} ({len(current_polygon_orig)} vertices) ---")
            try:
                name_input = input("Enter zone name (e.g. Zone A, doorway, conveyor): ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            zone_name = name_input if name_input else f"Zone_{len(completed_zones) + 1}"

            try:
                type_input = (
                    input("Enter zone type ('normal' or 'restricted') [default: normal]: ")
                    .strip()
                    .lower()
                )
            except (EOFError, KeyboardInterrupt):
                break

            zone_type = "restricted" if type_input in ("restricted", "r") else "normal"

            new_zone = Zone(
                name=zone_name,
                polygon=tuple(current_polygon_orig),
                zone_type=zone_type,
            )
            completed_zones.append(new_zone)
            current_polygon_orig.clear()
            print(f"Added zone '{zone_name}' ({zone_type}) with {len(new_zone.polygon)} vertices.\n")
            state["needs_redraw"] = True

        # 's' -> Save zones
        elif key in (ord("s"), ord("S")):
            save_zones(
                camera_id=camera_id,
                frame_width=orig_w,
                frame_height=orig_h,
                zones=completed_zones,
                out_path=out_path,
            )
            print(f"\n[Saved] Successfully saved {len(completed_zones)} zones to {out_path}\n")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
