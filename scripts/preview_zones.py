#!/usr/bin/env python
"""Draw zones on a video frame and save a preview image.

Usage
-----
python scripts/preview_zones.py \
    --video data/raw/cam01.mp4 \
    --zones configs/zones_cam01.json \
    --out zones_preview.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

# ── Ensure src/ is importable ─────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cctv.zones import draw_zones_on_frame, load_zones  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preview_zones")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render polygon zones onto a video frame and save preview image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, type=Path,
                   help="Path to source video file (e.g. data/raw/cam01.mp4).")
    p.add_argument("--zones", type=Path, default=None,
                   help="Path to zone configuration JSON (e.g. configs/zones_cam01.json).")
    p.add_argument("--camera-id", default=None,
                   help="Camera identifier (defaults to video stem if --zones not specified).")
    p.add_argument("--frame-idx", type=int, default=0,
                   help="Frame index to extract from the video.")
    p.add_argument("--out", type=Path, default=Path("zones_preview.png"),
                   help="Path to save output preview image.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_path: Path = args.video
    out_path: Path = args.out

    if not video_path.exists():
        logger.error("Video file does not exist: %s", video_path)
        return 1

    camera_id: str = args.camera_id or video_path.stem
    zones_path: Path = args.zones or (_REPO_ROOT / "configs" / f"zones_{camera_id}.json")

    if not zones_path.exists():
        logger.error("Zone config file does not exist: %s", zones_path)
        return 1

    # Extract requested frame from video
    cap = cv2.VideoCapture(str(video_path))
    if args.frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        logger.error("Could not read frame %d from: %s", args.frame_idx, video_path)
        return 1

    h, w = frame.shape[:2]
    logger.info("Video frame %d resolution: %dx%d (HxW)", args.frame_idx, h, w)

    # Load zones and scale if video resolution differs from JSON reference resolution
    try:
        zones = load_zones(zones_path, target_width=w, target_height=h)
    except Exception as exc:
        logger.error("Failed to load zones from %s: %s", zones_path, exc)
        return 1

    logger.info("Loaded %d zones from: %s", len(zones), zones_path)

    # Render zones onto frame
    # Use thicker lines on high-res frames so they are clear after downscaling
    line_thickness = max(2, int(round(w / 640.0)))
    annotated = draw_zones_on_frame(frame, zones, alpha=0.3, line_thickness=line_thickness)

    # Downscale to at most 1280 px wide for viewing
    max_w = 1280
    if w > max_w:
        scale = max_w / float(w)
        new_w = max_w
        new_h = int(round(h * scale))
        preview = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(
            "Downscaled preview from %dx%d to %dx%d (max %dpx wide)",
            w,
            h,
            new_w,
            new_h,
            max_w,
        )
    else:
        preview = annotated

    out_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(out_path), preview)
    if not success:
        logger.error("Failed to write preview image to: %s", out_path)
        return 1

    logger.info("Successfully saved zones preview to: %s", out_path)
    print(f"Preview image saved: {out_path} ({preview.shape[1]}x{preview.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
