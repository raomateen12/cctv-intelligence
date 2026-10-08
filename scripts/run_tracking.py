#!/usr/bin/env python
"""CLI entry point: run YOLO + ByteTrack/BoT-SORT on a recorded video.

Usage
-----
python scripts/run_tracking.py \\
    --video data/raw/cam01.mp4 \\
    --out-dir data/output/cam01 \\
    --model yolo11n.pt \\
    --stride 10 \\
    --imgsz 1280 \\
    --device cpu \\
    --conf 0.15 \\
    --tracker configs/trackers/botsort_reid.yaml
    # or: --tracker configs/trackers/bytetrack_tuned.yaml
    # or: --tracker configs/trackers/botsort_reid.yaml

Rules observed
--------------
Rule 4 : type hints, pathlib, logging (not print for library code); argparse for
         CLI arguments; no magic numbers — everything comes from args/config.
Rule 5 : --device defaults to cpu; --stride, --imgsz, --max-seconds all
         configurable.
Rule 9 : script prints real run_stats.json contents to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Make sure src/ is importable when running the script directly ─────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cctv.detect_track import VideoTracker  # noqa: E402

# ── Logging (Rule 4: logging, not print, in library code; script may use both) ─
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_tracking")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run YOLO person detection + ByteTrack on a video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, type=Path,
                   help="Path to input video file.")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Directory for outputs (tracks.csv, annotated.mp4, run_stats.json).")
    p.add_argument("--model", default="yolo11n.pt",
                   help="YOLO model checkpoint name or path.")
    p.add_argument("--stride", type=int, default=5,
                   help="Process every Nth frame. time_s always uses original frame idx.")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Inference image size (pixels, square).")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Stop after this many seconds of source video (optional).")
    p.add_argument("--device", default="cpu",
                   help="Inference device: 'cpu' or 'cuda:0'.")
    p.add_argument("--conf", type=float, default=0.15,
                   help="Minimum detector confidence threshold.")
    p.add_argument("--tracker", default="configs/trackers/botsort_reid.yaml",
                   help="Tracker yaml: custom config (e.g. configs/trackers/botsort_reid.yaml, "
                        "configs/trackers/bytetrack_tuned.yaml) or built-in (bytetrack.yaml, botsort.yaml).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    video_path: Path = args.video
    out_dir: Path = args.out_dir

    # ── Validate input early (Rule 4: fail fast with clear message) ───────────
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        return 1

    logger.info("=== run_tracking start ===")
    logger.info("  video      : %s", video_path)
    logger.info("  out_dir    : %s", out_dir)
    logger.info("  model      : %s", args.model)
    logger.info("  stride     : %d", args.stride)
    logger.info("  imgsz      : %d", args.imgsz)
    logger.info("  max_seconds: %s", args.max_seconds)
    logger.info("  device     : %s", args.device)
    logger.info("  conf       : %.2f", args.conf)
    logger.info("  tracker    : %s", args.tracker)

    tracker = VideoTracker(
        model_name=args.model,
        device=args.device,
        conf_thresh=args.conf,
        imgsz=args.imgsz,
        stride=args.stride,
        tracker=args.tracker,
    )

    try:
        stats = tracker.run(
            video_path=video_path,
            out_dir=out_dir,
            max_seconds=args.max_seconds,
        )
    except FileNotFoundError as exc:
        logger.error("File error: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("Video error: %s", exc)
        return 1

    # ── Print real run_stats.json to stdout (Rule 9) ──────────────────────────
    stats_dict = stats.to_dict()
    print("\n=== run_stats.json ===")
    print(json.dumps(stats_dict, indent=2))

    # Warn if processing was slow (Rule 9: honest reporting)
    proc_fps = stats_dict.get("processing_fps", 0.0)
    if proc_fps < 1.0:
        logger.warning(
            "Processing was very slow: %.2f processed-frames/sec. "
            "Consider increasing --stride or reducing --imgsz.",
            proc_fps,
        )
    elif proc_fps < 3.0:
        logger.warning(
            "Processing was slow: %.2f processed-frames/sec. "
            "Increase --stride or --imgsz=320 to speed up.",
            proc_fps,
        )

    out_mp4 = out_dir / "annotated.mp4"
    if out_mp4.exists():
        logger.info("Annotated video saved to: %s", out_mp4)
    else:
        logger.warning("annotated.mp4 was NOT created — check for ffmpeg errors above.")

    logger.info("=== run_tracking done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
