#!/usr/bin/env python
"""Compare multiple tracking runs from their tracks.csv and run_stats.json files.

Usage
-----
python scripts/compare_runs.py \\
    --base-dir data/output/cam01_compare \\
    --runs A B C D

For each run <name>:
  - reads  <base-dir>/<name>/tracks.csv
  - reads  <base-dir>/<name>/run_stats.json
  - computes proxy metrics and prints a table
  - saves  <base-dir>/summary.csv

IMPORTANT DISCLAIMER (printed at runtime and written to summary.csv header):
  These metrics are PROXIES only. Without ground-truth annotations we cannot
  compute MOTA, MOTP, IDF1 or the true number of ID switches.
  Judge ID-switch behaviour by watching each annotated.mp4 directly.

Rules observed
--------------
Rule 2: no fabrication — all numbers come from the CSV rows on disk.
Rule 4: type hints, pathlib, logging.
Rule 9: real results only; disclaimer stated clearly.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Make src/ importable ──────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_runs")

DISCLAIMER = (
    "DISCLAIMER: All metrics below are PROXIES computed from tracks.csv. "
    "Without ground-truth annotations, true MOTA/IDF1/ID-switch counts cannot "
    "be measured. Judge ID-switch behaviour by watching each annotated.mp4."
)

SHORT_TRACK_THRESHOLD_S = 1.0  # tracks shorter than this are counted as "short"


def analyse_run(run_dir: Path) -> dict:
    """Read tracks.csv + run_stats.json from *run_dir* and return a metrics dict."""
    csv_path = run_dir / "tracks.csv"
    stats_path = run_dir / "run_stats.json"

    if not csv_path.exists():
        logger.warning("tracks.csv not found in %s — skipping.", run_dir)
        return {}
    if not stats_path.exists():
        logger.warning("run_stats.json not found in %s — skipping.", run_dir)
        return {}

    # ── Load tracks ───────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning("tracks.csv in %s is empty — zero detections.", run_dir)
        return {
            "run": run_dir.name,
            "unique_track_ids": 0,
            "mean_track_len_s": 0.0,
            "median_track_len_s": 0.0,
            "short_tracks_lt1s": 0,
            "mean_dets_per_frame": 0.0,
            "wall_clock_s": 0.0,
            "tracker": "?",
            "conf_thresh": "?",
            "imgsz": "?",
        }

    # ── Per-track duration (time_s span) ─────────────────────────────────────
    track_spans = (
        df.groupby("track_id")["time_s"]
        .agg(lambda x: x.max() - x.min())
    )
    mean_len = float(track_spans.mean())
    median_len = float(track_spans.median())
    short_tracks = int((track_spans < SHORT_TRACK_THRESHOLD_S).sum())
    unique_ids = int(df["track_id"].nunique())

    # ── Mean detections per processed frame ──────────────────────────────────
    dets_per_frame = df.groupby("frame_idx").size()
    mean_dets = float(dets_per_frame.mean())

    # ── Load run_stats ────────────────────────────────────────────────────────
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    wall_s = stats.get("wall_clock_seconds", 0.0)
    tracker = stats.get("tracker", "?")
    conf = stats.get("conf_thresh", "?")
    imgsz = stats.get("imgsz", "?")
    processing_fps = stats.get("processing_fps", 0.0)

    return {
        "run": run_dir.name,
        "unique_track_ids": unique_ids,
        "mean_track_len_s": round(mean_len, 2),
        "median_track_len_s": round(median_len, 2),
        "short_tracks_lt1s": short_tracks,
        "mean_dets_per_frame": round(mean_dets, 2),
        "wall_clock_s": round(wall_s, 1),
        "processing_fps": round(processing_fps, 2),
        "tracker": tracker,
        "conf_thresh": conf,
        "imgsz": imgsz,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare multiple tracking runs and produce summary.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--base-dir", required=True, type=Path,
        help="Base directory containing one sub-folder per run.",
    )
    p.add_argument(
        "--runs", nargs="+", required=True,
        help="Names of run sub-folders (e.g. A B C D).",
    )
    p.add_argument(
        "--out-csv", type=Path, default=None,
        help="Output CSV path. Defaults to <base-dir>/summary.csv.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_dir: Path = args.base_dir
    out_csv: Path = args.out_csv or (base_dir / "summary.csv")

    print(f"\n{'='*70}")
    print(DISCLAIMER)
    print(f"{'='*70}\n")

    rows: list[dict] = []
    for name in args.runs:
        run_dir = base_dir / name
        if not run_dir.is_dir():
            logger.warning("Run directory not found: %s — skipping.", run_dir)
            continue
        logger.info("Analysing run: %s", name)
        row = analyse_run(run_dir)
        if row:
            rows.append(row)

    if not rows:
        logger.error("No valid runs found. Nothing to compare.")
        return 1

    # ── Print table ───────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(rows)

    col_order = [
        "run", "tracker", "conf_thresh", "imgsz",
        "unique_track_ids", "mean_track_len_s", "median_track_len_s",
        "short_tracks_lt1s", "mean_dets_per_frame",
        "wall_clock_s", "processing_fps",
    ]
    # Only include columns that exist (future-proofing)
    col_order = [c for c in col_order if c in summary_df.columns]
    summary_df = summary_df[col_order]

    print(summary_df.to_string(index=False))
    print()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Write disclaimer as first comment line
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# {DISCLAIMER}\n")
        f.write(f"# short_tracks threshold: < {SHORT_TRACK_THRESHOLD_S}s\n")
        summary_df.to_csv(f, index=False)

    logger.info("Summary saved to: %s", out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
