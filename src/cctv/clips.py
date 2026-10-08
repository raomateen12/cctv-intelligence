"""Evidence clip extraction and generation from video ranges."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def extract_clip(
    source_video_path: Path | str,
    output_clip_path: Path | str,
    start_frame: int,
    end_frame: int,
    padding_frames: int = 10,
) -> Path:
    """Extract a segment of video between start_frame and end_frame to output_clip_path."""
    src_path = Path(source_video_path)
    out_path = Path(output_clip_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.is_file():
        raise FileNotFoundError(f"Source video file not found: {src_path}")

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open source video: {src_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        actual_start = max(0, start_frame - padding_frames)
        actual_end = min(total_frames - 1, end_frame + padding_frames) if total_frames > 0 else end_frame + padding_frames

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)
        current_frame = actual_start

        while current_frame <= actual_end:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            current_frame += 1

        writer.release()
        logger.info("Saved clip from frame %d to %d to %s", actual_start, actual_end, out_path)
        return out_path
    finally:
        cap.release()
