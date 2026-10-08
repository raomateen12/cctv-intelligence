"""Video input and output utilities using pathlib, OpenCV, and imageio."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata describing a video source."""

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


def get_video_metadata(video_path: Path | str) -> VideoMetadata:
    """Retrieve metadata from a video file."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0

        return VideoMetadata(
            path=path,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_seconds=duration,
        )
    finally:
        cap.release()


def read_frames(
    video_path: Path | str,
    stride: int = 1,
    max_frames: int | None = None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield frame index and frame array from video according to stride."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")

    frame_idx = 0
    yielded_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                yield frame_idx, frame
                yielded_count += 1
                if max_frames is not None and yielded_count >= max_frames:
                    break

            frame_idx += 1
    finally:
        cap.release()
