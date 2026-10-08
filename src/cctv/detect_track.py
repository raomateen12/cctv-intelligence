"""Detection and tracking pipeline using YOLO + ByteTrack.

Rules observed:
- Rule 4: type hints, pathlib, logging, no magic numbers (all params configurable).
- Rule 5: device defaults to cpu; stride, imgsz, max_seconds configurable.
- Rule 2/3: frame_idx is always the ORIGINAL frame counter; time_s = frame_idx / fps.
- Class 0 = person (COCO). No other class is claimed.
- track_id None rows are skipped (tracker not yet assigned an ID).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Detection:
    """Bounding box detection (single frame, single box)."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackedObject:
    """Tracked object across frames (always carries original frame_idx)."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    frame_idx: int  # ORIGINAL frame index in the source video


@dataclass
class RunStats:
    """Statistics produced by a tracking run."""

    video_fps: float = 0.0
    total_frames: int = 0
    processed_frames: int = 0
    wall_clock_seconds: float = 0.0
    processing_fps: float = 0.0
    unique_track_ids: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "video_fps": self.video_fps,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "wall_clock_seconds": round(self.wall_clock_seconds, 2),
            "processing_fps": round(self.processing_fps, 2),
            "unique_track_ids": self.unique_track_ids,
        }
        d.update(self.extra)
        return d


# ── Frame-index / time helpers (pure, unit-testable) ─────────────────────────


def frame_idx_to_time(frame_idx: int, fps: float) -> float:
    """Convert an original frame index to wall-clock seconds.

    NEVER uses the processed-frame counter — always uses the original index.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return frame_idx / fps


def max_frames_from_seconds(max_seconds: float | None, fps: float) -> int | None:
    """Return the maximum original frame index limit from max_seconds, or None."""
    if max_seconds is None:
        return None
    return int(max_seconds * fps)


# ── Core tracker class ────────────────────────────────────────────────────────


class VideoTracker:
    """End-to-end YOLO + ByteTrack tracker for a single video.

    Parameters
    ----------
    model_name:
        YOLO model checkpoint (e.g. ``yolo11n.pt``).  Downloaded automatically
        by ultralytics on first use (< 10 MB; within Rule 6 limit).
    device:
        Inference device — ``"cpu"`` (default) or ``"cuda:0"``.
    conf_thresh:
        Minimum detector confidence to keep a detection.
    imgsz:
        Inference image size passed to YOLO.
    stride:
        Process every Nth frame.  Frame 0, stride, 2*stride, … are processed;
        time_s is always computed from the ORIGINAL frame index.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        device: str = "cpu",
        conf_thresh: float = 0.3,
        imgsz: int = 640,
        stride: int = 5,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.stride = stride
        self.tracker = tracker  # tracker yaml name or absolute path to custom yaml
        self._model: Any = None

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Lazy-load the YOLO model (idempotent)."""
        if self._model is None:
            from ultralytics import YOLO

            logger.info(
                "Loading YOLO model '%s' on device '%s'",
                self.model_name,
                self.device,
            )
            self._model = YOLO(self.model_name)

    # ── Single-frame inference ────────────────────────────────────────────────

    def track_frame(
        self,
        frame: np.ndarray,
        original_frame_idx: int,
    ) -> list[TrackedObject]:
        """Run tracking on one frame; return TrackedObject list (may be empty).

        Only class 0 (person) results are returned.
        Rows where track_id is None are skipped (tracker not yet initialised).
        """
        self.load_model()
        results = self._model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,  # bytetrack.yaml or custom yaml path
            classes=[0],           # person only; COCO model has no forklift class
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        tracked: list[TrackedObject] = []
        if not results:
            return tracked

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return tracked

        for box in boxes:
            # Skip boxes without a tracker ID (ByteTrack tentative track)
            if box.id is None:
                continue
            track_id = int(box.id[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = self._model.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

            tracked.append(
                TrackedObject(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    frame_idx=original_frame_idx,
                )
            )

        return tracked

    # ── Full-video run ────────────────────────────────────────────────────────

    def run(
        self,
        video_path: Path | str,
        out_dir: Path | str,
        max_seconds: float | None = None,
    ) -> RunStats:
        """Process *video_path*, write outputs to *out_dir*, return RunStats.

        Outputs
        -------
        tracks.csv
            One row per detection: frame_idx, time_s, track_id, cls, conf,
            x1, y1, x2, y2.
        annotated.mp4
            H.264 / yuv420p video playable in any standard player.
        run_stats.json
            Serialised RunStats dict.
        """
        import csv

        video_path = Path(video_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # ── Open source video ─────────────────────────────────────────────────
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        logger.info(
            "Video: %s | fps=%.2f | frames=%d | size=%dx%d",
            video_path.name, fps, total_frames, width, height,
        )

        # Effective FPS of the annotated output (stride-reduced)
        out_fps = max(fps / self.stride, 1.0)

        # Maximum original frame index to process
        max_frame_idx = max_frames_from_seconds(max_seconds, fps)

        # ── Set up raw OpenCV writer (will re-encode afterwards if needed) ────
        raw_out_path = out_dir / "_annotated_raw.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(raw_out_path), fourcc, out_fps, (width, height))

        # ── CSV writer ────────────────────────────────────────────────────────
        tracks_csv_path = out_dir / "tracks.csv"
        csv_file = tracks_csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["frame_idx", "time_s", "track_id", "cls", "conf",
             "x1", "y1", "x2", "y2"]
        )

        # ── Main loop ─────────────────────────────────────────────────────────
        all_track_ids: set[int] = set()
        processed_frames = 0
        original_frame_idx = 0
        t_start = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Honour max_seconds limit
                if max_frame_idx is not None and original_frame_idx >= max_frame_idx:
                    break

                if original_frame_idx % self.stride == 0:
                    # time_s uses ORIGINAL frame index, not the processed counter
                    time_s = frame_idx_to_time(original_frame_idx, fps)

                    tracked = self.track_frame(frame, original_frame_idx)
                    processed_frames += 1

                    # Write CSV rows
                    for obj in tracked:
                        all_track_ids.add(obj.track_id)
                        x1, y1, x2, y2 = obj.bbox_xyxy
                        csv_writer.writerow([
                            obj.frame_idx,
                            round(time_s, 4),
                            obj.track_id,
                            obj.class_name,
                            round(obj.confidence, 4),
                            round(x1, 1), round(y1, 1),
                            round(x2, 1), round(y2, 1),
                        ])

                    # Annotate frame
                    annotated = _draw_annotations(frame, tracked, time_s)
                    writer.write(annotated)

                original_frame_idx += 1

        finally:
            cap.release()
            writer.release()
            csv_file.close()

        wall_clock = time.perf_counter() - t_start
        proc_fps = processed_frames / wall_clock if wall_clock > 0 else 0.0

        stats = RunStats(
            video_fps=round(fps, 4),
            total_frames=total_frames,
            processed_frames=processed_frames,
            wall_clock_seconds=wall_clock,
            processing_fps=proc_fps,
            unique_track_ids=len(all_track_ids),
            extra={
                "tracker": self.tracker,
                "conf_thresh": self.conf_thresh,
                "imgsz": self.imgsz,
                "stride": self.stride,
                "model_name": self.model_name,
            },
        )

        # ── Re-encode to H.264 / yuv420p for broad player compatibility ───────
        final_mp4 = out_dir / "annotated.mp4"
        _reencode_h264(raw_out_path, final_mp4)
        try:
            raw_out_path.unlink()
        except OSError:
            pass

        # ── Write run_stats.json ──────────────────────────────────────────────
        stats_path = out_dir / "run_stats.json"
        stats_dict = stats.to_dict()
        stats_path.write_text(json.dumps(stats_dict, indent=2), encoding="utf-8")

        logger.info("Run complete. Stats: %s", stats_dict)
        return stats


# ── Drawing helpers ───────────────────────────────────────────────────────────


def _draw_annotations(
    frame: np.ndarray,
    tracked: list[TrackedObject],
    time_s: float,
) -> np.ndarray:
    """Draw bounding boxes, track IDs and timestamp on a copy of *frame*."""
    out = frame.copy()
    h = out.shape[0]

    for obj in tracked:
        x1, y1, x2, y2 = [int(v) for v in obj.bbox_xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"#{obj.track_id} {obj.class_name} {obj.confidence:.2f}"
        label_y = max(y1 - 8, 14)
        cv2.putText(
            out, label, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
        )

    # Timestamp overlay (bottom-left)
    ts_label = f"t={time_s:.2f}s"
    cv2.putText(
        out, ts_label, (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA,
    )
    return out


# ── ffmpeg re-encode (H.264 / yuv420p) ───────────────────────────────────────


def _reencode_h264(src: Path, dst: Path) -> None:
    """Re-encode *src* to *dst* as H.264 / yuv420p using the imageio-ffmpeg binary.

    This guarantees the output plays in any standard video player regardless of
    the OpenCV VideoWriter codec availability on the host machine.
    """
    import subprocess

    try:
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        logger.warning("imageio-ffmpeg not available (%s); skipping re-encode.", exc)
        src.rename(dst)
        return

    cmd = [
        ffmpeg_bin,
        "-y",            # overwrite
        "-i", str(src),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        "-an",           # no audio
        str(dst),
    ]
    logger.info("Re-encoding to H.264: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:  # pragma: no cover
        logger.error("ffmpeg re-encode failed:\n%s", result.stderr)
        raise RuntimeError(f"ffmpeg re-encode failed (exit {result.returncode})")
    logger.info("Saved annotated video: %s", dst)
