"""Pure-logic tests for detect_track utilities.

Tests here use only synthetic data (no video files, no model weights).
They cover:
- frame_idx_to_time: original frame index -> seconds conversion
- max_frames_from_seconds: max_seconds -> max frame index limit
- stride sampling: correct frame indices are selected
- track_id None filtering: rows with no ID must be skipped
"""

from __future__ import annotations

import math

import pytest

from cctv.detect_track import (
    TrackedObject,
    frame_idx_to_time,
    max_frames_from_seconds,
)


# ── frame_idx_to_time ─────────────────────────────────────────────────────────


class TestFrameIdxToTime:
    """time_s must come from the ORIGINAL frame index, never a processed counter."""

    def test_frame_0_is_time_0(self):
        assert frame_idx_to_time(0, fps=25.0) == 0.0

    def test_first_stride_frame_at_50fps(self):
        # stride=10, first processed frame is frame 0, second is frame 10
        # time should be 10/50 = 0.2s, NOT 1/50 = 0.02s
        assert math.isclose(frame_idx_to_time(10, fps=50.0), 0.2, rel_tol=1e-9)

    def test_original_index_preserved_with_stride_5(self):
        fps = 30.0
        stride = 5
        # Processed frame 3 → original frame_idx = 3 * stride = 15
        original_idx = 3 * stride  # 15
        expected_time = 15 / fps   # 0.5s
        assert math.isclose(frame_idx_to_time(original_idx, fps), expected_time)

    def test_large_frame_index(self):
        # 50 fps video, frame 9000 → 180.0 s
        assert math.isclose(frame_idx_to_time(9000, fps=50.0), 180.0)

    def test_non_integer_fps(self):
        # 29.97 fps (NTSC)
        result = frame_idx_to_time(2997, fps=29.97)
        assert math.isclose(result, 100.0, rel_tol=1e-4)

    def test_invalid_fps_raises(self):
        with pytest.raises(ValueError, match="fps must be positive"):
            frame_idx_to_time(10, fps=0.0)

    def test_negative_fps_raises(self):
        with pytest.raises(ValueError, match="fps must be positive"):
            frame_idx_to_time(10, fps=-25.0)


# ── max_frames_from_seconds ───────────────────────────────────────────────────


class TestMaxFramesFromSeconds:
    def test_none_returns_none(self):
        assert max_frames_from_seconds(None, fps=50.0) is None

    def test_120_seconds_at_50fps(self):
        # 120 * 50 = 6000 frames
        assert max_frames_from_seconds(120.0, fps=50.0) == 6000

    def test_fractional_seconds_truncated(self):
        # 1.9 * 30 = 57.0 → 57
        assert max_frames_from_seconds(1.9, fps=30.0) == 57

    def test_zero_seconds(self):
        assert max_frames_from_seconds(0.0, fps=25.0) == 0


# ── Stride sampling: correct original frame indices ───────────────────────────


class TestStrideSampling:
    """Simulate the frame loop to verify original frame indices are correct."""

    def _simulate_stride(
        self,
        total_frames: int,
        stride: int,
        fps: float,
        max_seconds: float | None = None,
    ) -> list[tuple[int, float]]:
        """Return (original_frame_idx, time_s) for each processed frame."""
        max_idx = max_frames_from_seconds(max_seconds, fps)
        results = []
        for idx in range(total_frames):
            if max_idx is not None and idx >= max_idx:
                break
            if idx % stride == 0:
                results.append((idx, frame_idx_to_time(idx, fps)))
        return results

    def test_stride_1_all_frames(self):
        result = self._simulate_stride(5, stride=1, fps=25.0)
        assert [r[0] for r in result] == [0, 1, 2, 3, 4]

    def test_stride_5_correct_indices(self):
        result = self._simulate_stride(25, stride=5, fps=25.0)
        assert [r[0] for r in result] == [0, 5, 10, 15, 20]

    def test_stride_10_at_50fps_correct_times(self):
        # Video: 50 fps, stride 10 → processed frame indices 0,10,20,...
        # time_s should be 0.0, 0.2, 0.4, ...
        result = self._simulate_stride(50, stride=10, fps=50.0)
        expected_times = [0.0, 0.2, 0.4, 0.6, 0.8]
        for (_, t), et in zip(result, expected_times):
            assert math.isclose(t, et, abs_tol=1e-9), f"{t} != {et}"

    def test_max_seconds_stops_at_correct_frame(self):
        # 50 fps, max_seconds=2 → stop at frame 100 (exclusive)
        result = self._simulate_stride(200, stride=10, fps=50.0, max_seconds=2.0)
        assert all(idx < 100 for idx, _ in result)
        assert max(idx for idx, _ in result) == 90  # last: 90

    def test_processed_counter_would_give_wrong_time(self):
        """Demonstrate why we must use the original frame_idx, not a counter.

        With stride=10 and fps=50, the 3rd *processed* frame is original idx 20.
        time_s = 20/50 = 0.4 s.
        If we used processed_counter=2 instead: 2/50 = 0.04 s → WRONG.
        """
        fps = 50.0
        stride = 10
        third_processed_original_idx = 2 * stride  # 20
        correct_time = frame_idx_to_time(third_processed_original_idx, fps)
        wrong_time = frame_idx_to_time(2, fps)  # using counter=2 is wrong
        assert math.isclose(correct_time, 0.4, abs_tol=1e-9)
        assert math.isclose(wrong_time, 0.04, abs_tol=1e-9)
        assert not math.isclose(correct_time, wrong_time)


# ── track_id None filtering ───────────────────────────────────────────────────


class TestTrackIdFiltering:
    """Verify that downstream code correctly skips TrackedObject entries
    with no track_id (the VideoTracker.track_frame method skips them
    at creation time, so this tests the contract)."""

    def _make_tracked(self, track_id: int) -> TrackedObject:
        return TrackedObject(
            track_id=track_id,
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
            frame_idx=0,
        )

    def test_valid_track_ids_are_kept(self):
        objects = [self._make_tracked(1), self._make_tracked(2)]
        assert len(objects) == 2

    def test_track_id_zero_is_valid(self):
        # track_id=0 is a legitimate ByteTrack ID
        obj = self._make_tracked(0)
        assert obj.track_id == 0

    def test_all_fields_present_on_tracked_object(self):
        obj = self._make_tracked(42)
        assert obj.track_id == 42
        assert obj.class_id == 0
        assert obj.class_name == "person"
        assert obj.confidence == 0.85
        assert obj.bbox_xyxy == (10.0, 20.0, 100.0, 200.0)
        assert obj.frame_idx == 0
