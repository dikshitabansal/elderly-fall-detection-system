"""Unit and integration tests for Video Analysis module.

Tests cover:
- Timestamp formatting and frame-to-time calculations
- Validation of non-existent, unsupported, empty, and corrupt video files
- Metadata extraction on valid video containers
- Temporal state resets between analysis runs
- Full-pipeline analysis on synthetic video
- Early cancellation via stop_check
"""
import os
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

from src.detector import FallState
from src.video_analysis import (
    VideoAnalyzer,
    VideoEventRecord,
    VideoMetadata,
    format_timestamp,
    validate_video_file,
)


class TestVideoAnalysis(unittest.TestCase):
    """Test suite for video file validation, metadata extraction, and analysis pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_video(
        self,
        filename: str = "test_clip.mp4",
        num_frames: int = 20,
        width: int = 320,
        height: int = 240,
        fps: float = 10.0
    ) -> Path:
        """Helper to create a synthetic video with moving rectangle."""
        file_path = self.temp_path / filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))

        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Draw moving white rectangle
            x = int((i / num_frames) * (width - 60))
            cv2.rectangle(frame, (x, 50), (x + 50, 150), (255, 255, 255), -1)
            writer.write(frame)

        writer.release()
        return file_path

    # =========================================================================
    # 1. Timestamp Formatting Tests
    # =========================================================================
    def test_format_timestamp_zero_and_sub_minute(self):
        """Verifies MM:SS.ss formatting for short durations."""
        self.assertEqual(format_timestamp(0.0), "00:00.00")
        self.assertEqual(format_timestamp(4.25), "00:04.25")
        self.assertEqual(format_timestamp(59.99), "00:59.99")

    def test_format_timestamp_minutes_and_hours(self):
        """Verifies MM:SS.ss and HH:MM:SS.ss transitions."""
        self.assertEqual(format_timestamp(65.5), "01:05.50")
        self.assertEqual(format_timestamp(125.04), "02:05.04")
        self.assertEqual(format_timestamp(3665.25), "01:01:05.25")

    def test_format_timestamp_negative(self):
        """Negative timestamps should clamp to zero."""
        self.assertEqual(format_timestamp(-5.0), "00:00.00")

    # =========================================================================
    # 2. File Validation Tests
    # =========================================================================
    def test_validate_nonexistent_file(self):
        """Non-existent file reports descriptive error."""
        bogus = self.temp_path / "does_not_exist.mp4"
        is_valid, err, meta = validate_video_file(bogus)
        self.assertFalse(is_valid)
        self.assertIn("does not exist", err.lower())
        self.assertIsNone(meta)

    def test_validate_unsupported_extension(self):
        """Unsupported file extensions report descriptive error."""
        bad_ext = self.temp_path / "document.pdf"
        bad_ext.write_text("dummy content")
        is_valid, err, meta = validate_video_file(bad_ext)
        self.assertFalse(is_valid)
        self.assertIn("unsupported video format", err.lower())
        self.assertIsNone(meta)

    def test_validate_empty_file(self):
        """Empty 0-byte file reports error."""
        empty_vid = self.temp_path / "empty.mp4"
        empty_vid.touch()
        is_valid, err, meta = validate_video_file(empty_vid)
        self.assertFalse(is_valid)
        self.assertIn("empty", err.lower())
        self.assertIsNone(meta)

    def test_validate_corrupt_file(self):
        """Corrupted file header reports decoding failure."""
        corrupt = self.temp_path / "corrupt.mp4"
        corrupt.write_bytes(b"\x00\x01\x02\x03JUNKBYTES" * 50)
        is_valid, err, meta = validate_video_file(corrupt)
        self.assertFalse(is_valid)
        self.assertTrue("failed" in err.lower() or "unable" in err.lower())
        self.assertIsNone(meta)

    def test_validate_valid_synthetic_video(self):
        """Valid video yields accurate metadata (resolution, duration, fps)."""
        video_path = self._create_synthetic_video("sample.mp4", num_frames=30, width=320, height=240, fps=15.0)
        is_valid, err, meta = validate_video_file(video_path)

        self.assertTrue(is_valid)
        self.assertIsNone(err)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.width, 320)
        self.assertEqual(meta.height, 240)
        self.assertAlmostEqual(meta.fps, 15.0, places=1)
        self.assertEqual(meta.total_frames, 30)
        self.assertAlmostEqual(meta.duration_sec, 2.0, places=1)
        self.assertEqual(meta.duration_formatted, "00:02.00")

    # =========================================================================
    # 3. Temporal State Reset Tests
    # =========================================================================
    def test_temporal_state_reset(self):
        """Calling reset_state() cleanly resets detector, motion, and features."""
        analyzer = VideoAnalyzer()

        # Simulate dirty state
        frame = np.ones((240, 320, 3), dtype=np.uint8) * 128
        analyzer.motion_analyzer.process_frame(frame)
        self.assertIsNotNone(analyzer.motion_analyzer._previous_gray)

        # Trigger detector update
        from src.features import PoseFeatures
        from src.motion import MotionResult
        dirty_feat = PoseFeatures(torso_angle_deg=80.0, camera_view_type="FULL_BODY")
        dirty_motion = MotionResult(has_motion=True, motion_score=0.1)
        analyzer.fall_detector.update(dirty_feat, dirty_motion, timestamp=1.0)
        self.assertGreaterEqual(len(analyzer.fall_detector.event_log), 0)

        # Reset
        analyzer.reset_state()

        # Verify pristine baseline
        self.assertIsNone(analyzer.motion_analyzer._previous_gray)
        self.assertEqual(len(analyzer.feature_extractor.history), 0)
        self.assertEqual(analyzer.fall_detector.current_state, FallState.NORMAL)
        self.assertEqual(len(analyzer.fall_detector.event_log), 0)

    # =========================================================================
    # 4. Pipeline Execution & Progress Callback Tests
    # =========================================================================
    def test_analyze_video_pipeline(self):
        """End-to-end video analysis processes all frames and calls progress callback."""
        video_path = self._create_synthetic_video("moving.mp4", num_frames=12, width=320, height=240, fps=10.0)
        analyzer = VideoAnalyzer(inference_resolution=(320, 240))

        progress_calls = []

        def on_progress(ratio, curr, total, annotated, status, meta):
            progress_calls.append((ratio, curr, total))

        result = analyzer.analyze_video(video_path, progress_callback=on_progress)

        self.assertEqual(result.total_frames_processed, 12)
        self.assertGreater(len(progress_calls), 0)
        # Final progress should reach 1.0
        self.assertAlmostEqual(progress_calls[-1][0], 1.0)
        self.assertEqual(progress_calls[-1][1], 12)
        self.assertEqual(progress_calls[-1][2], 12)

        # Telemetry verification
        self.assertGreater(result.average_processing_fps, 0.0)
        self.assertEqual(result.metadata.total_frames, 12)
        self.assertIsNotNone(result.last_annotated_frame)

    def test_analyze_video_stop_cancellation(self):
        """stop_check returning True terminates processing early and cleanly."""
        video_path = self._create_synthetic_video("long.mp4", num_frames=25, width=320, height=240, fps=10.0)
        analyzer = VideoAnalyzer(inference_resolution=(320, 240))

        frame_counter = [0]

        def stop_after_5_frames():
            return frame_counter[0] >= 5

        def on_progress(ratio, curr, total, annotated, status, meta):
            frame_counter[0] = curr

        result = analyzer.analyze_video(
            video_path,
            progress_callback=on_progress,
            stop_check=stop_after_5_frames
        )

        # Should stop after ~5-6 frames, well before 25
        self.assertLess(result.total_frames_processed, 25)
        self.assertGreaterEqual(result.total_frames_processed, 5)


if __name__ == "__main__":
    unittest.main()
