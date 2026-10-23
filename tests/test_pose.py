"""Tests for MediaPipe pose estimation module, topology, and occlusion diagnostics."""
import math
import os
import unittest
import numpy as np
from src.pose import (
    PoseEstimator,
    LandmarkPoint,
    PoseQuality,
    ProcessResult,
    POSE_CONNECTIONS,
    KEY_TORSO_LANDMARKS,
    KEY_LOWER_BODY_LANDMARKS
)
from src.utils import ensure_model_exists


class TestPoseEstimator(unittest.TestCase):
    """Test suite for PoseEstimator class, occlusion handling, and quality checks."""

    @classmethod
    def setUpClass(cls):
        # Ensure model is cached
        cls.model_path = ensure_model_exists()

    def test_model_file_exists(self):
        """Verify model file was downloaded and is valid size."""
        self.assertTrue(os.path.exists(self.model_path))
        self.assertGreater(os.path.getsize(self.model_path), 1_000_000)

    def test_pose_estimator_init_and_close(self):
        """Verify initialization and clean closure of detector."""
        estimator = PoseEstimator(model_path=self.model_path)
        self.assertIsNotNone(estimator.detector)
        estimator.close()
        self.assertIsNone(estimator.detector)

    def test_mediapipe_pose_topology(self):
        """Verify that POSE_CONNECTIONS uses valid 33-landmark MediaPipe topology."""
        self.assertEqual(len(POSE_CONNECTIONS), 35)
        for start, end in POSE_CONNECTIONS:
            self.assertTrue(0 <= start < 33, f"Invalid start index {start}")
            self.assertTrue(0 <= end < 33, f"Invalid end index {end}")
            self.assertNotEqual(start, end)

        # Confirm key torso connections exist
        self.assertIn((11, 12), POSE_CONNECTIONS)  # Left to Right shoulder
        self.assertIn((11, 23), POSE_CONNECTIONS)  # Left shoulder to left hip
        self.assertIn((12, 24), POSE_CONNECTIONS)  # Right shoulder to right hip
        self.assertIn((23, 24), POSE_CONNECTIONS)  # Left to Right hip

    def test_process_blank_frame(self):
        """Processing a blank image should return no detected poses without crashing."""
        with PoseEstimator(model_path=self.model_path) as estimator:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            ann_frame, result, detected = estimator.process_frame(blank, draw_skeleton=True)

            self.assertFalse(detected)
            self.assertEqual(ann_frame.shape, (480, 640, 3))
            self.assertIsNotNone(result)

            landmarks = estimator.extract_landmarks(result, (480, 640))
            self.assertEqual(len(landmarks), 0)

            # Check quality object on blank frame
            quality = estimator.last_pose_quality
            self.assertFalse(quality.is_detected)
            self.assertEqual(quality.reliability_status, "NO_PERSON")

    def test_process_invalid_frame(self):
        """Processing an empty frame returns gracefully."""
        with PoseEstimator(model_path=self.model_path) as estimator:
            empty = np.array([], dtype=np.uint8)
            ann_frame, result, detected = estimator.process_frame(empty)
            self.assertFalse(detected)
            self.assertIsNone(result)

    def test_process_frame_with_inference_resolution(self):
        """Verify processing with downscaled inference resolution."""
        with PoseEstimator(model_path=self.model_path, inference_resolution=(320, 240)) as estimator:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            ann_frame, result, detected = estimator.process_frame(blank, draw_skeleton=True)
            self.assertFalse(detected)
            self.assertEqual(ann_frame.shape, (480, 640, 3))

    def test_process_result_backward_compatibility(self):
        """Verify ProcessResult unpacks as a 3-tuple while exposing quality attributes."""
        quality = PoseQuality(is_detected=True, is_reliable=True, reliability_status="RELIABLE")
        res = ProcessResult(np.zeros((10, 10, 3)), None, True, quality)

        # 3-tuple unpacking
        ann, det_res, detected = res
        self.assertTrue(detected)
        self.assertIsNone(det_res)
        self.assertEqual(ann.shape, (10, 10, 3))

        # Direct attribute access
        self.assertEqual(res.pose_quality.reliability_status, "RELIABLE")

    def test_missing_landmarks(self):
        """Verify quality assessment flags missing or incomplete landmarks."""
        estimator = PoseEstimator(model_path=self.model_path)
        # Empty list
        q_empty = estimator.evaluate_pose_quality([])
        self.assertFalse(q_empty.is_detected)
        self.assertFalse(q_empty.is_reliable)
        self.assertEqual(q_empty.reliability_status, "UNRELIABLE")

        # Incomplete landmark list (e.g. only 10 points)
        partial_points = [
            LandmarkPoint(i, f"LM_{i}", 0.5, 0.5, 0.0, 320, 240, 0.9, 0.9)
            for i in range(10)
        ]
        q_partial = estimator.evaluate_pose_quality(partial_points)
        self.assertFalse(q_partial.is_reliable)
        self.assertEqual(q_partial.reliability_status, "UNRELIABLE")
        estimator.close()

    def test_low_confidence_landmarks(self):
        """Verify that occluded key landmarks trigger UNRELIABLE status and filtering."""
        estimator = PoseEstimator(model_path=self.model_path)
        # Build 33 landmarks where hips are occluded (visibility = 0.1)
        points = []
        for i in range(33):
            is_hip = i in (23, 24)
            vis = 0.1 if is_hip else 0.95
            pres = 0.1 if is_hip else 0.95
            points.append(
                LandmarkPoint(
                    landmark_id=i,
                    name=f"LM_{i}",
                    x_norm=0.5,
                    y_norm=0.5,
                    z_norm=0.0,
                    x_px=320,
                    y_px=240,
                    visibility=vis,
                    presence=pres,
                    is_visible=(vis >= 0.5),
                    is_present=(pres >= 0.5),
                    is_in_bounds=True,
                    is_reliable=(vis >= 0.5 and pres >= 0.5)
                )
            )

        # Quality evaluation
        quality = estimator.evaluate_pose_quality(points)
        self.assertFalse(quality.is_reliable)
        self.assertEqual(quality.reliability_status, "UNRELIABLE")
        self.assertIsNotNone(quality.warning_message)
        self.assertIn("torso", quality.warning_message.lower())

        # Filtering test: get_reliable_landmarks should exclude occluded hips 23 and 24
        reliable_map = estimator.get_reliable_landmarks(points, min_visibility=0.5)
        self.assertNotIn(23, reliable_map)
        self.assertNotIn(24, reliable_map)
        self.assertIn(11, reliable_map)  # Shoulders are visible
        self.assertEqual(len(reliable_map), 31)

        estimator.close()

    def test_boundary_violations(self):
        """Verify boundary clipping is accurately flagged when landmarks cross edges."""
        estimator = PoseEstimator(model_path=self.model_path)
        points = []
        for i in range(33):
            # Place left shoulder near left edge (x=0.01) and head near top edge (y=0.01)
            x = 0.01 if i == 11 else 0.5
            y = 0.01 if i == 0 else 0.5
            points.append(
                LandmarkPoint(
                    landmark_id=i,
                    name=f"LM_{i}",
                    x_norm=x,
                    y_norm=y,
                    z_norm=0.0,
                    x_px=int(x * 640),
                    y_px=int(y * 480),
                    visibility=0.9,
                    presence=0.9,
                    is_visible=True,
                    is_present=True,
                    is_in_bounds=True,
                    is_reliable=True
                )
            )

        quality = estimator.evaluate_pose_quality(points, edge_margin=0.03)
        self.assertTrue(quality.is_partially_out_of_frame)
        self.assertIn("LEFT", quality.boundary_violations)
        self.assertIn("TOP", quality.boundary_violations)
        self.assertIsNotNone(quality.warning_message)
        estimator.close()

    def test_invalid_pose_data(self):
        """Verify that NaNs, Infinities, and out-of-bound coords are handled cleanly."""
        estimator = PoseEstimator(model_path=self.model_path)
        points = []
        for i in range(33):
            if i == 0:
                x, y = float("nan"), float("nan")
            elif i == 1:
                x, y = float("inf"), 0.5
            elif i == 2:
                x, y = -0.5, 1.5  # Out of image bounds
            else:
                x, y = 0.5, 0.5

            is_valid = not (math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y))
            in_bounds = is_valid and (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0)

            points.append(
                LandmarkPoint(
                    landmark_id=i,
                    name=f"LM_{i}",
                    x_norm=x if is_valid else 0.0,
                    y_norm=y if is_valid else 0.0,
                    z_norm=0.0,
                    x_px=320 if in_bounds else 0,
                    y_px=240 if in_bounds else 0,
                    visibility=0.9,
                    presence=0.9,
                    is_visible=True,
                    is_present=True,
                    is_in_bounds=in_bounds,
                    is_reliable=in_bounds
                )
            )

        reliable_map = estimator.get_reliable_landmarks(points)
        # Invalid and out-of-bound points must be rejected
        self.assertNotIn(0, reliable_map)
        self.assertNotIn(1, reliable_map)
        self.assertNotIn(2, reliable_map)
        self.assertIn(3, reliable_map)
        estimator.close()


if __name__ == "__main__":
    unittest.main()
