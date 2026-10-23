"""Unit and integration tests for traditional image processing motion analysis."""
import unittest
import numpy as np
import cv2
from src.motion import MotionAnalyzer, MotionResult


class TestMotionAnalyzer(unittest.TestCase):
    """Test suite for MotionAnalyzer using synthetic generated image arrays."""

    def setUp(self):
        self.analyzer = MotionAnalyzer(
            blur_kernel_size=(5, 5),
            diff_threshold=20,
            min_contour_area=50.0,
            morph_kernel_size=(3, 3),
            motion_score_threshold=0.005
        )

    def test_preprocess_frame(self):
        """Verify grayscale conversion and Gaussian smoothing on synthetic frame."""
        # Create synthetic 3-channel BGR frame
        bgr = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Add high-frequency noise spike
        bgr[50, 50] = [255, 255, 255]

        blurred = self.analyzer.preprocess_frame(bgr)
        self.assertEqual(blurred.shape, (100, 100))
        self.assertEqual(len(blurred.shape), 2)
        # Gaussian smoothing should attenuate the single-pixel noise spike
        self.assertLess(blurred[50, 50], 255)

    def test_static_consecutive_frames_zero_motion(self):
        """Identical consecutive frames must produce zero motion difference."""
        frame = np.full((120, 160, 3), 70, dtype=np.uint8)

        res1 = self.analyzer.process_frame(frame)
        self.assertFalse(res1.has_motion)
        self.assertEqual(res1.motion_score, 0.0)

        res2 = self.analyzer.process_frame(frame)
        self.assertFalse(res2.has_motion)
        self.assertEqual(res2.motion_score, 0.0)
        self.assertEqual(res2.contours_count, 0)
        self.assertIsNone(res2.bounding_box)

    def test_moving_block_detection(self):
        """Moving synthetic square block must be detected and bounded."""
        f1 = np.zeros((200, 200, 3), dtype=np.uint8)
        self.analyzer.process_frame(f1)

        # Frame 2: introduce a white 50x50 block at (50, 50)
        f2 = f1.copy()
        cv2.rectangle(f2, (50, 50), (100, 100), (255, 255, 255), -1)

        res = self.analyzer.process_frame(f2)
        self.assertTrue(res.has_motion)
        self.assertGreater(res.motion_score, 0.0)
        self.assertIsNotNone(res.bounding_box)
        self.assertIsNotNone(res.motion_centroid)

        bx, by, bw, bh = res.bounding_box
        # Bounding box should enclose the (50, 50, 50, 50) block with morphological padding
        self.assertLessEqual(bx, 50)
        self.assertLessEqual(by, 50)
        self.assertGreaterEqual(bx + bw, 100)
        self.assertGreaterEqual(by + bh, 100)

    def test_morphological_noise_rejection(self):
        """Isolated 1-pixel noise should be eliminated by morphological opening."""
        diff = np.zeros((100, 100), dtype=np.uint8)
        # Add 3 isolated single-pixel noise spikes
        diff[10, 10] = 255
        diff[20, 80] = 255
        diff[75, 40] = 255

        mask = self.analyzer.threshold_and_morphology(diff)
        # Noise spikes should be completely eliminated (all zeros)
        self.assertEqual(np.count_nonzero(mask), 0)

    def test_reset_functionality(self):
        """Reset clears previous frame history."""
        f = np.full((50, 50, 3), 100, dtype=np.uint8)
        self.analyzer.process_frame(f)
        self.assertIsNotNone(self.analyzer._previous_gray)

        self.analyzer.reset()
        self.assertIsNone(self.analyzer._previous_gray)

    def test_empty_frame_handling(self):
        """Empty or None frames return safely without raising unhandled errors."""
        empty = np.array([], dtype=np.uint8)
        res = self.analyzer.process_frame(empty)
        self.assertFalse(res.has_motion)


if __name__ == "__main__":
    unittest.main()
