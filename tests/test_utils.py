"""Tests for system utilities."""
import time
import unittest
import numpy as np
from src.utils import FPSCalculator, draw_hud, ensure_model_exists


class TestUtils(unittest.TestCase):
    """Test suite for utilities."""

    def test_fps_calculator(self):
        """Verify FPS calculation behavior."""
        fps_calc = FPSCalculator(window_size=5)
        self.assertEqual(fps_calc.fps, 0.0)

        # Tick multiple times
        for _ in range(5):
            fps_calc.tick()
            time.sleep(0.01)

        self.assertGreater(fps_calc.fps, 0.0)

        # Reset
        fps_calc.reset()
        self.assertEqual(fps_calc.fps, 0.0)

    def test_draw_hud(self):
        """Verify HUD rendering output."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hud_frame = draw_hud(
            frame=frame,
            fps=30.0,
            camera_status="Live Streaming",
            pose_detected=True,
            resolution=(640, 480)
        )
        self.assertEqual(hud_frame.shape, (480, 640, 3))
        # Frame should have drawn content (pixel values > 0)
        self.assertGreater(hud_frame.max(), 0)

    def test_ensure_model_exists(self):
        """Verify model validation returns valid path."""
        path = ensure_model_exists()
        self.assertTrue(path.endswith(".task"))


if __name__ == "__main__":
    unittest.main()
