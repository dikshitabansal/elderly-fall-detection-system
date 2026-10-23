"""Tests for camera capture module."""
import unittest
import numpy as np
from src.camera import CameraCapture, CameraStatus


class TestCameraCapture(unittest.TestCase):
    """Test suite for CameraCapture class."""

    def test_camera_init_defaults(self):
        """Verify CameraCapture initialization with default values."""
        cam = CameraCapture(device_index=0)
        self.assertEqual(cam.device_index, 0)
        self.assertFalse(cam.is_opened())
        self.assertEqual(cam.get_resolution(), (0, 0))
        status = cam.get_status()
        self.assertIsInstance(status, CameraStatus)
        self.assertFalse(status.is_open)

    def test_camera_unavailable_index(self):
        """Verify that an unavailable camera index fails gracefully and cleans up."""
        cam = CameraCapture(device_index=99)
        opened = cam.open()
        self.assertFalse(opened)
        self.assertFalse(cam.is_opened())

        # Attempt to read frame from closed camera
        ret, frame = cam.read_frame()
        self.assertFalse(ret)
        self.assertIsNone(frame)

        # Status check
        status = cam.get_status()
        self.assertFalse(status.is_open)
        self.assertIsNotNone(status.error_message)

        # Clean release should not throw
        cam.release()
        self.assertFalse(cam.is_opened())

    def test_camera_context_manager(self):
        """Verify context manager releases resources on exit."""
        with CameraCapture(device_index=99) as cam:
            self.assertFalse(cam.is_opened())
        self.assertFalse(cam.is_opened())

    def test_camera_0_if_available(self):
        """Test physical camera 0 if hardware is available."""
        cam = CameraCapture(device_index=0)
        opened = cam.open()
        if opened:
            self.assertTrue(cam.is_opened())
            ret, frame = cam.read_frame()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(len(frame.shape), 3)

            res = cam.get_resolution()
            self.assertGreater(res[0], 0)
            self.assertGreater(res[1], 0)

            cam.release()
            self.assertFalse(cam.is_opened())
        else:
            # If device 0 is not physically connected in this environment, open returns False cleanly
            self.assertFalse(cam.is_opened())

    def test_camera_read_latest_frame_unavailable(self):
        """Verify read_latest_frame returns False cleanly when camera is closed."""
        cam = CameraCapture(device_index=99, threaded=True)
        ret, frame, new_id = cam.read_latest_frame(last_frame_id=5, timeout=0.01)
        self.assertFalse(ret)
        self.assertIsNone(frame)
        self.assertEqual(new_id, 5)

    def test_camera_read_latest_frame_deduplication(self):
        """Verify read_latest_frame with live camera increments frame IDs and deduplicates."""
        cam = CameraCapture(device_index=0, threaded=True)
        opened = cam.open()
        if opened:
            # First read with -1 should yield frame_id >= 1
            ret1, frame1, id1 = cam.read_latest_frame(last_frame_id=-1, timeout=0.1)
            self.assertTrue(ret1)
            self.assertIsNotNone(frame1)
            self.assertGreater(id1, 0)

            # Second read with new id should give id2 > id1
            ret2, frame2, id2 = cam.read_latest_frame(last_frame_id=id1, timeout=0.1)
            if ret2:
                self.assertGreater(id2, id1)

            cam.release()
            self.assertFalse(cam.is_opened())


if __name__ == "__main__":
    unittest.main()

