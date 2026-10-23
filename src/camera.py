"""Camera capture module for live webcam streaming using OpenCV.

Handles camera lifecycle, frame capture, resolution configuration,
threaded frame acquisition with a bounded latest-frame buffer,
graceful error handling for unavailable devices, and clean resource release.
"""
from dataclasses import dataclass
import logging
import threading
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraStatus:
    """Current state of the webcam device."""
    is_open: bool
    device_index: int
    resolution: Tuple[int, int]  # (width, height)
    error_message: Optional[str] = None


class CameraCapture:
    """Robust OpenCV webcam capture wrapper with automatic resource cleanup.

    Supports asynchronous background frame acquisition with a strictly bounded
    latest-frame buffer (size = 1) to decouple camera shutter synchronization
    from downstream inference without queue buildup or stale frame processing.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        threaded: bool = True
    ):
        """Initializes camera capture configuration.

        Args:
            device_index: Integer index of webcam device (default: 0).
            width: Optional target width in pixels.
            height: Optional target height in pixels.
            threaded: If True, uses background capture thread with bounded buffer.
        """
        self.device_index = int(device_index)
        self.target_width = width
        self.target_height = height
        self.threaded = threaded

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened: bool = False
        self._last_error: Optional[str] = None

        # Threaded acquisition synchronization (strictly bounded buffer of size 1)
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id: int = 0

    def open(self) -> bool:
        """Opens the camera device and verifies operational frame capture.

        Returns:
            True if camera is open and successfully delivers frames, False otherwise.
        """
        if self._is_opened and self._cap is not None and self._cap.isOpened():
            return True

        self.release()
        self._last_error = None

        try:
            # Initialize OpenCV VideoCapture for the specified device index
            cap = cv2.VideoCapture(self.device_index)

            if not cap.isOpened():
                cap.release()
                self._last_error = f"Cannot open camera index {self.device_index}. Device may be in use or disconnected."
                logger.warning(self._last_error)
                return False

            # Set hardware buffer size to 1 if supported by backend
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Request specific frame resolution if specified
            if self.target_width is not None and self.target_width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            if self.target_height is not None and self.target_height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

            # Verification: Attempt to read one live frame to confirm device operational
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                cap.release()
                self._last_error = f"Camera index {self.device_index} opened but failed to read initial frame."
                logger.warning(self._last_error)
                return False

            self._cap = cap
            self._is_opened = True
            self._latest_frame = test_frame
            self._frame_id = 1

            # Start threaded capture worker if enabled
            if self.threaded:
                self._running = True
                self._thread = threading.Thread(
                    target=self._capture_worker,
                    name=f"CameraWorker-{self.device_index}",
                    daemon=True
                )
                self._thread.start()

            logger.info("Camera %d opened successfully. Resolution: %s (threaded=%s)",
                        self.device_index, self.get_resolution(), self.threaded)
            return True

        except Exception as exc:
            self.release()
            self._last_error = f"Unexpected error opening camera {self.device_index}: {exc}"
            logger.error(self._last_error, exc_info=True)
            return False

    def _capture_worker(self) -> None:
        """Background thread worker continuously acquiring latest frame."""
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            with self._cond:
                # Update bounded single-slot buffer (size = 1)
                self._latest_frame = frame
                self._frame_id += 1
                self._cond.notify_all()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a single frame from the active camera.

        Backward-compatible interface for synchronous consumers.

        Returns:
            Tuple of (success_flag, bgr_frame_or_None).
        """
        if not self._is_opened or self._cap is None:
            return False, None

        if self.threaded:
            with self._cond:
                if self._latest_frame is None:
                    return False, None
                return True, self._latest_frame.copy()

        try:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._last_error = "Webcam stream disconnected or failed to read frame."
                return False, None
            return True, frame
        except Exception as exc:
            self._last_error = f"Exception during frame read: {exc}"
            return False, None

    def read_latest_frame(
        self,
        last_frame_id: int = -1,
        timeout: float = 0.1
    ) -> Tuple[bool, Optional[np.ndarray], int]:
        """Waits for and returns a newly acquired frame.

        Guarantees that downstream inference never processes the same frame
        multiple times unnecessarily. Employs a bounded 1-frame buffer to prevent
        latency buildup and excessive memory consumption.

        Args:
            last_frame_id: The sequence ID of the last frame processed by caller.
            timeout: Maximum seconds to wait for a new hardware frame.

        Returns:
            Tuple of (success_flag, frame_or_None, current_frame_id).
        """
        if not self._is_opened or self._cap is None:
            return False, None, last_frame_id

        if not self.threaded:
            ret, frame = self.read_frame()
            if ret and frame is not None:
                self._frame_id += 1
                return True, frame, self._frame_id
            return False, None, last_frame_id

        with self._cond:
            # If caller already processed current frame, wait for next frame
            if self._frame_id == last_frame_id:
                self._cond.wait_for(
                    lambda: self._frame_id > last_frame_id or not self._running,
                    timeout=timeout
                )

            if not self._running or self._latest_frame is None:
                return False, None, last_frame_id

            if self._frame_id > last_frame_id:
                return True, self._latest_frame, self._frame_id

            # Timeout expired without new hardware frame
            return False, None, last_frame_id

    def get_resolution(self) -> Tuple[int, int]:
        """Returns the actual resolution (width, height) of the video feed."""
        if self._cap is not None and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (0, 0)

    def is_opened(self) -> bool:
        """Returns True if the camera is currently open and functional."""
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    def get_status(self) -> CameraStatus:
        """Returns the current structured status of the camera."""
        return CameraStatus(
            is_open=self.is_opened(),
            device_index=self.device_index,
            resolution=self.get_resolution(),
            error_message=self._last_error
        )

    def release(self) -> None:
        """Cleanly releases the camera capture hardware and stops worker thread."""
        self._running = False
        with self._cond:
            self._cond.notify_all()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.6)
        self._thread = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:
                logger.debug("Error releasing VideoCapture: %s", exc)
            self._cap = None

        self._latest_frame = None
        self._is_opened = False

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()

    @staticmethod
    def list_available_cameras(max_tested: int = 4) -> List[int]:
        """Scans camera indices to discover available physical webcam devices.

        Args:
            max_tested: Maximum number of consecutive device indices to probe.

        Returns:
            List of accessible camera indices.
        """
        available: List[int] = []
        for idx in range(max_tested):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
                cap.release()
        return available
