"""Utility functions and helper classes for fall detection system.

Includes:
- FPS calculation with rolling window averaging.
- Automatic MediaPipe model downloading and validation.
- Clean on-frame HUD overlay with pose reliability, motion status, and fall state banners.
"""
from collections import deque
import os
from pathlib import Path
import time
from typing import Any, Optional, Tuple
import urllib.request
import cv2
import numpy as np


# Official MediaPipe Pose Landmarker model URL (Lite model ~5.8 MB)
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = os.path.join("models", "pose_landmarker_lite.task")


class FPSCalculator:
    """Calculates smoothed frames-per-second using a rolling time window."""

    def __init__(self, window_size: int = 30):
        """Initializes FPS calculator with a given rolling window size."""
        self.window_size = max(2, window_size)
        self._timestamps = deque(maxlen=self.window_size)
        self._last_fps = 0.0

    def tick(self) -> float:
        """Records a new frame timestamp and returns updated smoothed FPS."""
        now = time.perf_counter()
        self._timestamps.append(now)

        if len(self._timestamps) > 1:
            total_duration = self._timestamps[-1] - self._timestamps[0]
            if total_duration > 0:
                frame_count = len(self._timestamps) - 1
                self._last_fps = frame_count / total_duration
        return self._last_fps

    @property
    def fps(self) -> float:
        """Returns the most recently computed FPS."""
        return self._last_fps

    def reset(self) -> None:
        """Resets recorded timestamps and FPS counter."""
        self._timestamps.clear()
        self._last_fps = 0.0


def ensure_model_exists(
    model_path: Optional[str] = None,
    download_url: str = DEFAULT_MODEL_URL
) -> str:
    """Ensures the MediaPipe Pose model exists locally, downloading if missing.

    Args:
        model_path: Path where model should be stored. Defaults to models/pose_landmarker_lite.task.
        download_url: Official URL to fetch model if missing.

    Returns:
        Absolute path to the validated model file.

    Raises:
        RuntimeError: If download fails and file cannot be found.
    """
    if model_path is None:
        # Resolve path relative to project root
        project_root = Path(__file__).resolve().parent.parent
        target_path = project_root / "models" / "pose_landmarker_lite.task"
    else:
        target_path = Path(model_path).resolve()

    if not target_path.exists() or target_path.stat().st_size == 0:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"Downloading pose landmarker model to {target_path} ...")
            urllib.request.urlretrieve(download_url, str(target_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download MediaPipe pose landmarker model from {download_url}: {exc}"
            ) from exc

    return str(target_path)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    camera_status: str = "Connected",
    pose_detected: bool = False,
    resolution: Optional[Tuple[int, int]] = None,
    pose_quality: Optional[Any] = None,
    fall_status: Optional[Any] = None
) -> np.ndarray:
    """Draws a diagnostic HUD overlay and fall state indicators on the frame.

    Args:
        frame: OpenCV BGR image.
        fps: Current measured FPS.
        camera_status: Human-readable camera status string.
        pose_detected: True if a human pose is actively tracked.
        resolution: (width, height) tuple if known.
        pose_quality: Optional PoseQuality object indicating tracking reliability.
        fall_status: Optional FallDetectorStatus object indicating fall state.

    Returns:
        Annotated BGR image with HUD badge and active state alert banners.
    """
    output = frame.copy()
    h, w = output.shape[:2]

    has_warning = (pose_quality is not None and getattr(pose_quality, "warning_message", None) is not None)
    has_fall = (fall_status is not None)

    box_height = 138 if (has_warning or has_fall) else 105
    box_width = 330

    # Semi-transparent overlay banner in top-left
    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (box_width, box_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, output, 0.28, 0, output)
    cv2.rectangle(output, (10, 10), (box_width, box_height), (80, 80, 80), 1)

    # Line 1: System Title
    cv2.putText(
        output,
        "FALL MONITOR & POSE TRACKING",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # Line 2: Camera Status & Resolution
    res_str = f" [{resolution[0]}x{resolution[1]}]" if resolution else ""
    cam_color = (100, 255, 100) if "Connected" in camera_status or "Live" in camera_status else (100, 100, 255)
    cv2.putText(
        output,
        f"CAM: {camera_status}{res_str}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        cam_color,
        1,
        cv2.LINE_AA
    )

    # Line 3: Live Measured FPS
    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 230, 255),
        1,
        cv2.LINE_AA
    )

    # Line 4: Pose Tracking & Reliability
    if pose_quality is not None and getattr(pose_quality, "is_detected", False):
        rel_status = getattr(pose_quality, "reliability_status", "RELIABLE")
        if rel_status == "RELIABLE":
            pose_str = "POSE: RELIABLE"
            pose_color = (50, 255, 50)
        elif rel_status == "DEGRADED":
            pose_str = "POSE: DEGRADED (OCCLUDED)"
            pose_color = (0, 215, 255)
        elif rel_status == "UNRELIABLE":
            pose_str = "POSE: UNRELIABLE (OCCLUDED)"
            pose_color = (0, 50, 255)
        elif rel_status == "OUT_OF_FRAME":
            pose_str = "POSE: CLIPPED AT BOUNDARY"
            pose_color = (0, 100, 255)
        else:
            pose_str = "POSE: TRACKED"
            pose_color = (50, 255, 50)
    elif pose_detected:
        pose_str = "POSE: TRACKED"
        pose_color = (50, 255, 50)
    else:
        pose_str = "POSE: SEARCHING..."
        pose_color = (0, 165, 255)

    cv2.putText(
        output,
        pose_str,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        pose_color,
        1,
        cv2.LINE_AA
    )

    # Line 5: Fall Detection State
    if fall_status is not None:
        state_val = getattr(fall_status, "current_state", None)
        state_str = state_val.value if hasattr(state_val, "value") else str(state_val)
        is_susp = getattr(fall_status, "is_decision_suspended", False)

        if state_str == "FALL_CONFIRMED":
            st_color = (0, 0, 255)  # Bright Red
            txt = "STATE: FALL CONFIRMED"
        elif state_str == "POSSIBLE_FALL":
            st_color = (0, 165, 255)  # Amber
            txt = "STATE: POSSIBLE FALL (VERIFYING)"
        elif state_str == "RECOVERY":
            st_color = (255, 255, 0)  # Cyan
            txt = "STATE: RECOVERY IN PROGRESS"
        elif state_str == "UNCERTAIN":
            st_color = (0, 140, 255)  # Orange
            txt = "STATE: UNCERTAIN (OCCLUDED/LOW CONF)"
        elif is_susp:
            st_color = (0, 215, 255)  # Yellow
            txt = "STATE: NORMAL [VIEW SUSPENDED]"
        else:
            st_color = (100, 255, 100)  # Green
            txt = "STATE: NORMAL (MONITORING)"

        cv2.putText(
            output,
            txt,
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            st_color,
            1,
            cv2.LINE_AA
        )

        # Line 6: Advisory message if view suspended or warning active
        if is_susp:
            cv2.putText(
                output,
                "UPPER-BODY VIEW: CCTV REQUIRED",
                (20, 128),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 215, 255),
                1,
                cv2.LINE_AA
            )
        elif state_str == "UNCERTAIN":
            cv2.putText(
                output,
                "TRACKING LOST POST-IMPACT",
                (20, 128),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 140, 255),
                1,
                cv2.LINE_AA
            )
        elif has_warning:
            warn_msg = str(getattr(pose_quality, "warning_message", ""))
            cv2.putText(
                output,
                f"! {warn_msg[:36]}",
                (20, 128),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 140, 255),
                1,
                cv2.LINE_AA
            )

        # High-visibility alerts across top of frame
        if state_str == "FALL_CONFIRMED":
            # Red border around entire frame
            cv2.rectangle(output, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
            # Banner
            cv2.rectangle(output, (0, 0), (w, 42), (0, 0, 200), -1)
            cv2.putText(
                output,
                "EMERGENCY: FALL CONFIRMED - FLOOR IMMOBILITY DETECTED",
                (int(w * 0.05), 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
        elif state_str == "POSSIBLE_FALL":
            cv2.rectangle(output, (0, 0), (w, 36), (0, 140, 255), -1)
            cv2.putText(
                output,
                "WARNING: RAPID DESCENT DETECTED - VERIFYING POSTURE...",
                (int(w * 0.05), 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )
        elif state_str == "UNCERTAIN":
            cv2.rectangle(output, (0, 0), (w, 36), (0, 100, 255), -1)
            cv2.putText(
                output,
                "ATTENTION: TRACKING UNCERTAIN - PERSON OBSCURED",
                (int(w * 0.05), 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    return output
