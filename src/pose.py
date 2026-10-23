"""Pose estimation module using Google MediaPipe PoseLandmarker.

Estimates 33 body landmarks in real-time from camera frames, assesses tracking
reliability and occlusion, detects out-of-frame boundaries, and renders a
confidence-aware skeletal overlay.
"""
from dataclasses import dataclass, field
import logging
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarksConnections

from src.utils import ensure_model_exists

logger = logging.getLogger(__name__)

# ==============================================================================
# Canonical MediaPipe Pose Topology Definitions (33 Landmarks, 35 Connections)
# ==============================================================================
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS
]

# Key Posture Landmarks required for reliable body orientation & fall detection
KEY_TORSO_LANDMARKS: List[int] = [
    vision.PoseLandmark.LEFT_SHOULDER.value,   # 11
    vision.PoseLandmark.RIGHT_SHOULDER.value,  # 12
    vision.PoseLandmark.LEFT_HIP.value,        # 23
    vision.PoseLandmark.RIGHT_HIP.value,       # 24
]

KEY_LOWER_BODY_LANDMARKS: List[int] = [
    vision.PoseLandmark.LEFT_KNEE.value,       # 25
    vision.PoseLandmark.RIGHT_KNEE.value,      # 26
    vision.PoseLandmark.LEFT_ANKLE.value,      # 27
    vision.PoseLandmark.RIGHT_ANKLE.value,     # 28
]

KEY_UPPER_BODY_LANDMARKS: List[int] = [
    vision.PoseLandmark.NOSE.value,            # 0
    vision.PoseLandmark.LEFT_ELBOW.value,      # 13
    vision.PoseLandmark.RIGHT_ELBOW.value,     # 14
    vision.PoseLandmark.LEFT_WRIST.value,      # 15
    vision.PoseLandmark.RIGHT_WRIST.value,     # 16
]

LANDMARK_NAMES: Dict[int, str] = {
    idx: vision.PoseLandmark(idx).name for idx in range(len(vision.PoseLandmark))
}


@dataclass
class LandmarkPoint:
    """Represents a single body landmark point with confidence & bounds evaluation."""
    landmark_id: int
    name: str
    x_norm: float
    y_norm: float
    z_norm: float
    x_px: int
    y_px: int
    visibility: float
    presence: float
    is_visible: bool = True
    is_present: bool = True
    is_in_bounds: bool = True
    is_reliable: bool = True


@dataclass
class PoseQuality:
    """Comprehensive diagnostic evaluation of tracking reliability & occlusion."""
    is_detected: bool = False
    is_reliable: bool = False
    is_partially_out_of_frame: bool = False
    reliability_status: str = "NO_PERSON"  # "RELIABLE", "DEGRADED", "UNRELIABLE", "NO_PERSON"
    visibility_score: float = 0.0
    tracked_count: int = 0
    missing_key_landmarks: List[str] = field(default_factory=list)
    boundary_violations: List[str] = field(default_factory=list)
    warning_message: Optional[str] = None
    confidence_score: float = 0.0
    available_keypoints: List[str] = field(default_factory=list)


class ProcessResult(tuple):
    """Container for process_frame result with backward-compatible 3-tuple unpacking."""
    def __new__(cls, annotated_frame: np.ndarray, detection_result: Any, pose_detected: bool, pose_quality: Optional[PoseQuality] = None):
        return super().__new__(cls, (annotated_frame, detection_result, pose_detected))

    def __init__(self, annotated_frame: np.ndarray, detection_result: Any, pose_detected: bool, pose_quality: Optional[PoseQuality] = None):
        self.annotated_frame = annotated_frame
        self.detection_result = detection_result
        self.pose_detected = pose_detected
        self.pose_quality = pose_quality or PoseQuality()


class PoseEstimator:
    """Wrapper for MediaPipe PoseLandmarker with temporal video tracking & quality checks."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_pose_detection_confidence: float = 0.35,
        min_pose_presence_confidence: float = 0.35,
        min_tracking_confidence: float = 0.35,
        num_poses: int = 1,
        inference_resolution: Optional[Tuple[int, int]] = None,
        running_mode: Optional[vision.RunningMode] = None
    ):
        """Initializes MediaPipe PoseLandmarker with optimized video tracking.

        Args:
            model_path: Path to pose_landmarker_lite.task (auto-downloaded if None).
            min_pose_detection_confidence: Minimum confidence for pose detection.
            min_pose_presence_confidence: Minimum confidence for pose presence.
            min_tracking_confidence: Minimum confidence for landmark tracking.
            num_poses: Maximum number of human poses to detect simultaneously.
            inference_resolution: Optional (width, height) to downscale for faster inference.
            running_mode: RunningMode.VIDEO (default for live tracking) or RunningMode.IMAGE.
        """
        self.model_path = ensure_model_exists(model_path)
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_pose_presence_confidence = min_pose_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.num_poses = num_poses
        self.inference_resolution = inference_resolution
        self.running_mode = running_mode or vision.RunningMode.VIDEO
        self.detector: Optional[vision.PoseLandmarker] = None
        self._last_timestamp_ms: int = 0
        self.last_pose_quality: PoseQuality = PoseQuality()

        self._init_detector()

    def _init_detector(self) -> None:
        """Configures and initializes the underlying vision.PoseLandmarker."""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=self.running_mode,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        logger.info("MediaPipe PoseLandmarker initialized (%s) from: %s", self.running_mode.name, self.model_path)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        draw_skeleton: bool = True,
        inference_resolution: Optional[Tuple[int, int]] = None,
        timestamp_ms: Optional[int] = None
    ) -> ProcessResult:
        """Detects body landmarks on a BGR camera frame and assesses quality.

        Returns a ProcessResult which unpacks as a standard (annotated_bgr, detection_result, pose_detected)
        3-tuple for full backward compatibility, while exposing .pose_quality.
        """
        if self.detector is None:
            raise RuntimeError("PoseEstimator detector is not initialized or has been closed.")

        if frame_bgr is None or frame_bgr.size == 0:
            quality = PoseQuality(is_detected=False, reliability_status="NO_PERSON")
            self.last_pose_quality = quality
            return ProcessResult(frame_bgr, None, False, quality)

        annotated_bgr = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]

        # Determine inference resolution preserving aspect ratio to avoid body geometry distortion
        target_res = inference_resolution or self.inference_resolution
        if target_res is not None:
            max_dim = max(target_res)
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                new_w = max(16, int(round(w * scale)))
                new_h = max(16, int(round(h * scale)))
                infer_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                infer_bgr = frame_bgr
        else:
            infer_bgr = frame_bgr

        # Convert OpenCV BGR to RGB required by MediaPipe
        rgb_frame = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Execute pose landmark estimation with appropriate running mode
        if self.running_mode == vision.RunningMode.VIDEO:
            if timestamp_ms is None:
                now_ms = int(time.perf_counter() * 1000)
                if now_ms <= self._last_timestamp_ms:
                    now_ms = self._last_timestamp_ms + 1
                self._last_timestamp_ms = now_ms
                ts = now_ms
            else:
                if timestamp_ms <= self._last_timestamp_ms:
                    timestamp_ms = self._last_timestamp_ms + 1
                self._last_timestamp_ms = timestamp_ms
                ts = timestamp_ms
            detection_result = self.detector.detect_for_video(mp_image, ts)
        else:
            detection_result = self.detector.detect(mp_image)

        pose_detected = bool(detection_result and detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0)

        # Quality evaluation & landmark extraction
        quality = PoseQuality()
        extracted_landmarks: List[List[LandmarkPoint]] = []

        if pose_detected:
            extracted_landmarks = self.extract_landmarks(detection_result, (h, w))
            if extracted_landmarks:
                primary_pose = extracted_landmarks[0]
                quality = self.evaluate_pose_quality(primary_pose)

        self.last_pose_quality = quality

        # Render confidence-aware skeletal overlay and edge boundary alerts
        if draw_skeleton and pose_detected and extracted_landmarks:
            self._draw_confidence_skeleton(annotated_bgr, extracted_landmarks[0], quality)

        return ProcessResult(annotated_bgr, detection_result, pose_detected, quality)

    def extract_landmarks(
        self,
        detection_result: Any,
        image_shape: Tuple[int, int]
    ) -> List[List[LandmarkPoint]]:
        """Extracts structured landmarks with visibility, presence, and bound checks.

        Args:
            detection_result: PoseLandmarkerResult from process_frame.
            image_shape: (height, width) of the corresponding frame.

        Returns:
            List of detected persons, each containing 33 LandmarkPoint instances.
        """
        if not detection_result or not detection_result.pose_landmarks:
            return []

        h, w = image_shape[:2]
        all_poses: List[List[LandmarkPoint]] = []

        for pose in detection_result.pose_landmarks:
            pose_points: List[LandmarkPoint] = []
            for idx, lm in enumerate(pose):
                x_val = float(lm.x)
                y_val = float(lm.y)
                z_val = float(lm.z) if lm.z is not None else 0.0

                # Validate numeric sanity
                if math.isnan(x_val) or math.isinf(x_val):
                    x_val = 0.0
                if math.isnan(y_val) or math.isinf(y_val):
                    y_val = 0.0

                px_x = int(round(np.clip(x_val * w, 0, w - 1)))
                px_y = int(round(np.clip(y_val * h, 0, h - 1)))

                vis = float(lm.visibility) if lm.visibility is not None else 1.0
                pres = float(lm.presence) if lm.presence is not None else 1.0

                # Check confidence and bounding limits
                is_vis = vis >= self.min_pose_presence_confidence
                is_pres = pres >= self.min_pose_presence_confidence
                is_in_bounds = (0.0 <= x_val <= 1.0) and (0.0 <= y_val <= 1.0)
                is_rel = is_vis and is_pres and is_in_bounds

                name = LANDMARK_NAMES.get(idx, f"LANDMARK_{idx}")

                pose_points.append(
                    LandmarkPoint(
                        landmark_id=idx,
                        name=name,
                        x_norm=x_val,
                        y_norm=y_val,
                        z_norm=z_val,
                        x_px=px_x,
                        y_px=px_y,
                        visibility=vis,
                        presence=pres,
                        is_visible=is_vis,
                        is_present=is_pres,
                        is_in_bounds=is_in_bounds,
                        is_reliable=is_rel,
                    )
                )
            all_poses.append(pose_points)

        return all_poses

    @staticmethod
    def get_reliable_landmarks(
        landmarks: List[LandmarkPoint],
        min_visibility: float = 0.5,
        min_presence: float = 0.5
    ) -> Dict[int, LandmarkPoint]:
        """Filters out low-confidence, occluded, or out-of-frame landmarks.

        Downstream posture and fall algorithms MUST call this method to avoid
        computing angles or center-of-mass on hallucinated or occluded points.

        Args:
            landmarks: List of 33 LandmarkPoint instances for a subject.
            min_visibility: Threshold for visibility confidence (not occluded).
            min_presence: Threshold for presence confidence (in-frame).

        Returns:
            Dictionary mapping landmark_id -> LandmarkPoint for reliable points only.
        """
        reliable: Dict[int, LandmarkPoint] = {}
        for lm in landmarks:
            if (
                lm.visibility >= min_visibility
                and lm.presence >= min_presence
                and lm.is_in_bounds
                and not (math.isnan(lm.x_norm) or math.isnan(lm.y_norm))
            ):
                reliable[lm.landmark_id] = lm
        return reliable

    def evaluate_pose_quality(
        self,
        landmarks: List[LandmarkPoint],
        edge_margin: float = 0.03
    ) -> PoseQuality:
        """Assesses pose tracking reliability, keypoint occlusion, and boundary clipping.

        Args:
            landmarks: 33 LandmarkPoint instances.
            edge_margin: Fractional margin (0.03 = 3%) to detect frame clipping.

        Returns:
            Structured PoseQuality diagnostic object.
        """
        if not landmarks or len(landmarks) < 33:
            return PoseQuality(
                is_detected=False,
                is_reliable=False,
                reliability_status="UNRELIABLE",
                warning_message="Incomplete landmark set detected."
            )

        # 1. Edge Boundary Clipping Detection
        violations: Set[str] = set()
        reliable_count = 0
        key_torso_visibilities = []

        for lm in landmarks:
            if lm.is_reliable:
                reliable_count += 1

            # Check boundary proximity for visible landmarks
            if lm.visibility > 0.35 and lm.presence > 0.35:
                if lm.x_norm <= edge_margin:
                    violations.add("LEFT")
                if lm.x_norm >= 1.0 - edge_margin:
                    violations.add("RIGHT")
                if lm.y_norm <= edge_margin:
                    violations.add("TOP")
                if lm.y_norm >= 1.0 - edge_margin:
                    violations.add("BOTTOM")

        boundary_list = sorted(list(violations))
        is_partially_out = len(boundary_list) > 0

        # 2. Key Torso & Posture Occlusion Evaluation
        missing_keys: List[str] = []
        torso_reliable_count = 0

        for torso_id in KEY_TORSO_LANDMARKS:
            lm = landmarks[torso_id]
            key_torso_visibilities.append(lm.visibility)
            if lm.is_reliable:
                torso_reliable_count += 1
            else:
                missing_keys.append(lm.name)

        lower_body_reliable = 0
        for leg_id in KEY_LOWER_BODY_LANDMARKS:
            if landmarks[leg_id].is_reliable:
                lower_body_reliable += 1

        avg_vis = float(np.mean(key_torso_visibilities)) if key_torso_visibilities else 0.0

        # Torso integrity: Shoulders (11, 12) and Hips (23, 24)
        left_shoulder_ok = landmarks[11].is_reliable
        right_shoulder_ok = landmarks[12].is_reliable
        left_hip_ok = landmarks[23].is_reliable
        right_hip_ok = landmarks[24].is_reliable

        shoulders_missing = not (left_shoulder_ok or right_shoulder_ok)
        hips_missing = not (left_hip_ok or right_hip_ok)

        # 3. Determine Overall Reliability Status
        if shoulders_missing or hips_missing or torso_reliable_count < 2:
            status = "UNRELIABLE"
            is_reliable = False
            warning = f"Tracking Unreliable: Key torso landmarks occluded ({', '.join(missing_keys[:2])})."
        elif is_partially_out and (shoulders_missing or hips_missing):
            status = "OUT_OF_FRAME"
            is_reliable = False
            warning = f"Subject partially out of frame ({', '.join(boundary_list)} edge)."
        elif missing_keys or lower_body_reliable < 2:
            status = "DEGRADED"
            is_reliable = True  # Can still perform partial torso analysis with caution
            warning = f"Tracking Degraded: Occlusion on {missing_keys[0] if missing_keys else 'lower body'}."
        elif is_partially_out:
            status = "DEGRADED"
            is_reliable = True
            warning = f"Subject near edge: {', '.join(boundary_list)}."
        else:
            status = "RELIABLE"
            is_reliable = True
            warning = None

        # Available keypoints summary
        available_keys: List[str] = []
        if left_shoulder_ok or right_shoulder_ok:
            available_keys.append("SHOULDERS")
        if left_hip_ok or right_hip_ok:
            available_keys.append("HIPS")
        if lower_body_reliable >= 1:
            available_keys.append("LEGS")
        if len(landmarks) > 0 and landmarks[0].is_reliable:
            available_keys.append("HEAD")

        confidence_score = float(reliable_count / max(1, len(landmarks)))

        return PoseQuality(
            is_detected=True,
            is_reliable=is_reliable,
            is_partially_out_of_frame=is_partially_out,
            reliability_status=status,
            visibility_score=avg_vis,
            tracked_count=reliable_count,
            missing_key_landmarks=missing_keys,
            boundary_violations=boundary_list,
            warning_message=warning,
            confidence_score=confidence_score,
            available_keypoints=available_keys
        )

    def _draw_confidence_skeleton(
        self,
        canvas: np.ndarray,
        landmarks: List[LandmarkPoint],
        quality: PoseQuality
    ) -> None:
        """Renders confidence-aware skeletal joints, occlusions, and frame boundaries."""
        h, w = canvas.shape[:2]

        # Draw edge boundary warning stripes if subject is clipped
        if quality.is_partially_out_of_frame:
            for edge in quality.boundary_violations:
                if edge == "LEFT":
                    cv2.line(canvas, (2, 0), (2, h), (0, 140, 255), 5)
                elif edge == "RIGHT":
                    cv2.line(canvas, (w - 3, 0), (w - 3, h), (0, 140, 255), 5)
                elif edge == "TOP":
                    cv2.line(canvas, (0, 2), (w, 2), (0, 140, 255), 5)
                elif edge == "BOTTOM":
                    cv2.line(canvas, (0, h - 3), (w, h - 3), (0, 140, 255), 5)

        # Draw Skeletal Connection Lines
        for start_id, end_id in POSE_CONNECTIONS:
            p1 = landmarks[start_id]
            p2 = landmarks[end_id]

            if not (p1.is_in_bounds and p2.is_in_bounds):
                continue

            pt1 = (p1.x_px, p1.y_px)
            pt2 = (p2.x_px, p2.y_px)

            # Both endpoints reliable -> solid bright green
            if p1.is_reliable and p2.is_reliable:
                cv2.line(canvas, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
            # One endpoint occluded / low confidence -> amber warning line
            elif (p1.visibility >= 0.35 and p2.visibility >= 0.35):
                cv2.line(canvas, pt1, pt2, (0, 215, 255), 1, cv2.LINE_AA)
            # Severely occluded points -> omit line to prevent misleading connections

        # Draw Landmark Joint Dots
        for lm in landmarks:
            if not lm.is_in_bounds:
                continue

            pt = (lm.x_px, lm.y_px)

            if lm.is_reliable and lm.visibility >= 0.65:
                # High confidence -> green dot with white ring
                cv2.circle(canvas, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)
            elif lm.presence >= 0.35:
                # Moderate / occluded -> amber indicator
                cv2.circle(canvas, pt, 4, (0, 165, 255), -1, cv2.LINE_AA)
            else:
                # Very low confidence -> red indicator
                cv2.circle(canvas, pt, 3, (0, 0, 255), -1, cv2.LINE_AA)

        # Draw status alert badge if tracking is compromised
        if quality.warning_message is not None:
            text = f"! {quality.warning_message}"
            color = (0, 0, 255) if quality.reliability_status == "UNRELIABLE" else (0, 140, 255)
            # Badge background
            cv2.rectangle(canvas, (10, h - 38), (min(w - 10, 580), h - 10), (20, 20, 20), -1)
            cv2.rectangle(canvas, (10, h - 38), (min(w - 10, 580), h - 10), color, 1)
            cv2.putText(canvas, text[:60], (18, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

    def close(self) -> None:
        """Releases the MediaPipe PoseLandmarker model and memory resources."""
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception as exc:
                logger.debug("Error closing PoseLandmarker: %s", exc)
            self.detector = None

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
