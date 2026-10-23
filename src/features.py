"""Pose-derived feature extraction and temporal kinematics module.

Extracts geometric, postural, and temporal features from MediaPipe landmarks:
1. Mid-shoulder, mid-hip, and lower-body anchors
2. Torso orientation angle from vertical (0° = upright, 90° = horizontal)
3. Body bounding-box dimensions and aspect ratio (W / H)
4. Full-body vs Upper-body camera view classification
5. Temporal vertical velocities and orientation change rates over a rolling time window
"""
from collections import deque
from dataclasses import dataclass, field
import math
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.pose import LandmarkPoint, PoseQuality, KEY_TORSO_LANDMARKS, KEY_LOWER_BODY_LANDMARKS


@dataclass
class PoseFeatures:
    """Instantaneous geometric and postural features extracted from a single frame."""
    timestamp: float = 0.0

    # Key body anchor coordinates (normalized [0, 1] or None if occluded)
    mid_shoulder: Optional[Tuple[float, float]] = None
    mid_hip: Optional[Tuple[float, float]] = None
    left_shoulder: Optional[Tuple[float, float]] = None
    right_shoulder: Optional[Tuple[float, float]] = None
    left_hip: Optional[Tuple[float, float]] = None
    right_hip: Optional[Tuple[float, float]] = None
    left_knee: Optional[Tuple[float, float]] = None
    right_knee: Optional[Tuple[float, float]] = None
    left_ankle: Optional[Tuple[float, float]] = None
    right_ankle: Optional[Tuple[float, float]] = None

    # Torso & posture geometry
    torso_center: Optional[Tuple[float, float]] = None  # (cx, cy)
    torso_angle_deg: float = 0.0                        # Degrees from vertical (0° = upright, 90° = horizontal)
    torso_angle_reliable: bool = False
    is_horizontal: bool = False                         # True if torso_angle >= threshold or wide aspect ratio
    body_center_y: float = 0.0                          # Normalized vertical coordinate of body center

    # Body bounding-box geometry (normalized)
    bbox_norm: Optional[Tuple[float, float, float, float]] = None  # (xmin, ymin, xmax, ymax)
    bbox_aspect_ratio: float = 0.0                     # width / height (typically < 0.6 upright, > 0.8 lying)
    bbox_height: float = 0.0

    # Camera view adequacy & visibility
    is_torso_visible: bool = False
    is_lower_body_visible: bool = False
    is_full_body_visible: bool = False
    camera_view_type: str = "INSUFFICIENT"              # "FULL_BODY", "UPPER_BODY_ONLY", "PARTIAL", "INSUFFICIENT"
    reliable_landmarks_count: int = 0
    average_visibility: float = 0.0
    tracking_confidence: float = 0.0
    available_keypoints: List[str] = field(default_factory=list)
    posture_summary: str = "Unknown"

    # Temporal kinematics (computed across feature buffer)
    vertical_velocity: float = 0.0                      # dy/dt in normalized units/sec (+ downward)
    angular_velocity: float = 0.0                       # d(angle)/dt in deg/sec
    max_recent_downward_velocity: float = 0.0           # Peak downward velocity in recent window
    cumulative_vertical_drop: float = 0.0               # Cumulative drop from recent upright baseline (normalized)


class PoseFeatureExtractor:
    """Extracts instantaneous posture features and temporal kinematics over rolling buffer."""

    def __init__(
        self,
        history_window_size: int = 60,
        horizontal_angle_threshold: float = 50.0,
        min_visibility_threshold: float = 0.5
    ):
        """Initializes feature extractor.

        Args:
            history_window_size: Number of past feature frames to retain in buffer (~2-2.5s).
            horizontal_angle_threshold: Torso angle from vertical to be considered horizontal.
            min_visibility_threshold: Confidence threshold for a landmark to be considered reliable.
        """
        self.history_window_size = history_window_size
        self.horizontal_angle_threshold = horizontal_angle_threshold
        self.min_visibility_threshold = min_visibility_threshold

        self._history: deque[PoseFeatures] = deque(maxlen=self.history_window_size)

    def reset(self) -> None:
        """Clears the temporal feature history."""
        self._history.clear()

    @property
    def history(self) -> List[PoseFeatures]:
        """Returns the list of recent feature frames."""
        return list(self._history)

    def extract_features(
        self,
        landmarks: List[LandmarkPoint],
        quality: Optional[PoseQuality] = None,
        timestamp: Optional[float] = None
    ) -> PoseFeatures:
        """Extracts complete geometric, postural, and temporal features from landmark set.

        Args:
            landmarks: LandmarkPoint instances from PoseEstimator.
            quality: PoseQuality diagnostic object.
            timestamp: Frame timestamp (defaults to time.perf_counter()).

        Returns:
            PoseFeatures dataclass.
        """
        now = timestamp if timestamp is not None else time.perf_counter()

        if not landmarks:
            empty_feat = PoseFeatures(timestamp=now, camera_view_type="INSUFFICIENT")
            self._history.append(empty_feat)
            return empty_feat

        # 1. Filter reliable points (relaxed threshold to 0.35 to retain points in realistic scenarios)
        effective_vis_thresh = min(self.min_visibility_threshold, 0.35)
        reliable_map: Dict[int, LandmarkPoint] = {}
        for lm in landmarks:
            if lm.is_reliable and lm.visibility >= effective_vis_thresh:
                reliable_map[lm.landmark_id] = lm

        # 2. Extract key anatomical anchors
        l_sh = (reliable_map[11].x_norm, reliable_map[11].y_norm) if 11 in reliable_map else None
        r_sh = (reliable_map[12].x_norm, reliable_map[12].y_norm) if 12 in reliable_map else None
        l_hip = (reliable_map[23].x_norm, reliable_map[23].y_norm) if 23 in reliable_map else None
        r_hip = (reliable_map[24].x_norm, reliable_map[24].y_norm) if 24 in reliable_map else None

        l_knee = (reliable_map[25].x_norm, reliable_map[25].y_norm) if 25 in reliable_map else None
        r_knee = (reliable_map[26].x_norm, reliable_map[26].y_norm) if 26 in reliable_map else None
        l_ank = (reliable_map[27].x_norm, reliable_map[27].y_norm) if 27 in reliable_map else None
        r_ank = (reliable_map[28].x_norm, reliable_map[28].y_norm) if 28 in reliable_map else None

        # Compute midpoint anchors
        mid_sh = self._compute_midpoint(l_sh, r_sh)
        mid_hip = self._compute_midpoint(l_hip, r_hip)

        # 3. Torso center & orientation angle
        torso_center = None
        torso_angle_deg = 0.0
        torso_angle_reliable = False

        if mid_sh is not None and mid_hip is not None:
            torso_center = ((mid_sh[0] + mid_hip[0]) / 2.0, (mid_sh[1] + mid_hip[1]) / 2.0)
            torso_angle_deg, torso_angle_reliable = self._compute_torso_angle(mid_sh, mid_hip)
        elif mid_sh is not None:
            torso_center = mid_sh
        elif mid_hip is not None:
            torso_center = mid_hip

        is_horizontal = torso_angle_reliable and (torso_angle_deg >= self.horizontal_angle_threshold)

        # 4. Body bounding box and aspect ratio
        bbox_norm, aspect_ratio, bbox_height = self._compute_bounding_box(reliable_map)

        # Body center Y (normalized)
        if torso_center is not None:
            body_center_y = torso_center[1]
        elif mid_hip is not None:
            body_center_y = mid_hip[1]
        elif mid_sh is not None:
            body_center_y = mid_sh[1]
        elif bbox_norm is not None:
            body_center_y = (bbox_norm[1] + bbox_norm[3]) / 2.0
        else:
            body_center_y = 0.0

        # Posture summary
        if torso_angle_reliable:
            if torso_angle_deg < 25.0:
                posture_summary = "Upright"
            elif torso_angle_deg < 50.0:
                posture_summary = "Tilted / Leaning"
            else:
                posture_summary = "Horizontal / Fallen"
        elif aspect_ratio > 0.85:
            posture_summary = "Wide / Horizontal"
        else:
            posture_summary = "Uncertain"

        # 5. Camera View Classification & Visibility
        is_torso_visible = (l_sh is not None or r_sh is not None) and (l_hip is not None or r_hip is not None)
        is_lower_body_visible = (l_knee is not None or r_knee is not None or l_ank is not None or r_ank is not None)
        is_full_body_visible = is_torso_visible and is_lower_body_visible

        if is_full_body_visible:
            camera_view = "FULL_BODY"
        elif (l_sh is not None or r_sh is not None):
            camera_view = "UPPER_BODY_ONLY"
        elif len(reliable_map) >= 4:
            camera_view = "PARTIAL"
        else:
            camera_view = "INSUFFICIENT"

        avg_vis = float(np.mean([lm.visibility for lm in landmarks])) if landmarks else 0.0
        conf_score = quality.confidence_score if quality is not None else avg_vis

        # Build list of available keypoint names
        avail_kps = [
            name for idx, name in [
                (11, "left_shoulder"), (12, "right_shoulder"),
                (23, "left_hip"), (24, "right_hip"),
                (25, "left_knee"), (26, "right_knee"),
                (27, "left_ankle"), (28, "right_ankle")
            ] if idx in reliable_map
        ]

        # Construct current feature snapshot
        features = PoseFeatures(
            timestamp=now,
            mid_shoulder=mid_sh,
            mid_hip=mid_hip,
            left_shoulder=l_sh,
            right_shoulder=r_sh,
            left_hip=l_hip,
            right_hip=r_hip,
            left_knee=l_knee,
            right_knee=r_knee,
            left_ankle=l_ank,
            right_ankle=r_ank,
            torso_center=torso_center,
            torso_angle_deg=torso_angle_deg,
            torso_angle_reliable=torso_angle_reliable,
            is_horizontal=is_horizontal,
            body_center_y=body_center_y,
            bbox_norm=bbox_norm,
            bbox_aspect_ratio=aspect_ratio,
            bbox_height=bbox_height,
            is_torso_visible=is_torso_visible,
            is_lower_body_visible=is_lower_body_visible,
            is_full_body_visible=is_full_body_visible,
            camera_view_type=camera_view,
            reliable_landmarks_count=len(reliable_map),
            average_visibility=avg_vis,
            tracking_confidence=conf_score,
            available_keypoints=avail_kps,
            posture_summary=posture_summary
        )

        # 6. Compute temporal kinematics against historical buffer
        self._compute_temporal_kinematics(features)

        # Store in rolling buffer
        self._history.append(features)

        return features

    @staticmethod
    def _compute_midpoint(
        pt1: Optional[Tuple[float, float]],
        pt2: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """Calculates midpoint of two points, falling back to whichever point exists."""
        if pt1 is not None and pt2 is not None:
            return ((pt1[0] + pt2[0]) / 2.0, (pt1[1] + pt2[1]) / 2.0)
        elif pt1 is not None:
            return pt1
        elif pt2 is not None:
            return pt2
        return None

    @staticmethod
    def _compute_torso_angle(
        mid_shoulder: Tuple[float, float],
        mid_hip: Tuple[float, float]
    ) -> Tuple[float, bool]:
        """Calculates the acute angle of the torso vector from the upward vertical axis.

        In image coordinates, y increases downward.
        When standing upright: shoulder is above hip -> shoulder.y < hip.y.
        Vector hip -> shoulder: dx = sh.x - hip.x, dy = sh.y - hip.y (dy is negative).
        Upward vertical reference: (0, -1).

        Angle = arccos( dot(v, up) / |v| ) * 180 / pi.
        0° = perfectly upright vertical
        90° = perfectly horizontal
        """
        dx = mid_shoulder[0] - mid_hip[0]
        dy = mid_shoulder[1] - mid_hip[1]
        length = math.hypot(dx, dy)

        if length < 1e-4:
            return 0.0, False

        # Unit vector from hip to shoulder
        ux = dx / length
        uy = dy / length

        # Upward vertical vector is (0, -1) -> dot product is -uy
        # When upright: uy is approx -1.0 -> dot is +1.0 -> acos(1) = 0°
        # When lying flat: uy is approx 0.0 -> dot is 0.0 -> acos(0) = 90°
        # When inverted (upside down): uy is +1.0 -> dot is -1.0 -> acos(-1) = 180°
        dot = -uy
        dot = max(-1.0, min(1.0, dot))
        angle_rad = math.acos(dot)
        angle_deg = math.degrees(angle_rad)

        return float(angle_deg), True

    @staticmethod
    def _compute_bounding_box(
        reliable_map: Dict[int, LandmarkPoint]
    ) -> Tuple[Optional[Tuple[float, float, float, float]], float, float]:
        """Calculates enclosing bounding box and aspect ratio from reliable points."""
        if not reliable_map:
            return None, 0.0, 0.0

        xs = [lm.x_norm for lm in reliable_map.values()]
        ys = [lm.y_norm for lm in reliable_map.values()]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        w = max(0.001, xmax - xmin)
        h = max(0.001, ymax - ymin)
        aspect_ratio = w / h

        return (xmin, ymin, xmax, ymax), aspect_ratio, h

    def _compute_temporal_kinematics(self, current: PoseFeatures) -> None:
        """Calculates vertical velocity, cumulative drop, and angular velocity using the history buffer."""
        if not self._history:
            return

        # Find reference frame approximately 0.15s - 0.3s ago for noise-immune derivative
        target_dt = 0.20
        ref_feat = None

        for prev in reversed(self._history):
            dt = current.timestamp - prev.timestamp
            if dt >= target_dt:
                ref_feat = prev
                break

        if ref_feat is None:
            ref_feat = self._history[0]
            dt = current.timestamp - ref_feat.timestamp

        if dt > 0.01:
            # Vertical velocity: dy/dt (+ is moving downwards toward floor)
            if current.body_center_y > 0.0 and ref_feat.body_center_y > 0.0:
                dy = current.body_center_y - ref_feat.body_center_y
                current.vertical_velocity = float(dy / dt)
            elif current.torso_center is not None and ref_feat.torso_center is not None:
                dy = current.torso_center[1] - ref_feat.torso_center[1]
                current.vertical_velocity = float(dy / dt)

            # Angular velocity: change in torso angle per second
            if current.torso_angle_reliable and ref_feat.torso_angle_reliable:
                d_theta = current.torso_angle_deg - ref_feat.torso_angle_deg
                current.angular_velocity = float(d_theta / dt)

        # Track maximum downward velocity in the current history window
        recent_vy = [f.vertical_velocity for f in self._history if f.vertical_velocity is not None]
        recent_vy.append(current.vertical_velocity)
        current.max_recent_downward_velocity = float(max(recent_vy)) if recent_vy else 0.0

        # Cumulative vertical drop from recent upright baseline
        upright_baseline_candidates = [
            f.body_center_y for f in self._history
            if f.body_center_y > 0.0 and (not f.torso_angle_reliable or f.torso_angle_deg < 35.0)
        ]
        if upright_baseline_candidates:
            baseline_y = min(upright_baseline_candidates)
        else:
            all_past_y = [f.body_center_y for f in self._history if f.body_center_y > 0.0]
            baseline_y = min(all_past_y) if all_past_y else current.body_center_y

        if current.body_center_y > 0.0 and baseline_y > 0.0:
            current.cumulative_vertical_drop = float(max(0.0, current.body_center_y - baseline_y))
