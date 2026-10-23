"""Full-body fall detection state machine and multi-modal decision architecture.

Integrates traditional image processing motion cues with pose-derived kinematics.
Enforces clinical reliability safeguards:
1. Rejects static horizontal false positives (requires prior upright state + rapid downward descent).
2. Differentiates transient disturbances (POSSIBLE_FALL) from verified events (FALL_CONFIRMED).
3. Suspends full-body fall classification when camera view lacks lower-body landmarks (e.g. laptop webcams).
4. Employs refractory cooldowns to prevent duplicate alert storms.
5. Tracks recovery transitions back to NORMAL.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import time
from typing import List, Optional, Tuple

from src.features import PoseFeatures
from src.motion import MotionResult

logger = logging.getLogger(__name__)


class FallState(str, Enum):
    """Core states of the full-body fall-detection finite state machine."""
    NORMAL = "NORMAL"
    POSSIBLE_FALL = "POSSIBLE_FALL"
    FALL_CONFIRMED = "FALL_CONFIRMED"
    RECOVERY = "RECOVERY"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class FallDetectionEvent:
    """Historical record of a state transition or confirmed detection event."""
    event_id: int
    timestamp: float
    iso_time: str
    from_state: str
    to_state: str
    trigger_reason: str
    torso_angle_deg: float
    motion_score: float
    camera_view: str
    is_decision_suspended: bool = False
    video_timestamp: Optional[str] = None
    tracking_quality: Optional[str] = None


@dataclass
class FallDetectorStatus:
    """Instantaneous operational snapshot of the fall-detection system."""
    current_state: FallState = FallState.NORMAL
    previous_state: FallState = FallState.NORMAL
    is_fall_active: bool = False
    is_decision_suspended: bool = False
    suspension_reason: Optional[str] = None
    state_duration_sec: float = 0.0
    active_trigger_reason: Optional[str] = None
    alert_dispatched: bool = False
    camera_view_adequate: bool = False
    recent_events_count: int = 0
    confidence_score: float = 0.0
    active_signals: List[str] = field(default_factory=list)


class FallDetector:
    """Full-body fall detector combining posture kinematics with motion energy."""

    def __init__(
        self,
        confirmation_duration_sec: float = 1.0,
        recovery_duration_sec: float = 1.5,
        refractory_period_sec: float = 5.0,
        horizontal_angle_threshold: float = 50.0,
        downward_velocity_threshold: float = 0.25,
        angular_velocity_threshold: float = 45.0,
        min_motion_threshold: float = 0.005,
        vertical_drop_threshold: float = 0.12,
        require_full_body_view: bool = False
    ):
        """Initializes detector thresholds and timing parameters.

        Args:
            confirmation_duration_sec: Sustained horizontal/immobile time needed to confirm fall.
            recovery_duration_sec: Sustained upright posture needed to complete recovery.
            refractory_period_sec: Cooldown window suppressing duplicate alerts.
            horizontal_angle_threshold: Minimum torso angle (from vertical) considered fallen.
            downward_velocity_threshold: Minimum vertical descent velocity (norm_units/sec) for sudden collapse.
            angular_velocity_threshold: Minimum angular change rate (deg/sec) for sudden tilt.
            min_motion_threshold: Minimum motion score confirming active movement during collapse.
            vertical_drop_threshold: Minimum vertical drop (normalized body height) indicating a fall.
            require_full_body_view: If True, suspends fall confirmation when camera view lacks lower body.
        """
        self.confirmation_duration_sec = confirmation_duration_sec
        self.recovery_duration_sec = recovery_duration_sec
        self.refractory_period_sec = refractory_period_sec
        self.horizontal_angle_threshold = horizontal_angle_threshold
        self.downward_velocity_threshold = downward_velocity_threshold
        self.angular_velocity_threshold = angular_velocity_threshold
        self.min_motion_threshold = min_motion_threshold
        self.vertical_drop_threshold = vertical_drop_threshold
        self.require_full_body_view = require_full_body_view

        # State machine tracking
        self._state: FallState = FallState.NORMAL
        self._previous_state: FallState = FallState.NORMAL
        self._state_enter_timestamp: float = time.perf_counter()
        self._last_alert_timestamp: float = -999999.0
        self._event_counter: int = 0
        self._event_log: List[FallDetectionEvent] = []

        # Kinematic context buffers
        self._last_upright_timestamp: float = time.perf_counter()
        self._upright_baseline_y: Optional[float] = None
        self._pre_fall_baseline_y: Optional[float] = None
        self._last_motion_timestamp: float = -999.0
        self._possible_fall_trigger: Optional[str] = None
        self._is_first_frame: bool = True

    @property
    def current_state(self) -> FallState:
        """Returns the active state machine state."""
        return self._state

    @property
    def event_log(self) -> List[FallDetectionEvent]:
        """Returns chronological history of all transition events."""
        return list(self._event_log)

    def reset(self, initial_timestamp: Optional[float] = None, clear_events: bool = True) -> None:
        """Resets detector to initial NORMAL state and clears event timers.

        Args:
            initial_timestamp: Optional timestamp in seconds to align baseline timers.
            clear_events: Whether to wipe the historical transition log.
        """
        self._state = FallState.NORMAL
        self._previous_state = FallState.NORMAL
        init_ts = initial_timestamp if initial_timestamp is not None else time.perf_counter()
        self._state_enter_timestamp = init_ts
        self._last_alert_timestamp = -999999.0
        self._last_upright_timestamp = init_ts
        self._upright_baseline_y = None
        self._pre_fall_baseline_y = None
        self._last_motion_timestamp = -999.0
        self._possible_fall_trigger = None
        self._is_first_frame = (initial_timestamp is None)
        if clear_events:
            self._event_log.clear()
            self._event_counter = 0

    def _transition_to(
        self,
        new_state: FallState,
        reason: str,
        features: PoseFeatures,
        motion: MotionResult,
        now: float,
        is_suspended: bool = False
    ) -> None:
        """Executes state transition, resets state timer, and logs event record."""
        old_state = self._state
        self._previous_state = old_state
        self._state = new_state
        self._state_enter_timestamp = now

        if old_state == FallState.NORMAL and new_state == FallState.POSSIBLE_FALL:
            self._pre_fall_baseline_y = self._upright_baseline_y

        self._event_counter += 1
        iso_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        vid_time_str = f"{int(now // 60):02d}:{now % 60:05.2f}"

        event = FallDetectionEvent(
            event_id=self._event_counter,
            timestamp=now,
            iso_time=iso_str,
            from_state=old_state.value,
            to_state=new_state.value,
            trigger_reason=reason,
            torso_angle_deg=features.torso_angle_deg,
            motion_score=motion.motion_score,
            camera_view=features.camera_view_type,
            is_decision_suspended=is_suspended,
            video_timestamp=vid_time_str,
            tracking_quality=f"{features.camera_view_type} (vis: {features.average_visibility:.2f})"
        )
        self._event_log.append(event)
        logger.info("Fall Detector Transition: %s -> %s (Reason: %s, View: %s)",
                    old_state.value, new_state.value, reason, features.camera_view_type)

    def update(
        self,
        features: PoseFeatures,
        motion: MotionResult,
        timestamp: Optional[float] = None
    ) -> FallDetectorStatus:
        """Updates finite state machine using combined posture and motion features.

        Args:
            features: Current frame's PoseFeatures from PoseFeatureExtractor.
            motion: Current frame's MotionResult from MotionAnalyzer.
            timestamp: Frame timestamp in seconds.

        Returns:
            FallDetectorStatus describing state, view adequacy, and alerts.
        """
        now = timestamp if timestamp is not None else time.perf_counter()
        if self._is_first_frame:
            self._state_enter_timestamp = now
            self._last_upright_timestamp = now
            self._is_first_frame = False

        state_duration = now - self._state_enter_timestamp

        # 1. Camera View Adequacy Verification
        if self.require_full_body_view:
            camera_adequate = (features.camera_view_type == "FULL_BODY")
            is_suspended = not camera_adequate
            if is_suspended:
                if features.camera_view_type == "UPPER_BODY_ONLY":
                    suspension_msg = "SUSPENDED: Full-body landmarks required. Laptop view is limited to upper body."
                else:
                    suspension_msg = "SUSPENDED: Insufficient landmark visibility for full-body fall verification."
            else:
                suspension_msg = None
        else:
            camera_adequate = (features.camera_view_type != "INSUFFICIENT")
            if not camera_adequate and features.tracking_confidence < 0.20:
                is_suspended = True
                suspension_msg = "SUSPENDED: Insufficient landmark visibility."
            else:
                is_suspended = False
                suspension_msg = None

        # Track upright baseline continuity (only when genuinely upright at standing elevation)
        is_currently_upright = (
            features.torso_angle_reliable
            and features.torso_angle_deg < 28.0
            and features.camera_view_type != "INSUFFICIENT"
            and (self._pre_fall_baseline_y is None or features.body_center_y <= self._pre_fall_baseline_y + 0.08)
        )
        if is_currently_upright:
            self._last_upright_timestamp = now
            if self._state == FallState.NORMAL and features.body_center_y > 0.0:
                if self._upright_baseline_y is None:
                    self._upright_baseline_y = features.body_center_y
                else:
                    # Smoothly adapt baseline during continuous upright walking
                    self._upright_baseline_y = 0.95 * self._upright_baseline_y + 0.05 * features.body_center_y

        # Time elapsed since last known upright posture
        time_since_upright = now - self._last_upright_timestamp

        # Motion activity tracking (retains disturbance burst over recent collapse window)
        if motion.has_motion or motion.motion_score >= self.min_motion_threshold:
            self._last_motion_timestamp = now
        has_motion_burst = (
            (now - self._last_motion_timestamp < 2.0)
            or (motion.motion_score >= 0.003)
            or motion.has_motion
        )

        # Multi-signal kinematic triggers
        sig_rapid_descent = (
            features.vertical_velocity >= self.downward_velocity_threshold
            or features.max_recent_downward_velocity >= self.downward_velocity_threshold
        )
        ref_baseline = self._pre_fall_baseline_y if self._pre_fall_baseline_y is not None else self._upright_baseline_y
        drop_from_baseline = (
            (features.body_center_y - ref_baseline)
            if (ref_baseline is not None and features.body_center_y > 0)
            else 0.0
        )
        effective_drop = max(features.cumulative_vertical_drop, drop_from_baseline)
        sig_large_drop = (effective_drop >= self.vertical_drop_threshold)
        sig_rapid_tilt = abs(features.angular_velocity) >= self.angular_velocity_threshold
        sig_tilted_posture = (features.torso_angle_reliable and features.torso_angle_deg >= 30.0)
        sig_horizontal = (
            features.torso_angle_reliable
            and features.torso_angle_deg >= self.horizontal_angle_threshold
        ) or features.is_horizontal
        sig_bbox_flatten = (features.bbox_aspect_ratio >= 0.70)

        # Posture evaluation for sustained fallen condition
        is_fallen_posture = (
            sig_horizontal
            or (sig_tilted_posture and sig_large_drop)
            or (sig_bbox_flatten and sig_large_drop)
            or (sig_tilted_posture and drop_from_baseline > 0.10)
        )

        # Active signal telemetry
        active_signals: List[str] = []
        if sig_rapid_descent:
            active_signals.append(f"rapid_descent(vy={features.vertical_velocity:.2f})")
        if sig_large_drop:
            active_signals.append(f"vertical_drop(dy={effective_drop:.2f})")
        if sig_rapid_tilt:
            active_signals.append(f"rapid_tilt({features.angular_velocity:.1f}°/s)")
        if sig_tilted_posture or sig_horizontal:
            active_signals.append(f"angle({features.torso_angle_deg:.1f}°)")
        if sig_bbox_flatten:
            active_signals.append(f"aspect_ratio({features.bbox_aspect_ratio:.2f})")
        if has_motion_burst:
            active_signals.append(f"motion({motion.motion_score:.3f})")

        alert_dispatched = False

        # ======================================================================
        # State Machine Transitions
        # ======================================================================
        if self._state == FallState.NORMAL:
            # Safeguard 1: NEVER trigger a fall based solely on static horizontal posture.
            # Must detect dynamic collapse (rapid descent or large drop) from prior upright posture.
            had_recent_upright = (time_since_upright < 3.0) or (drop_from_baseline > 0.10)
            has_descent = (sig_rapid_descent or sig_large_drop)
            has_posture_shift = (
                sig_horizontal
                or sig_rapid_tilt
                or (sig_tilted_posture and sig_large_drop)
                or (sig_bbox_flatten and sig_tilted_posture)
                or (not features.torso_angle_reliable and sig_bbox_flatten)
            )

            if had_recent_upright and has_descent and has_posture_shift and has_motion_burst:
                trigger = f"Fall collapse initiated: {', '.join(active_signals)}"
                self._possible_fall_trigger = trigger
                self._transition_to(
                    FallState.POSSIBLE_FALL,
                    trigger,
                    features,
                    motion,
                    now,
                    is_suspended=is_suspended
                )

        elif self._state == FallState.POSSIBLE_FALL:
            # Check for immediate recovery / false alarm (e.g. tying shoe or intentional controlled bend)
            if is_currently_upright:
                self._pre_fall_baseline_y = None
                self._transition_to(
                    FallState.NORMAL,
                    "Immediate self-recovery: Upright posture restored before confirmation threshold.",
                    features,
                    motion,
                    now,
                    is_suspended=False
                )
            # Person obscured or moved out of frame following impact
            elif features.camera_view_type == "INSUFFICIENT":
                self._transition_to(
                    FallState.UNCERTAIN,
                    "Tracking lost or person obscured at frame boundary following fall impact.",
                    features,
                    motion,
                    now,
                    is_suspended=False
                )
            # Evaluate sustained fallen evidence
            elif is_fallen_posture:
                if state_duration >= self.confirmation_duration_sec:
                    if is_suspended:
                        logger.debug("Fall confirmation suppressed: %s", suspension_msg)
                    else:
                        can_alert = (now - self._last_alert_timestamp) >= self.refractory_period_sec
                        reason = (
                            f"Confirmed: Sustained fallen posture for {state_duration:.1f}s "
                            f"(drop={effective_drop:.2f}, angle={features.torso_angle_deg:.1f}°, "
                            f"AR={features.bbox_aspect_ratio:.2f})."
                        )
                        self._transition_to(
                            FallState.FALL_CONFIRMED,
                            reason,
                            features,
                            motion,
                            now,
                            is_suspended=False
                        )
                        if can_alert:
                            self._last_alert_timestamp = now
                            alert_dispatched = True
            elif state_duration >= (self.confirmation_duration_sec * 2.5):
                # Timeout without sustaining fallen posture -> return to NORMAL
                self._pre_fall_baseline_y = None
                self._transition_to(
                    FallState.NORMAL,
                    "Possible fall expired without sustained posture evidence.",
                    features,
                    motion,
                    now,
                    is_suspended=False
                )

        elif self._state == FallState.FALL_CONFIRMED:
            # Monitor for subject rising or beginning recovery
            if is_currently_upright:
                self._transition_to(
                    FallState.RECOVERY,
                    "Subject rising: Torso angle restored below 28° with restored upright elevation.",
                    features,
                    motion,
                    now,
                    is_suspended=False
                )
            elif features.camera_view_type == "INSUFFICIENT":
                self._transition_to(
                    FallState.UNCERTAIN,
                    "Tracking lost while fall was confirmed.",
                    features,
                    motion,
                    now,
                    is_suspended=False
                )

        elif self._state == FallState.RECOVERY:
            # Confirm stabilized recovery
            if is_currently_upright and state_duration >= self.recovery_duration_sec:
                self._pre_fall_baseline_y = None
                self._transition_to(
                    FallState.NORMAL,
                    f"Recovery verified: Stabilized upright posture maintained for {state_duration:.1f}s.",
                    features,
                    motion,
                    now,
                    is_suspended=False
                )
            # Relapse check: if subject collapses back during recovery
            elif is_fallen_posture and not is_suspended:
                self._transition_to(
                    FallState.FALL_CONFIRMED,
                    "Relapse: Subject returned to fallen posture during recovery phase.",
                    features,
                    motion,
                    now,
                    is_suspended=False
                )

        elif self._state == FallState.UNCERTAIN:
            # Check if person reappears upright
            if is_currently_upright:
                self._transition_to(
                    FallState.NORMAL,
                    "Tracking restored: Subject observed upright.",
                    features,
                    motion,
                    now,
                    is_suspended=False
                )
            # Check if person reappears in confirmed fall state
            elif is_fallen_posture and not is_suspended:
                self._transition_to(
                    FallState.FALL_CONFIRMED,
                    "Tracking restored: Subject observed in fallen posture.",
                    features,
                    motion,
                    now,
                    is_suspended=False
                )
            elif state_duration >= 5.0:
                self._transition_to(
                    FallState.NORMAL,
                    "Uncertain state timed out without tracking restoration.",
                    features,
                    motion,
                    now,
                    is_suspended=False
                )

        return FallDetectorStatus(
            current_state=self._state,
            previous_state=self._previous_state,
            is_fall_active=(self._state == FallState.FALL_CONFIRMED),
            is_decision_suspended=is_suspended,
            suspension_reason=suspension_msg,
            state_duration_sec=state_duration,
            active_trigger_reason=self._possible_fall_trigger if self._state != FallState.NORMAL else None,
            alert_dispatched=alert_dispatched,
            camera_view_adequate=camera_adequate,
            recent_events_count=len(self._event_log),
            confidence_score=features.tracking_confidence,
            active_signals=active_signals
        )
