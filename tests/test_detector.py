"""Unit tests for full-body fall detection state machine and reliability safeguards."""
import unittest
from src.features import PoseFeatures
from src.motion import MotionResult
from src.detector import FallDetector, FallDetectorStatus, FallState


class TestFallDetector(unittest.TestCase):
    """Test suite for FallDetector state transitions and clinical safeguards."""

    def setUp(self):
        self.detector = FallDetector(
            confirmation_duration_sec=0.5,
            recovery_duration_sec=0.5,
            refractory_period_sec=3.0,
            horizontal_angle_threshold=55.0,
            downward_velocity_threshold=0.30,
            angular_velocity_threshold=50.0,
            min_motion_threshold=0.01,
            require_full_body_view=True
        )

        self.motion_none = MotionResult(has_motion=False, motion_score=0.0)
        self.motion_active = MotionResult(has_motion=True, motion_score=0.04)

        self.feat_upright = PoseFeatures(
            torso_angle_deg=5.0,
            torso_angle_reliable=True,
            is_horizontal=False,
            camera_view_type="FULL_BODY",
            vertical_velocity=0.0,
            angular_velocity=0.0
        )

        self.feat_collapse = PoseFeatures(
            torso_angle_deg=75.0,
            torso_angle_reliable=True,
            is_horizontal=True,
            camera_view_type="FULL_BODY",
            vertical_velocity=0.55,
            angular_velocity=85.0
        )

    def test_full_state_transition_lifecycle(self):
        """Verify complete state flow: NORMAL -> POSSIBLE_FALL -> FALL_CONFIRMED -> RECOVERY -> NORMAL."""
        # Step 1: Upright baseline in NORMAL
        st = self.detector.update(self.feat_upright, self.motion_none, timestamp=0.0)
        self.assertEqual(st.current_state, FallState.NORMAL)

        # Step 2: Rapid collapse triggers POSSIBLE_FALL
        st = self.detector.update(self.feat_collapse, self.motion_active, timestamp=0.5)
        self.assertEqual(st.current_state, FallState.POSSIBLE_FALL)
        self.assertFalse(st.is_fall_active)

        # Step 3: Sustained horizontal posture for > confirmation_duration_sec -> FALL_CONFIRMED
        st = self.detector.update(self.feat_collapse, self.motion_none, timestamp=1.1)
        self.assertEqual(st.current_state, FallState.FALL_CONFIRMED)
        self.assertTrue(st.is_fall_active)
        self.assertTrue(st.alert_dispatched)

        # Step 4: Subject begins rising -> RECOVERY
        st = self.detector.update(self.feat_upright, self.motion_active, timestamp=1.8)
        self.assertEqual(st.current_state, FallState.RECOVERY)

        # Step 5: Stabilized upright posture for > recovery_duration_sec -> NORMAL
        st = self.detector.update(self.feat_upright, self.motion_none, timestamp=2.4)
        self.assertEqual(st.current_state, FallState.NORMAL)
        self.assertFalse(st.is_fall_active)

    def test_safeguard_static_horizontal_no_fall(self):
        """Safeguard 1: Horizontal posture alone must NEVER trigger a fall without collapse kinematics."""
        feat_lying = PoseFeatures(
            torso_angle_deg=85.0,
            torso_angle_reliable=True,
            is_horizontal=True,
            camera_view_type="FULL_BODY",
            vertical_velocity=0.0,
            angular_velocity=0.0
        )

        # Update over several consecutive seconds
        for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
            st = self.detector.update(feat_lying, self.motion_none, timestamp=t)
            self.assertEqual(st.current_state, FallState.NORMAL)
            self.assertFalse(st.is_fall_active)

    def test_safeguard_upper_body_only_suspends_decision(self):
        """Safeguard 2: Upper-body only camera view must suspend full-body fall confirmation."""
        # Upright baseline
        self.detector.update(self.feat_upright, self.motion_none, timestamp=0.0)

        # Collapse occurring with upper-body only view (laptop webcam)
        feat_upper_collapse = PoseFeatures(
            torso_angle_deg=75.0,
            torso_angle_reliable=True,
            is_horizontal=True,
            camera_view_type="UPPER_BODY_ONLY",
            vertical_velocity=0.55,
            angular_velocity=85.0
        )

        # Triggers POSSIBLE_FALL in suspended mode
        st = self.detector.update(feat_upper_collapse, self.motion_active, timestamp=0.5)
        self.assertEqual(st.current_state, FallState.POSSIBLE_FALL)
        self.assertTrue(st.is_decision_suspended)
        self.assertIn("upper body", st.suspension_reason.lower())

        # Sustained horizontal: MUST NOT confirm fall
        st_sustained = self.detector.update(feat_upper_collapse, self.motion_none, timestamp=1.2)
        self.assertEqual(st_sustained.current_state, FallState.POSSIBLE_FALL)
        self.assertFalse(st_sustained.is_fall_active)
        self.assertTrue(st_sustained.is_decision_suspended)

    def test_safeguard_immediate_recovery_cancels_possible_fall(self):
        """Immediate return to upright posture cancels POSSIBLE_FALL without confirmation."""
        # Upright -> Collapse
        self.detector.update(self.feat_upright, self.motion_none, timestamp=0.0)
        self.detector.update(self.feat_collapse, self.motion_active, timestamp=0.3)
        self.assertEqual(self.detector.current_state, FallState.POSSIBLE_FALL)

        # Person immediately stands back up within 0.2s (e.g. tying shoe or picking object)
        st = self.detector.update(self.feat_upright, self.motion_active, timestamp=0.5)
        self.assertEqual(st.current_state, FallState.NORMAL)
        self.assertFalse(st.is_fall_active)

    def test_safeguard_duplicate_alert_suppression(self):
        """Refractory period suppresses duplicate alarms during the same incident."""
        self.detector.update(self.feat_upright, self.motion_none, timestamp=0.0)
        self.detector.update(self.feat_collapse, self.motion_active, timestamp=0.5)

        # First confirmation dispatches alert
        st1 = self.detector.update(self.feat_collapse, self.motion_none, timestamp=1.2)
        self.assertEqual(st1.current_state, FallState.FALL_CONFIRMED)
        self.assertTrue(st1.alert_dispatched)

        # Subsequent frame while still fallen within refractory window (3.0s) MUST NOT alert again
        st2 = self.detector.update(self.feat_collapse, self.motion_none, timestamp=2.0)
        self.assertEqual(st2.current_state, FallState.FALL_CONFIRMED)
        self.assertFalse(st2.alert_dispatched)

    def test_ordinary_movement_stays_normal(self):
        """Gradual sitting or controlled walking remains in NORMAL state."""
        # Controlled transition: moderate angle, low downward velocity
        feat_sitting = PoseFeatures(
            torso_angle_deg=25.0,
            torso_angle_reliable=True,
            is_horizontal=False,
            camera_view_type="FULL_BODY",
            vertical_velocity=0.08,
            angular_velocity=12.0
        )
        st = self.detector.update(feat_sitting, self.motion_active, timestamp=1.0)
        self.assertEqual(st.current_state, FallState.NORMAL)

    def test_stair_fall_non_horizontal_angle(self):
        """Falls on stairs/uneven surfaces where posture is slumped (non-horizontal ~40°) and vertical drop is large."""
        detector = FallDetector(
            confirmation_duration_sec=0.5,
            require_full_body_view=False
        )

        feat_stair_upright = PoseFeatures(
            torso_angle_deg=6.0,
            torso_angle_reliable=True,
            body_center_y=0.32,
            camera_view_type="FULL_BODY",
            vertical_velocity=0.02
        )
        detector.update(feat_stair_upright, self.motion_active, timestamp=0.0)

        # Tumble occurs: drop of 0.18, angle tilts to 42°, aspect ratio widens
        feat_stair_tumble = PoseFeatures(
            torso_angle_deg=42.0,
            torso_angle_reliable=True,
            body_center_y=0.52,
            cumulative_vertical_drop=0.18,
            bbox_aspect_ratio=1.35,
            camera_view_type="FULL_BODY",
            vertical_velocity=0.30,
            angular_velocity=55.0
        )
        st1 = detector.update(feat_stair_tumble, self.motion_active, timestamp=0.4)
        self.assertEqual(st1.current_state, FallState.POSSIBLE_FALL)

        # Sustained at base of stairs for > 0.5s -> FALL_CONFIRMED
        feat_stair_collapsed = PoseFeatures(
            torso_angle_deg=40.0,
            torso_angle_reliable=True,
            body_center_y=0.54,
            cumulative_vertical_drop=0.18,
            bbox_aspect_ratio=1.30,
            camera_view_type="FULL_BODY",
            vertical_velocity=0.01,
            angular_velocity=0.0
        )
        st2 = detector.update(feat_stair_collapsed, self.motion_none, timestamp=1.0)
        self.assertEqual(st2.current_state, FallState.FALL_CONFIRMED)
        self.assertTrue(st2.is_fall_active)

    def test_occluded_disappearance_triggers_uncertain(self):
        """When landmarks are lost or person disappears outside frame after collapse, transition to UNCERTAIN."""
        detector = FallDetector(confirmation_duration_sec=0.5, require_full_body_view=False)
        detector.update(self.feat_upright, self.motion_none, timestamp=0.0)

        # Dynamic collapse into frame boundary
        detector.update(self.feat_collapse, self.motion_active, timestamp=0.3)
        self.assertEqual(detector.current_state, FallState.POSSIBLE_FALL)

        # Person disappears / tracking lost
        feat_lost = PoseFeatures(
            camera_view_type="INSUFFICIENT",
            tracking_confidence=0.0,
            torso_angle_reliable=False
        )
        st = detector.update(feat_lost, self.motion_none, timestamp=0.6)
        self.assertEqual(st.current_state, FallState.UNCERTAIN)

    def test_normal_walking_down_stairs_no_false_alarm(self):
        """Walking down stairs with upright posture and gradual descent must not trigger a fall."""
        detector = FallDetector(confirmation_duration_sec=0.5, require_full_body_view=False)

        for step_idx in range(6):
            t = step_idx * 0.4
            y_pos = 0.20 + (step_idx * 0.03)  # Gradual walking descent
            feat_step = PoseFeatures(
                torso_angle_deg=8.0,
                torso_angle_reliable=True,
                body_center_y=y_pos,
                camera_view_type="FULL_BODY",
                vertical_velocity=0.05,
                bbox_aspect_ratio=0.50
            )
            st = detector.update(feat_step, self.motion_active, timestamp=t)
            self.assertEqual(st.current_state, FallState.NORMAL)

    def test_partial_landmarks_fall_detection(self):
        """Fall detection succeeds with partial landmarks when knees/ankles are occluded."""
        detector = FallDetector(confirmation_duration_sec=0.5, require_full_body_view=False)
        detector.update(self.feat_upright, self.motion_none, timestamp=0.0)

        feat_partial_fall = PoseFeatures(
            torso_angle_deg=65.0,
            torso_angle_reliable=True,
            is_horizontal=True,
            camera_view_type="PARTIAL",
            vertical_velocity=0.45,
            angular_velocity=60.0
        )
        st1 = detector.update(feat_partial_fall, self.motion_active, timestamp=0.4)
        self.assertEqual(st1.current_state, FallState.POSSIBLE_FALL)

        st2 = detector.update(feat_partial_fall, self.motion_none, timestamp=1.0)
        self.assertEqual(st2.current_state, FallState.FALL_CONFIRMED)

    def test_reset(self):
        """Reset clears state back to NORMAL."""
        self.detector.update(self.feat_upright, self.motion_none, timestamp=0.0)
        self.detector.update(self.feat_collapse, self.motion_active, timestamp=0.5)
        self.assertEqual(self.detector.current_state, FallState.POSSIBLE_FALL)

        self.detector.reset()
        self.assertEqual(self.detector.current_state, FallState.NORMAL)


if __name__ == "__main__":
    unittest.main()
