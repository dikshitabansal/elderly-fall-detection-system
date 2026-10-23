"""Unit tests for pose-derived features, posture geometry, and temporal kinematics."""
import unittest
from src.pose import LandmarkPoint, PoseQuality
from src.features import PoseFeatureExtractor, PoseFeatures


def build_synthetic_landmarks(
    sh_y: float = 0.2,
    hip_y: float = 0.5,
    sh_x: float = 0.5,
    hip_x: float = 0.5,
    has_lower_body: bool = True,
    torso_visible: bool = True
) -> list:
    """Helper to generate 33 LandmarkPoints with controlled geometry and visibility."""
    pts = []
    for i in range(33):
        x, y = 0.5, 0.5
        vis = 0.95

        if i == 11:  # Left shoulder
            x, y = sh_x - 0.05, sh_y
            vis = 0.95 if torso_visible else 0.1
        elif i == 12:  # Right shoulder
            x, y = sh_x + 0.05, sh_y
            vis = 0.95 if torso_visible else 0.1
        elif i == 23:  # Left hip
            x, y = hip_x - 0.04, hip_y
            vis = 0.95 if torso_visible else 0.1
        elif i == 24:  # Right hip
            x, y = hip_x + 0.04, hip_y
            vis = 0.95 if torso_visible else 0.1
        elif i in (25, 26):  # Knees
            x, y = 0.5, 0.7 if has_lower_body else 0.5
            vis = 0.95 if has_lower_body else 0.1
        elif i in (27, 28):  # Ankles
            x, y = 0.5, 0.9 if has_lower_body else 0.5
            vis = 0.95 if has_lower_body else 0.1

        is_rel = (vis >= 0.5)
        pts.append(
            LandmarkPoint(
                landmark_id=i,
                name=f"LM_{i}",
                x_norm=x,
                y_norm=y,
                z_norm=0.0,
                x_px=int(x * 640),
                y_px=int(y * 480),
                visibility=vis,
                presence=vis,
                is_visible=is_rel,
                is_present=is_rel,
                is_in_bounds=True,
                is_reliable=is_rel
            )
        )
    return pts


class TestPoseFeatureExtractor(unittest.TestCase):
    """Test suite for PoseFeatureExtractor."""

    def setUp(self):
        self.extractor = PoseFeatureExtractor(
            history_window_size=15,
            horizontal_angle_threshold=60.0
        )

    def test_upright_posture_features(self):
        """Upright posture should yield near-zero torso angle and vertical aspect ratio."""
        pts = build_synthetic_landmarks(sh_y=0.2, hip_y=0.5, has_lower_body=True)
        feat = self.extractor.extract_features(pts, timestamp=0.0)

        self.assertTrue(feat.torso_angle_reliable)
        # Shoulders vertically above hips -> 0° angle
        self.assertAlmostEqual(feat.torso_angle_deg, 0.0, delta=2.0)
        self.assertFalse(feat.is_horizontal)
        # Vertical posture bounding box: height > width -> aspect ratio < 0.6
        self.assertLess(feat.bbox_aspect_ratio, 0.6)
        self.assertTrue(feat.is_full_body_visible)
        self.assertEqual(feat.camera_view_type, "FULL_BODY")

    def test_horizontal_posture_features(self):
        """Horizontal lying posture should yield ~90° torso angle and is_horizontal=True."""
        # Shoulders and hips aligned horizontally at y=0.8
        pts = build_synthetic_landmarks(sh_y=0.8, hip_y=0.8, sh_x=0.3, hip_x=0.7, has_lower_body=True)
        feat = self.extractor.extract_features(pts, timestamp=0.0)

        self.assertTrue(feat.torso_angle_reliable)
        self.assertAlmostEqual(feat.torso_angle_deg, 90.0, delta=2.0)
        self.assertTrue(feat.is_horizontal)
        # Horizontal bounding box: width > height -> aspect ratio > 1.0
        self.assertGreater(feat.bbox_aspect_ratio, 1.0)

    def test_upper_body_only_camera_view(self):
        """Laptop camera capturing only head/torso must be flagged as UPPER_BODY_ONLY."""
        # Lower body landmarks occluded/not present
        pts = build_synthetic_landmarks(sh_y=0.2, hip_y=0.5, has_lower_body=False)
        feat = self.extractor.extract_features(pts, timestamp=0.0)

        self.assertTrue(feat.is_torso_visible)
        self.assertFalse(feat.is_lower_body_visible)
        self.assertFalse(feat.is_full_body_visible)
        self.assertEqual(feat.camera_view_type, "UPPER_BODY_ONLY")

    def test_occluded_torso_landmarks(self):
        """When torso landmarks are occluded, torso angle should be marked unreliable."""
        pts = build_synthetic_landmarks(torso_visible=False)
        feat = self.extractor.extract_features(pts, timestamp=0.0)

        self.assertFalse(feat.torso_angle_reliable)
        self.assertFalse(feat.is_torso_visible)

    def test_temporal_kinematics(self):
        """Rapid downward movement must produce positive vertical velocity."""
        # Frame 1: Upright standing at y=0.2
        pts1 = build_synthetic_landmarks(sh_y=0.2, hip_y=0.5)
        self.extractor.extract_features(pts1, timestamp=0.0)

        # Frame 2: 0.25s later, fallen to y=0.7 (downward plunge)
        pts2 = build_synthetic_landmarks(sh_y=0.7, hip_y=0.7, sh_x=0.3, hip_x=0.7)
        feat2 = self.extractor.extract_features(pts2, timestamp=0.25)

        # Downward vertical displacement rate should be positive
        self.assertGreater(feat2.vertical_velocity, 0.5)
        # Angular rate of change should be positive (tilted from 0° to 90°)
        self.assertGreater(feat2.angular_velocity, 50.0)

    def test_empty_landmarks_handling(self):
        """Empty landmark list returns safe default features without errors."""
        feat = self.extractor.extract_features([], timestamp=0.0)
        self.assertFalse(feat.torso_angle_reliable)
        self.assertEqual(feat.camera_view_type, "INSUFFICIENT")

    def test_partial_landmarks_extraction(self):
        """Subset of landmarks correctly extracts available keypoints and classifies view."""
        # Only shoulders and hips present
        pts = build_synthetic_landmarks(sh_y=0.25, hip_y=0.55, has_lower_body=False, torso_visible=True)
        feat = self.extractor.extract_features(pts, timestamp=0.0)

        self.assertTrue(feat.torso_angle_reliable)
        self.assertIn("left_shoulder", feat.available_keypoints)
        self.assertIn("right_shoulder", feat.available_keypoints)
        self.assertIn("left_hip", feat.available_keypoints)
        self.assertIn("right_hip", feat.available_keypoints)
        self.assertNotIn("left_knee", feat.available_keypoints)
        self.assertEqual(feat.camera_view_type, "UPPER_BODY_ONLY")
        self.assertEqual(feat.posture_summary, "Upright")

    def test_cumulative_vertical_drop_calculation(self):
        """Elevated posture followed by lower body center produces cumulative vertical drop."""
        pts_high = build_synthetic_landmarks(sh_y=0.20, hip_y=0.45)
        self.extractor.extract_features(pts_high, timestamp=0.0)

        pts_mid = build_synthetic_landmarks(sh_y=0.35, hip_y=0.60)
        self.extractor.extract_features(pts_mid, timestamp=0.5)

        pts_low = build_synthetic_landmarks(sh_y=0.50, hip_y=0.75)
        feat_low = self.extractor.extract_features(pts_low, timestamp=1.0)

        self.assertGreater(feat_low.cumulative_vertical_drop, 0.15)


if __name__ == "__main__":
    unittest.main()
