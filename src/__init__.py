"""Elderly Fall Detection and Monitoring System package."""

__version__ = "0.2.0"

from src.camera import CameraCapture
from src.detector import FallDetector, FallDetectorStatus, FallState, FallDetectionEvent
from src.features import PoseFeatureExtractor, PoseFeatures
from src.motion import MotionAnalyzer, MotionResult
from src.pose import PoseEstimator, PoseQuality
from src.utils import FPSCalculator, draw_hud
from src.video_analysis import (
    VideoAnalyzer,
    VideoMetadata,
    VideoAnalysisResult,
    VideoEventRecord,
    validate_video_file,
    format_timestamp
)

__all__ = [
    "CameraCapture",
    "FallDetector",
    "FallDetectorStatus",
    "FallState",
    "FallDetectionEvent",
    "PoseFeatureExtractor",
    "PoseFeatures",
    "MotionAnalyzer",
    "MotionResult",
    "PoseEstimator",
    "PoseQuality",
    "FPSCalculator",
    "draw_hud",
    "VideoAnalyzer",
    "VideoMetadata",
    "VideoAnalysisResult",
    "VideoEventRecord",
    "validate_video_file",
    "format_timestamp"
]
