"""Video Analysis module for processing uploaded video files.

Integrates:
- Validation of video formats, headers, and OpenCV decode capabilities
- Metadata extraction (resolution, duration, source FPS, frame count)
- Frame-by-frame processing reusing existing motion, pose, feature, and detector modules
- Temporal state resets between analysis runs
- Timestamp-indexed event extraction with keyframe capture
"""
from dataclasses import dataclass, field
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.detector import FallDetectionEvent, FallDetector, FallDetectorStatus, FallState
from src.features import PoseFeatureExtractor, PoseFeatures
from src.motion import MotionAnalyzer, MotionResult
from src.pose import PoseEstimator, PoseQuality
from src.utils import draw_hud

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def format_timestamp(seconds: float) -> str:
    """Formats seconds into MM:SS.ss (or HH:MM:SS.ss if >= 1 hour).

    Args:
        seconds: Time duration or offset in seconds.

    Returns:
        Formatted string, e.g. '01:23.45'.
    """
    if seconds < 0:
        seconds = 0.0

    total_secs = int(seconds)
    millis = int((seconds - total_secs) * 100)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:02d}"
    return f"{minutes:02d}:{secs:02d}.{millis:02d}"


@dataclass
class VideoMetadata:
    """Container for video container and codec properties."""
    file_path: str
    file_name: str
    file_size_bytes: int
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    duration_formatted: str
    codec: str = ""
    is_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class VideoEventRecord:
    """Structured record of a detection event occurring within a video."""
    event_id: int
    video_timestamp_sec: float
    video_timestamp_formatted: str
    frame_index: int
    from_state: str
    to_state: str
    trigger_reason: str
    torso_angle_deg: float
    motion_score: float
    camera_view: str
    is_decision_suspended: bool
    tracking_quality: str
    confidence_score: float = 0.0
    active_signals: List[str] = field(default_factory=list)
    keyframe_bgr: Optional[np.ndarray] = None


@dataclass
class VideoAnalysisResult:
    """Aggregated results and diagnostic telemetry from video processing."""
    metadata: VideoMetadata
    events: List[VideoEventRecord]
    total_frames_processed: int
    falls_confirmed_count: int
    possible_falls_count: int
    recoveries_count: int
    uncertain_count: int = 0
    overall_view_type: str = "FULL_BODY"
    average_processing_fps: float = 0.0
    keyframe_images: Dict[int, np.ndarray] = field(default_factory=dict)
    last_annotated_frame: Optional[np.ndarray] = None


def validate_video_file(file_path: Union[str, Path]) -> Tuple[bool, Optional[str], Optional[VideoMetadata]]:
    """Validates that a video file exists, is readable, and can be decoded by OpenCV.

    Args:
        file_path: Path to the target video file.

    Returns:
        Tuple of (is_valid, error_message, metadata_or_None).
    """
    path = Path(file_path)

    # 1. Check file existence
    if not path.is_file():
        return False, f"File does not exist: {path}", None

    # 2. Check extension
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return False, f"Unsupported video format '{ext}'. Supported formats: {supported_str}", None

    # 3. Check non-empty size
    try:
        size = path.stat().st_size
    except Exception as exc:
        return False, f"Unable to read file statistics: {exc}", None

    if size == 0:
        return False, "Uploaded video file is empty (0 bytes).", None

    # 4. Probe with OpenCV VideoCapture
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False, "OpenCV failed to open the video file. Format or codec may not be supported.", None

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # FourCC codec string
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        # Test frame read to confirm decodability
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            return False, "OpenCV was unable to decode any frames from this file. The file may be corrupt or truncated.", None

        # Fallback for FPS if missing or corrupted
        if fps <= 0.0 or math.isnan(fps):
            logger.warning("Video reports invalid FPS (%.2f), falling back to 30.0", fps)
            fps = 30.0

        if total_frames <= 0:
            total_frames = 1

        duration_sec = total_frames / fps
        duration_fmt = format_timestamp(duration_sec)

        metadata = VideoMetadata(
            file_path=str(path),
            file_name=path.name,
            file_size_bytes=size,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration_sec,
            duration_formatted=duration_fmt,
            codec=codec_str,
            is_valid=True
        )
        return True, None, metadata

    except Exception as exc:
        return False, f"Unexpected error reading video metadata: {exc}", None
    finally:
        cap.release()


class VideoAnalyzer:
    """Engine for frame-by-frame video fall detection analysis."""

    def __init__(
        self,
        confirmation_duration_sec: float = 1.0,
        horizontal_angle_threshold: float = 50.0,
        require_full_body_view: bool = False,
        inference_resolution: Optional[Tuple[int, int]] = (480, 360),
        draw_skeleton: bool = True,
        draw_motion: bool = True,
        show_hud: bool = True
    ):
        self.confirmation_duration_sec = confirmation_duration_sec
        self.horizontal_angle_threshold = horizontal_angle_threshold
        self.require_full_body_view = require_full_body_view
        self.inference_resolution = inference_resolution
        self.draw_skeleton = draw_skeleton
        self.draw_motion = draw_motion
        self.show_hud = show_hud

        # Shared modular components
        self.motion_analyzer = MotionAnalyzer()
        self.feature_extractor = PoseFeatureExtractor(
            horizontal_angle_threshold=self.horizontal_angle_threshold
        )
        self.fall_detector = FallDetector(
            confirmation_duration_sec=self.confirmation_duration_sec,
            horizontal_angle_threshold=self.horizontal_angle_threshold,
            require_full_body_view=self.require_full_body_view
        )

    def reset_state(self) -> None:
        """Resets all temporal modules before a new video analysis run."""
        self.motion_analyzer.reset()
        self.feature_extractor.reset()
        self.fall_detector.reset(initial_timestamp=0.0, clear_events=True)

    def analyze_video(
        self,
        video_path: Union[str, Path],
        progress_callback: Optional[Callable[[float, int, int, np.ndarray, FallDetectorStatus, VideoMetadata], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None
    ) -> VideoAnalysisResult:
        """Processes a video file frame-by-frame and produces comprehensive fall analysis.

        Args:
            video_path: Path to local video file.
            progress_callback: Optional callback receiving (progress_pct, current_frame, total_frames, annotated_frame, fall_status, metadata).
            stop_check: Optional callable returning True if processing should be cancelled.

        Returns:
            VideoAnalysisResult containing all telemetry and detected events.
        """
        is_valid, err, metadata = validate_video_file(video_path)
        if not is_valid or metadata is None:
            raise ValueError(f"Cannot analyze video: {err}")

        # Reset temporal state machine and buffers
        self.reset_state()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        pose_estimator = None
        try:
            pose_estimator = PoseEstimator(
                num_poses=1,
                inference_resolution=self.inference_resolution
            )

            fps = metadata.fps
            total_frames = metadata.total_frames
            width = metadata.width
            height = metadata.height
            actual_res = (width, height)

            processed_count = 0
            start_wall_time = time.perf_counter()
            events: List[VideoEventRecord] = []
            keyframe_images: Dict[int, np.ndarray] = {}
            last_event_count = 0
            view_counts: Dict[str, int] = {"FULL_BODY": 0, "UPPER_BODY_ONLY": 0, "INSUFFICIENT": 0}
            last_annotated: Optional[np.ndarray] = None

            while True:
                if stop_check is not None and stop_check():
                    logger.info("Video analysis cancelled by user request at frame %d", processed_count)
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx = processed_count
                processed_count += 1

                # Calculate source video timestamp
                video_time_sec = frame_idx / fps
                video_time_ms = int(video_time_sec * 1000)

                # 1. Traditional motion analysis
                motion_result = self.motion_analyzer.process_frame(frame)

                # 2. MediaPipe Pose estimation with video timestamp
                pose_result = pose_estimator.process_frame(
                    frame_bgr=frame,
                    draw_skeleton=self.draw_skeleton,
                    timestamp_ms=video_time_ms
                )
                annotated_bgr = pose_result.annotated_frame
                pose_detected = pose_result.pose_detected
                quality = pose_result.pose_quality

                # 3. Pose landmark extraction
                primary_landmarks = []
                if pose_detected and pose_result.detection_result:
                    extracted = pose_estimator.extract_landmarks(pose_result.detection_result, actual_res)
                    if extracted:
                        primary_landmarks = extracted[0]

                # 4. Kinematic feature extraction
                features = self.feature_extractor.extract_features(
                    landmarks=primary_landmarks,
                    quality=quality,
                    timestamp=video_time_sec
                )

                # Track view distribution
                view_counts[features.camera_view_type] = view_counts.get(features.camera_view_type, 0) + 1

                # 5. Multi-state fall detection
                status = self.fall_detector.update(
                    features=features,
                    motion=motion_result,
                    timestamp=video_time_sec
                )

                # 6. Overlays and HUD
                if self.draw_motion and motion_result.has_motion:
                    annotated_bgr = self.motion_analyzer.draw_motion_overlay(annotated_bgr, motion_result)

                if self.show_hud:
                    annotated_bgr = draw_hud(
                        frame=annotated_bgr,
                        fps=fps,
                        camera_status=f"Video ({format_timestamp(video_time_sec)})",
                        pose_detected=pose_detected,
                        resolution=actual_res,
                        pose_quality=quality,
                        fall_status=status
                    )

                last_annotated = annotated_bgr

                # Check if a new state transition event occurred
                detector_events = self.fall_detector.event_log
                if len(detector_events) > last_event_count:
                    # New event logged
                    latest_evt: FallDetectionEvent = detector_events[-1]
                    keyframe_thumb = annotated_bgr.copy()
                    event_rec = VideoEventRecord(
                        event_id=latest_evt.event_id,
                        video_timestamp_sec=latest_evt.timestamp,
                        video_timestamp_formatted=format_timestamp(latest_evt.timestamp),
                        frame_index=frame_idx,
                        from_state=latest_evt.from_state,
                        to_state=latest_evt.to_state,
                        trigger_reason=latest_evt.trigger_reason,
                        torso_angle_deg=latest_evt.torso_angle_deg,
                        motion_score=latest_evt.motion_score,
                        camera_view=latest_evt.camera_view,
                        is_decision_suspended=latest_evt.is_decision_suspended,
                        tracking_quality=f"{features.camera_view_type} (vis: {features.average_visibility:.2f})",
                        confidence_score=status.confidence_score,
                        active_signals=list(status.active_signals),
                        keyframe_bgr=keyframe_thumb
                    )
                    events.append(event_rec)
                    keyframe_images[latest_evt.event_id] = keyframe_thumb
                    last_event_count = len(detector_events)

                # Progress notification callback
                if progress_callback is not None:
                    progress_ratio = min(1.0, processed_count / max(1, total_frames))
                    progress_callback(
                        progress_ratio,
                        processed_count,
                        total_frames,
                        annotated_bgr,
                        status,
                        metadata
                    )

            total_elapsed = max(0.001, time.perf_counter() - start_wall_time)
            avg_processing_fps = processed_count / total_elapsed

            # Determine overall camera view classification
            if view_counts["FULL_BODY"] >= (processed_count * 0.60):
                overall_view = "FULL_BODY"
            elif view_counts["UPPER_BODY_ONLY"] >= (processed_count * 0.40):
                overall_view = "UPPER_BODY_ONLY"
            elif view_counts.get("PARTIAL", 0) >= (processed_count * 0.40):
                overall_view = "PARTIAL"
            else:
                overall_view = "INSUFFICIENT"

            confirmed_count = sum(1 for e in events if e.to_state == FallState.FALL_CONFIRMED.value)
            possible_count = sum(1 for e in events if e.to_state == FallState.POSSIBLE_FALL.value)
            recovery_count = sum(1 for e in events if e.to_state == FallState.RECOVERY.value)
            uncertain_count = sum(1 for e in events if e.to_state == FallState.UNCERTAIN.value)

            return VideoAnalysisResult(
                metadata=metadata,
                events=events,
                total_frames_processed=processed_count,
                falls_confirmed_count=confirmed_count,
                possible_falls_count=possible_count,
                recoveries_count=recovery_count,
                uncertain_count=uncertain_count,
                overall_view_type=overall_view,
                average_processing_fps=avg_processing_fps,
                keyframe_images=keyframe_images,
                last_annotated_frame=last_annotated
            )

        finally:
            cap.release()
            if pose_estimator is not None:
                pose_estimator.close()
