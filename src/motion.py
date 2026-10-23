"""Traditional image processing module for motion detection and contour analysis.

Implements:
1. Grayscale conversion & Gaussian smoothing
2. Consecutive-frame temporal differencing
3. Binary thresholding & morphological operations (opening & dilation)
4. Motion region clustering, contour bounding boxes, and centroid kinematics
"""
from dataclasses import dataclass, field
import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MotionResult:
    """Structured result of traditional frame-differencing motion analysis."""
    has_motion: bool = False
    motion_score: float = 0.0           # Fraction of image experiencing motion [0.0, 1.0]
    motion_energy: float = 0.0          # Normalized sum of difference pixel intensities
    motion_mask: Optional[np.ndarray] = None  # 2D uint8 binary mask (255 for motion)
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) in pixels
    motion_centroid: Optional[Tuple[float, float]] = None     # (cx, cy) in normalized [0, 1] coords
    contours_count: int = 0
    contour_areas: List[float] = field(default_factory=list)


class MotionAnalyzer:
    """Consecutive-frame motion analyzer using classic computer vision techniques."""

    def __init__(
        self,
        blur_kernel_size: Tuple[int, int] = (21, 21),
        diff_threshold: int = 25,
        min_contour_area: float = 500.0,
        morph_kernel_size: Tuple[int, int] = (5, 5),
        motion_score_threshold: float = 0.008
    ):
        """Initializes motion analyzer parameters.

        Args:
            blur_kernel_size: Gaussian kernel size (must be positive odd integers).
            diff_threshold: Pixel intensity difference threshold for binary cutoff.
            min_contour_area: Minimum pixel area to consider a contour as valid motion.
            morph_kernel_size: Kernel dimensions for morphological opening/dilation.
            motion_score_threshold: Fraction of frame pixels needed to trigger has_motion.
        """
        self.blur_kernel_size = blur_kernel_size
        self.diff_threshold = diff_threshold
        self.min_contour_area = min_contour_area
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
        self.motion_score_threshold = motion_score_threshold

        self._previous_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Resets the temporal reference frame."""
        self._previous_gray = None

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Converts BGR frame to smoothed grayscale image.

        Stage 1: Grayscale conversion + Gaussian blur to attenuate high-frequency noise.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Input frame is empty or invalid.")

        if len(frame_bgr.shape) == 2:
            gray = frame_bgr
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)
        return blurred

    def compute_frame_difference(
        self,
        current_gray: np.ndarray,
        previous_gray: np.ndarray
    ) -> np.ndarray:
        """Calculates absolute consecutive-frame pixel difference.

        Stage 2: Absolute differencing between t and t-1.
        """
        return cv2.absdiff(current_gray, previous_gray)

    def threshold_and_morphology(self, diff_frame: np.ndarray) -> np.ndarray:
        """Generates clean binary motion mask via thresholding and morphological filtering.

        Stage 3: Binary thresholding + Morphological Opening (removes speckles)
                 + Morphological Dilation (bridges fractured motion blobs).
        """
        _, thresh = cv2.threshold(
            diff_frame,
            self.diff_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # Morphological OPEN: erosion followed by dilation (removes small isolated noise)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.morph_kernel, iterations=1)

        # Morphological DILATE: expands connected motion regions to unify body movement
        dilated = cv2.dilate(opened, self.morph_kernel, iterations=2)
        return dilated

    def analyze_motion_regions(
        self,
        motion_mask: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> Tuple[bool, float, float, Optional[Tuple[int, int, int, int]], Optional[Tuple[float, float]], int, List[float]]:
        """Extracts contours, bounding box, centroid, and motion metrics.

        Stage 4: Contour extraction & kinematic region properties.
        """
        h, w = frame_shape[:2]
        total_pixels = h * w

        # Find external contours
        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = []
        valid_areas = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_contour_area:
                valid_contours.append(cnt)
                valid_areas.append(float(area))

        total_motion_pixels = float(np.count_nonzero(motion_mask))
        motion_score = total_motion_pixels / float(total_pixels) if total_pixels > 0 else 0.0
        motion_energy = float(np.sum(motion_mask) / (255.0 * total_pixels)) if total_pixels > 0 else 0.0

        has_motion = motion_score >= self.motion_score_threshold and len(valid_contours) > 0

        bounding_box = None
        motion_centroid = None

        if valid_contours:
            # Aggregate all points to form combined motion envelope
            all_points = np.vstack(valid_contours)
            x, y, bw, bh = cv2.boundingRect(all_points)
            bounding_box = (int(x), int(y), int(bw), int(bh))

            # Compute centroid of total motion mask using spatial moments
            moments = cv2.moments(motion_mask)
            if moments["m00"] > 0:
                cx = float(moments["m10"] / moments["m00"]) / float(w)
                cy = float(moments["m01"] / moments["m00"]) / float(h)
                motion_centroid = (cx, cy)
            else:
                motion_centroid = ((x + bw / 2.0) / float(w), (y + bh / 2.0) / float(h))

        return has_motion, motion_score, motion_energy, bounding_box, motion_centroid, len(valid_contours), valid_areas

    def process_frame(self, frame_bgr: np.ndarray) -> MotionResult:
        """Executes full traditional image processing motion pipeline.

        Args:
            frame_bgr: Current video frame (OpenCV BGR).

        Returns:
            MotionResult dataclass with mask, score, and region geometry.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return MotionResult(has_motion=False)

        h, w = frame_bgr.shape[:2]
        current_gray = self.preprocess_frame(frame_bgr)

        # First frame reference initialization
        if self._previous_gray is None or self._previous_gray.shape != current_gray.shape:
            self._previous_gray = current_gray
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return MotionResult(
                has_motion=False,
                motion_score=0.0,
                motion_energy=0.0,
                motion_mask=empty_mask,
                bounding_box=None,
                motion_centroid=None,
                contours_count=0,
                contour_areas=[]
            )

        # 1. Consecutive-frame differencing
        diff = self.compute_frame_difference(current_gray, self._previous_gray)

        # 2. Thresholding + Morphological operations
        motion_mask = self.threshold_and_morphology(diff)

        # 3. Contour & region analysis
        has_motion, score, energy, bbox, centroid, count, areas = self.analyze_motion_regions(
            motion_mask, (h, w)
        )

        # Update reference frame for next consecutive step
        self._previous_gray = current_gray

        return MotionResult(
            has_motion=has_motion,
            motion_score=score,
            motion_energy=energy,
            motion_mask=motion_mask,
            bounding_box=bbox,
            motion_centroid=centroid,
            contours_count=count,
            contour_areas=areas
        )

    def draw_motion_overlay(
        self,
        canvas: np.ndarray,
        motion_result: MotionResult,
        color: Tuple[int, int, int] = (0, 165, 255),
        draw_box: bool = True,
        draw_centroid: bool = True
    ) -> np.ndarray:
        """Renders motion bounding box and centroid marker on display canvas."""
        if not motion_result.has_motion:
            return canvas

        h, w = canvas.shape[:2]

        if draw_box and motion_result.bounding_box:
            x, y, bw, bh = motion_result.bounding_box
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(
                canvas,
                f"MOTION: {motion_result.motion_score * 100:.1f}%",
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )

        if draw_centroid and motion_result.motion_centroid:
            cx_px = int(motion_result.motion_centroid[0] * w)
            cy_px = int(motion_result.motion_centroid[1] * h)
            cv2.circle(canvas, (cx_px, cy_px), 5, (0, 255, 255), -1)
            cv2.circle(canvas, (cx_px, cy_px), 7, (0, 0, 0), 1)

        return canvas
