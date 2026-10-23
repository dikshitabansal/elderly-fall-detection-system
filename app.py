"""Real-Time Elderly Fall Detection and Monitoring System.

Dual-Mode Dashboard:
1. MODE 1 — LIVE MONITORING:
   - Live webcam acquisition & threaded frame buffering
   - Traditional image processing motion analysis (frame differencing & contours)
   - MediaPipe Pose tracking with occlusion diagnostics & boundary detection
   - Full-body fall detection state machine with clinical safeguards
   - Timestamped detection event logging & dual-feed visualization

2. MODE 2 — VIDEO ANALYSIS:
   - Upload and validation of prerecorded MP4/AVI videos
   - Metadata extraction (resolution, duration, source FPS, frame count)
   - Frame-by-frame analysis reusing motion, pose, feature, and detector modules
   - Source-video timestamp indexing and keyframe capture
   - Caching to avoid re-analysis on ordinary Streamlit reruns
"""
import logging
import os
from pathlib import Path
import time
from typing import List, Optional
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.camera import CameraCapture
from src.detector import FallDetector, FallDetectorStatus, FallState
from src.features import PoseFeatureExtractor, PoseFeatures
from src.motion import MotionAnalyzer, MotionResult
from src.pose import PoseEstimator
from src.utils import FPSCalculator, draw_hud
from src.video_analysis import (
    VideoAnalyzer,
    VideoAnalysisResult,
    VideoMetadata,
    format_timestamp,
    validate_video_file
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fall_monitor_app")

# Page Configuration
st.set_page_config(
    page_title="Elderly Fall Detection & Monitoring System",
    page_icon="🧍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
    """
    <style>
    .state-normal {
        background-color: #1b4d2e;
        color: #79f29d;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .state-possible {
        background-color: #78350f;
        color: #fde68a;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .state-confirmed {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
        animation: pulse 1s infinite alternate;
    }
    .state-recovery {
        background-color: #1e3a8a;
        color: #bfdbfe;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .view-badge {
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    @keyframes pulse {
        0% { opacity: 0.8; }
        100% { opacity: 1.0; transform: scale(1.02); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Session State Initialization
if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False


# ==========================================
# Sidebar: System Mode & Controls
# ==========================================
st.sidebar.title("🧍 Elderly Fall Monitor")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Operating Mode",
    options=["🎥 Live Monitoring", "📁 Video Analysis"],
    index=0,
    help="Switch between live webcam monitoring and uploaded video file analysis"
)
st.sidebar.markdown("---")


# ==============================================================================
# MODE 1: LIVE MONITORING
# ==============================================================================
if app_mode == "🎥 Live Monitoring":
    st.sidebar.subheader("🎮 Live Controls")

    col_start, col_stop = st.sidebar.columns(2)
    start_clicked = col_start.button(
        "▶ Start",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_streaming,
        help="Activate live camera feed, motion analysis, and fall detection"
    )
    stop_clicked = col_stop.button(
        "⏹ Stop",
        type="secondary",
        use_container_width=True,
        disabled=not st.session_state.is_streaming,
        help="Stop camera feed and release hardware cleanly"
    )

    if start_clicked:
        st.session_state.is_streaming = True
        st.rerun()

    if stop_clicked:
        st.session_state.is_streaming = False
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("📷 Camera Settings")

    device_index = st.sidebar.number_input(
        "Camera Index",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        disabled=st.session_state.is_streaming,
        help="Physical index of the webcam device (default: 0)"
    )

    res_selection = st.sidebar.selectbox(
        "Frame Resolution",
        options=["640 x 480 (Recommended)", "1280 x 720 (HD)", "Default Camera Resolution"],
        index=0,
        disabled=st.session_state.is_streaming
    )

    target_w, target_h = None, None
    if "640 x 480" in res_selection:
        target_w, target_h = 640, 480
    elif "1280 x 720" in res_selection:
        target_w, target_h = 1280, 720

    st.sidebar.markdown("---")
    st.sidebar.subheader("🦴 Pose & Vision Settings")

    infer_scale_opt = st.sidebar.selectbox(
        "Inference Resolution",
        options=["480 x 360 (Balanced - Recommended)", "Native Capture (Full Resolution)", "320 x 240 (High Performance)"],
        index=0,
        disabled=st.session_state.is_streaming,
        help="Neural inference scale. Full visual resolution is preserved on display."
    )

    infer_res = None
    if "480 x 360" in infer_scale_opt:
        infer_res = (480, 360)
    elif "320 x 240" in infer_scale_opt:
        infer_res = (320, 240)

    draw_skeleton = st.sidebar.checkbox(
        "Draw Skeletal Overlay",
        value=True,
        help="Draw body landmarks and confidence connections on live feed"
    )

    draw_motion = st.sidebar.checkbox(
        "Draw Motion Bounding Box",
        value=True,
        help="Draw bounding envelope around frame-differencing motion regions"
    )

    show_hud = st.sidebar.checkbox(
        "Show On-Screen HUD",
        value=True,
        help="Render diagnostic metrics, tracking status, and emergency alert banners"
    )

    show_motion_mask = st.sidebar.checkbox(
        "Show Motion Mask Feed",
        value=True,
        help="Display the binary frame-differencing motion mask alongside the main stream"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Fall Detection Safeguards")

    require_full_body = st.sidebar.checkbox(
        "Require Full-Body View",
        value=True,
        help="Guards against false alarms: Suspends fall confirmation when camera view is restricted to upper body (head/torso only)."
    )

    confirm_sec = st.sidebar.slider(
        "Confirmation Window (s)",
        min_value=0.5,
        max_value=3.0,
        value=1.2,
        step=0.1,
        disabled=st.session_state.is_streaming,
        help="Sustained horizontal posture duration required to confirm a fall"
    )

    horizontal_deg = st.sidebar.slider(
        "Horizontal Angle (°)",
        min_value=40.0,
        max_value=80.0,
        value=55.0,
        step=5.0,
        disabled=st.session_state.is_streaming,
        help="Torso angle from vertical to be considered horizontal"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("System: Real-Time Elderly Fall Detection & Monitoring")
    st.sidebar.caption("Architecture: Multi-Modal (Pose Kinematics + Traditional Image Processing)")

    # Main Dashboard UI
    st.title("🧍 Real-Time Elderly Fall Detection and Monitoring System")
    st.caption(
        "Multi-Modal Vision System combining Traditional Image Processing (Frame Differencing, Morphological Contours) "
        "with MediaPipe Pose Kinematics and Multi-State Clinical Safeguards."
    )

    status_container = st.empty()

    # Dashboard Metrics Row (5 columns)
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        state_metric = st.empty()
        if not st.session_state.is_streaming:
            state_metric.metric("Detection State", "STANDBY")
    with m2:
        view_metric = st.empty()
        if not st.session_state.is_streaming:
            view_metric.metric("Camera View", "N/A")
    with m3:
        angle_metric = st.empty()
        if not st.session_state.is_streaming:
            angle_metric.metric("Torso Angle", "0.0°")
    with m4:
        motion_metric = st.empty()
        if not st.session_state.is_streaming:
            motion_metric.metric("Motion Energy", "0.0%")
    with m5:
        fps_metric = st.empty()
        if not st.session_state.is_streaming:
            fps_metric.metric("Measured FPS", "0.0")

    st.markdown("---")

    col_video, col_mask = st.columns([1.5, 1.0]) if show_motion_mask else (st.container(), None)
    with col_video:
        st.subheader("🎥 Live Monitoring Stream")
        video_container = st.empty()

    if show_motion_mask and col_mask is not None:
        with col_mask:
            st.subheader("⚡ Motion Differencing Mask")
            mask_container = st.empty()

    if not st.session_state.is_streaming:
        status_container.info(
            "ℹ️ **System Standby**: Click **▶ Start** in the sidebar to activate the live webcam stream, "
            "traditional motion differencing, and full-body fall detection safeguards."
        )
        video_container.info("Webcam feed is currently inactive. Press **▶ Start** to begin.")
        if show_motion_mask and col_mask is not None:
            mask_container.info("Motion difference mask will appear here when camera is streaming.")

    # Detection Event History Log Section
    st.markdown("---")
    st.subheader("📋 Detection Event History Log")
    event_table_placeholder = st.empty()
    event_table_placeholder.caption("No events logged yet. Active state transitions will appear here with timestamps.")

    # Operational Live Streaming Loop
    if st.session_state.is_streaming:
        status_container.success("🟢 **Live Stream Active**: Pipeline operational.")

        camera = CameraCapture(
            device_index=int(device_index),
            width=target_w,
            height=target_h,
            threaded=True
        )

        if not camera.open():
            error_msg = camera.get_status().error_message or f"Unable to open camera index {device_index}."
            status_container.error(f"❌ **Camera Unavailable**: {error_msg}")
            state_metric.metric("Detection State", "UNAVAILABLE")
            st.session_state.is_streaming = False
            camera.release()
        else:
            pose_estimator = None
            motion_analyzer = MotionAnalyzer()
            feature_extractor = PoseFeatureExtractor(horizontal_angle_threshold=float(horizontal_deg))
            fall_detector = FallDetector(
                confirmation_duration_sec=float(confirm_sec),
                horizontal_angle_threshold=float(horizontal_deg),
                require_full_body_view=require_full_body
            )
            fps_calculator = FPSCalculator(window_size=30)

            try:
                pose_estimator = PoseEstimator(
                    num_poses=1,
                    inference_resolution=infer_res
                )
                actual_res = camera.get_resolution()

                last_frame_id = -1
                last_metric_update = 0.0

                while st.session_state.is_streaming:
                    success, frame, last_frame_id = camera.read_latest_frame(
                        last_frame_id=last_frame_id,
                        timeout=0.08
                    )

                    if not success or frame is None:
                        if not camera.is_opened():
                            status_container.error("⚠️ **Camera Disconnected**: Stream stopped.")
                            st.session_state.is_streaming = False
                            break
                        time.sleep(0.001)
                        continue

                    current_fps = fps_calculator.tick()
                    now_ts = time.perf_counter()

                    # Stage 1: Traditional Image Processing
                    motion_result = motion_analyzer.process_frame(frame)

                    # Stage 2: MediaPipe Pose Estimation
                    pose_result = pose_estimator.process_frame(
                        frame_bgr=frame,
                        draw_skeleton=draw_skeleton
                    )
                    annotated_bgr = pose_result.annotated_frame
                    pose_detected = pose_result.pose_detected
                    quality = pose_result.pose_quality

                    # Stage 3: Pose-Derived Features
                    primary_landmarks = []
                    if pose_detected and pose_result.detection_result:
                        extracted = pose_estimator.extract_landmarks(pose_result.detection_result, actual_res)
                        if extracted:
                            primary_landmarks = extracted[0]

                    features = feature_extractor.extract_features(
                        landmarks=primary_landmarks,
                        quality=quality,
                        timestamp=now_ts
                    )

                    # Stage 4: Multi-State Fall Detector
                    fall_status = fall_detector.update(
                        features=features,
                        motion=motion_result,
                        timestamp=now_ts
                    )

                    # Stage 5: Overlays & Rendering
                    if draw_motion and motion_result.has_motion:
                        annotated_bgr = motion_analyzer.draw_motion_overlay(annotated_bgr, motion_result)

                    if show_hud:
                        annotated_bgr = draw_hud(
                            frame=annotated_bgr,
                            fps=current_fps,
                            camera_status=f"Live (Index {device_index})",
                            pose_detected=pose_detected,
                            resolution=actual_res,
                            pose_quality=quality,
                            fall_status=fall_status
                        )

                    # Streamlit Video Rendering (Fast Direct JPEG)
                    ret_enc, jpeg_buf = cv2.imencode('.jpg', annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret_enc:
                        video_container.image(jpeg_buf.tobytes(), output_format="JPEG", use_container_width=True)

                    if show_motion_mask and col_mask is not None and motion_result.motion_mask is not None:
                        mask_colored = cv2.applyColorMap(motion_result.motion_mask, cv2.COLORMAP_JET)
                        ret_mask, mask_buf = cv2.imencode('.jpg', mask_colored, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ret_mask:
                            mask_container.image(mask_buf.tobytes(), output_format="JPEG", use_container_width=True)

                    # Throttled UI & Status Updates (every 250ms)
                    if now_ts - last_metric_update >= 0.25:
                        last_metric_update = now_ts
                        fps_metric.metric("Measured FPS", f"{current_fps:.1f}")
                        angle_metric.metric("Torso Angle", f"{features.torso_angle_deg:.1f}°")
                        motion_metric.metric("Motion Energy", f"{motion_result.motion_score * 100:.1f}%")

                        state_val = fall_status.current_state.value
                        if state_val == "FALL_CONFIRMED":
                            state_metric.metric("Detection State", "🔴 FALL CONFIRMED", delta="EMERGENCY", delta_color="inverse")
                        elif state_val == "POSSIBLE_FALL":
                            state_metric.metric("Detection State", "🟡 POSSIBLE FALL", delta="Verifying...", delta_color="inverse")
                        elif state_val == "RECOVERY":
                            state_metric.metric("Detection State", "🔵 RECOVERY", delta="Rising Up")
                        elif state_val == "UNCERTAIN":
                            state_metric.metric("Detection State", "🟠 UNCERTAIN", delta="Person Obscured", delta_color="inverse")
                        else:
                            state_metric.metric("Detection State", "🟢 NORMAL", delta="Monitoring")

                        if features.camera_view_type == "FULL_BODY":
                            view_metric.metric("Camera View", "Full Body", delta="CCTV Ready")
                        elif features.camera_view_type == "UPPER_BODY_ONLY":
                            view_metric.metric("Camera View", "Upper Body Only", delta="Suspended", delta_color="inverse")
                        else:
                            view_metric.metric("Camera View", "Insufficient", delta="Occluded", delta_color="inverse")

                        if fall_status.current_state == FallState.FALL_CONFIRMED:
                            status_container.error(
                                "🚨 **EMERGENCY: FALL CONFIRMED** — Floor immobility verified following rapid collapse. "
                                f"(Torso Angle: {features.torso_angle_deg:.1f}°, Duration: {fall_status.state_duration_sec:.1f}s)"
                            )
                        elif fall_status.current_state == FallState.POSSIBLE_FALL:
                            if fall_status.is_decision_suspended:
                                status_container.warning(
                                    "⚠️ **Possible Fall Detected (Suspended)**: Rapid descent observed, but **laptop webcam captures only upper body**. "
                                    "Full-body CCTV view required to confirm floor impact."
                                )
                            else:
                                status_container.warning(
                                    "⚠️ **Possible Fall Detected**: Rapid collapse detected. Verifying sustained horizontal posture..."
                                )
                        elif fall_status.is_decision_suspended:
                            status_container.info(
                                "ℹ️ **Upper-Body Webcam View**: Head, shoulders, and torso are tracked. "
                                "Full-body fall classification is **guarded/suspended** until a full-body CCTV camera view is provided."
                            )
                        elif quality.warning_message:
                            status_container.warning(f"⚠️ **Tracking Advisory**: {quality.warning_message}")
                        else:
                            status_container.success("🟢 **Monitoring Normal**: Subject is stable and posture is upright.")

                        # Update Detection Event History Log Table
                        events = fall_detector.event_log
                        if events:
                            table_data = [
                                {
                                    "Time": e.iso_time,
                                    "From State": e.from_state,
                                    "To State": e.to_state,
                                    "Torso Angle": f"{e.torso_angle_deg:.1f}°",
                                    "Motion": f"{e.motion_score * 100:.1f}%",
                                    "Camera View": e.camera_view,
                                    "Suspended": "YES" if e.is_decision_suspended else "NO",
                                    "Reason": e.trigger_reason[:65]
                                }
                                for e in reversed(events[-12:])
                            ]
                            event_table_placeholder.dataframe(
                                pd.DataFrame(table_data),
                                use_container_width=True,
                                hide_index=True
                            )

                    time.sleep(0.001)

            except Exception as exc:
                logger.error("Exception in streaming loop: %s", exc, exc_info=True)
                status_container.error(f"Error during streaming: {exc}")
                st.session_state.is_streaming = False

            finally:
                camera.release()
                if pose_estimator is not None:
                    pose_estimator.close()
                state_metric.metric("Detection State", "STANDBY")
                fps_metric.metric("Measured FPS", "0.0")


# ==============================================================================
# MODE 2: VIDEO ANALYSIS
# ==============================================================================
else:
    # Ensure live stream is halted when switching to video analysis
    if st.session_state.is_streaming:
        st.session_state.is_streaming = False

    st.sidebar.subheader("⚙️ Analysis Settings")

    infer_scale_opt = st.sidebar.selectbox(
        "Inference Resolution",
        options=["480 x 360 (Balanced - Recommended)", "Native Capture (Full Resolution)", "320 x 240 (High Performance)"],
        index=0,
        help="Downscale resolution for faster pose inference."
    )

    infer_res = None
    if "480 x 360" in infer_scale_opt:
        infer_res = (480, 360)
    elif "320 x 240" in infer_scale_opt:
        infer_res = (320, 240)

    draw_skeleton = st.sidebar.checkbox("Draw Skeletal Overlay", value=True)
    draw_motion = st.sidebar.checkbox("Draw Motion Bounding Box", value=True)
    show_hud = st.sidebar.checkbox("Show Diagnostic HUD Overlay", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Fall Detection Safeguards")

    require_full_body = st.sidebar.checkbox(
        "Require Full-Body View",
        value=False,
        help="If enabled, suspends fall confirmation when video only captures upper body. Keep disabled for realistic videos with partial occlusion."
    )

    confirm_sec = st.sidebar.slider(
        "Confirmation Window (s)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Sustained fallen/horizontal posture duration required to confirm a fall"
    )

    horizontal_deg = st.sidebar.slider(
        "Horizontal Angle (°)",
        min_value=35.0,
        max_value=80.0,
        value=50.0,
        step=5.0,
        help="Torso angle from vertical to be considered horizontal"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Mode: Pre-recorded Video Analysis")
    st.sidebar.caption("Formats: MP4, AVI")

    # Main UI Header
    st.title("📁 Uploaded Video Fall Detection & Posture Analysis")
    st.caption(
        "Perform automated, frame-by-frame posture kinematics and motion analysis on recorded video footage. "
        "Every frame is evaluated using the exact same multi-modal pipeline and clinical reliability safeguards as live monitoring."
    )

    # File Uploader
    uploaded_file = st.file_uploader(
        "Choose a video file to analyze (MP4 or AVI)",
        type=["mp4", "avi"],
        help="Select a video file from your computer. The file will remain stored strictly locally."
    )

    if uploaded_file is None:
        st.info("ℹ️ **Upload a video file** above to inspect video metadata and run the automated fall detection pipeline.")
    else:
        # Save uploaded file locally in temp_uploads
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        local_video_path = temp_dir / uploaded_file.name

        with open(local_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Validate video file with OpenCV
        is_valid, err_msg, metadata = validate_video_file(local_video_path)

        if not is_valid or metadata is None:
            st.error(f"❌ **Invalid or Unreadable Video File**: {err_msg}")
        else:
            # Display Video Metadata Cards
            st.markdown("### 📊 Video File Properties")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Resolution", f"{metadata.width} × {metadata.height} px")
            c2.metric("Source FPS", f"{metadata.fps:.2f} FPS")
            c3.metric("Duration", f"{metadata.duration_formatted} ({metadata.duration_sec:.1f}s)")
            c4.metric("Total Frames", f"{metadata.total_frames:,} frames")

            st.markdown("---")

            # Unique cache signature for avoiding repeated re-analysis on reruns
            cache_key = (
                f"{uploaded_file.name}_{uploaded_file.size}_{require_full_body}_"
                f"{confirm_sec}_{horizontal_deg}_{infer_scale_opt}_{draw_skeleton}_{draw_motion}"
            )

            col_btn, col_info = st.columns([1.2, 3.8])
            analyze_clicked = col_btn.button("🔍 Analyze Video", type="primary", use_container_width=True)

            # Check if cached results exist
            cached_result: Optional[VideoAnalysisResult] = st.session_state.get(f"res_{cache_key}")

            preview_placeholder = st.empty()
            progress_bar = st.empty()
            status_text = st.empty()

            if analyze_clicked:
                analyzer = VideoAnalyzer(
                    confirmation_duration_sec=float(confirm_sec),
                    horizontal_angle_threshold=float(horizontal_deg),
                    require_full_body_view=require_full_body,
                    inference_resolution=infer_res,
                    draw_skeleton=draw_skeleton,
                    draw_motion=draw_motion,
                    show_hud=show_hud
                )

                prog = progress_bar.progress(0.0, text="Initializing neural tracking pipeline...")
                status_text.info("⏳ Processing video frames through motion and pose estimation modules...")

                preview_state = {"last_time": 0.0}

                def progress_callback(ratio, curr, total, annotated_bgr, fall_status, meta):
                    now = time.perf_counter()
                    prog.progress(ratio, text=f"Analyzing Frame {curr:,} of {total:,} ({ratio * 100:.1f}%)")

                    # Throttle Streamlit UI image preview to ~10 FPS for fluid updates without choking browser
                    if now - preview_state["last_time"] >= 0.10 or curr == total:
                        preview_state["last_time"] = now
                        ret, buf = cv2.imencode('.jpg', annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ret:
                            preview_placeholder.image(buf.tobytes(), output_format="JPEG", use_container_width=True)

                try:
                    result = analyzer.analyze_video(local_video_path, progress_callback=progress_callback)
                    st.session_state[f"res_{cache_key}"] = result
                    cached_result = result
                    progress_bar.empty()
                    status_text.success(f"✅ **Analysis Complete**: Processed {result.total_frames_processed:,} frames at {result.average_processing_fps:.1f} processing FPS.")
                except Exception as exc:
                    logger.error("Error analyzing video: %s", exc, exc_info=True)
                    status_text.error(f"❌ Error during video processing: {exc}")

            elif cached_result is not None:
                st.caption("ℹ️ **Cached Results**: Displaying previously computed analysis. Click **🔍 Analyze Video** to re-process.")
                if cached_result.last_annotated_frame is not None:
                    ret, buf = cv2.imencode('.jpg', cached_result.last_annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        preview_placeholder.image(buf.tobytes(), caption="Analyzed Video Frame Preview", output_format="JPEG", use_container_width=True)

            # Display Analysis Summary & Event Log if results are available
            if cached_result is not None:
                st.markdown("---")
                st.subheader("📈 Detection Telemetry & Incident Summary")

                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("Frames Analyzed", f"{cached_result.total_frames_processed:,}")
                s2.metric(
                    "Falls Confirmed",
                    f"{cached_result.falls_confirmed_count}",
                    delta="EMERGENCY" if cached_result.falls_confirmed_count > 0 else None,
                    delta_color="inverse"
                )
                s3.metric("Possible Falls", f"{cached_result.possible_falls_count}")
                s4.metric("Recoveries", f"{cached_result.recoveries_count}")
                s5.metric("Avg Speed", f"{cached_result.average_processing_fps:.1f} FPS")

                # Overall incident status banner
                if cached_result.falls_confirmed_count > 0:
                    st.error(
                        f"🚨 **EMERGENCY: {cached_result.falls_confirmed_count} Fall Incident(s) Confirmed** — "
                        "Rapid descent and sustained post-fall floor immobility verified in video footage."
                    )
                elif cached_result.possible_falls_count > 0:
                    if cached_result.overall_view_type == "UPPER_BODY_ONLY":
                        st.warning(
                            "⚠️ **Possible Fall Detected (Suspended)**: Rapid movement observed, but camera view was restricted "
                            "to upper body. Full-body CCTV footage is required to confirm floor impact."
                        )
                    else:
                        st.warning("⚠️ **Possible Fall Event Recorded**: Subject experienced abrupt transition but recovered before confirmation threshold.")
                else:
                    st.success("🟢 **Monitoring Normal**: No fall incidents or sustained floor immobility detected in this video.")

                # Camera View Adequacy Notice
                if cached_result.overall_view_type == "FULL_BODY":
                    st.info("👁️ **Camera View Adequacy: Full Body (CCTV Ready)** — Essential head, torso, hip, and leg landmarks were consistently tracked.")
                elif cached_result.overall_view_type == "UPPER_BODY_ONLY":
                    st.warning("👁️ **Camera View Adequacy: Upper Body Only** — Lower body was not visible. In accordance with system safeguards, fall decisions were suspended.")
                else:
                    st.error("👁️ **Camera View Adequacy: Insufficient** — Landmarks were occluded or person was outside the frame for significant portions of the video.")

                # Keyframe Incident Gallery (if incidents occurred)
                if cached_result.events:
                    st.markdown("---")
                    st.subheader("📸 Incident Keyframe Snapshots")
                    cols = st.columns(min(4, max(1, len(cached_result.events))))
                    for idx, evt in enumerate(cached_result.events[:4]):
                        with cols[idx % 4]:
                            if evt.keyframe_bgr is not None:
                                ret_th, buf_th = cv2.imencode('.jpg', evt.keyframe_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                                if ret_th:
                                    st.image(
                                        buf_th.tobytes(),
                                        caption=f"Event #{evt.event_id}: {evt.to_state} at {evt.video_timestamp_formatted}",
                                        use_container_width=True
                                    )

                # Detection Event History Table
                st.markdown("---")
                st.subheader("📋 Source Video Detection Event Log")
                if cached_result.events:
                    table_rows = [
                        {
                            "Video Time": e.video_timestamp_formatted,
                            "Timestamp (s)": f"{e.video_timestamp_sec:.2f}s",
                            "Frame #": e.frame_index,
                            "From State": e.from_state,
                            "To State": e.to_state,
                            "Confidence": f"{getattr(e, 'confidence_score', 0.0):.2f}",
                            "Torso Angle": f"{e.torso_angle_deg:.1f}°",
                            "Motion": f"{e.motion_score * 100:.1f}%",
                            "Camera View": e.camera_view,
                            "Decision": "SUSPENDED" if e.is_decision_suspended else "ACTIVE",
                            "Trigger Reason": e.trigger_reason
                        }
                        for e in cached_result.events
                    ]
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No state transitions or abnormal events logged for this video. The subject maintained stable upright posture.")
