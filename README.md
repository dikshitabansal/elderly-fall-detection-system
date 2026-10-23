# Real-Time Elderly Fall Detection and Monitoring System

**Milestone 4**: Realistic Multi-Signal Fall Detection (Stairs, Occlusion, Aspect Ratio & 5-State FSM)

An operational real-time vision system built using OpenCV, Google MediaPipe, and Streamlit. The system combines traditional image processing motion analysis with skeletal posture kinematics, governed by a 5-state finite state machine with strict clinical reliability safeguards. It supports both **Live Monitoring** (webcam/CCTV) and **Video Analysis** (MP4/AVI prerecorded footage).

---

## 📌 Operating Modes & System Architecture

The application provides two completely decoupled operating modes selectable via the Streamlit interface:

### MODE 1 — Live Monitoring
- **Live Camera Acquisition**: Threaded capture worker with bounded latest-frame buffer (size = 1) and frame deduplication sequence counter.
- **Traditional Image Processing (`src/motion.py`)**: Grayscale smoothing, consecutive-frame differencing $|I_t - I_{t-1}|$, binary thresholding, morphological opening & dilation, contour extraction, bounding box tracking, and motion energy scoring.
- **Pose Kinematics (`src/features.py`)**: Mid-shoulder, mid-hip, and lower-body anchors with visibility filtering, acute torso orientation angle relative to vertical ($0^\circ = \text{upright}$, $90^\circ = \text{horizontal}$), bounding-box aspect ratio ($W/H$), camera view adequacy classification (`FULL_BODY`, `UPPER_BODY_ONLY`, `PARTIAL`, `INSUFFICIENT`), and temporal kinematics ($v_y = dy/dt$ downward velocity, $\omega = d\theta/dt$ angular velocity, and cumulative vertical drop from baseline).
- **Multi-State FSM (`src/detector.py`)**: 5-state finite state machine (`NORMAL` $\rightarrow$ `POSSIBLE_FALL` $\rightarrow$ `FALL_CONFIRMED` $\rightarrow$ `RECOVERY`, plus `UNCERTAIN` for boundary occlusion) with clinical safeguards against static false positives, laptop webcam upper-body suspension, immediate recovery cancellation, and duplicate alert suppression.
- **Interactive Telemetry Dashboard (`app.py`)**: Dual video stream (live skeleton overlay + colorized binary motion mask), real-time metric cards, emergency alert banners, and timestamped detection event log table.

### MODE 2 — Uploaded Video Analysis (`src/video_analysis.py`)
- **Format Validation**: Validates file integrity, headers, and OpenCV decoding for `.mp4` and `.avi` containers; reports actionable errors for corrupt, empty, or unsupported files.
- **Video Metadata Extraction**: Automatically computes and displays resolution ($W \times H$), source FPS, duration (`MM:SS.ss`), and total frame count.
- **Frame-by-Frame Pipeline Reuse**: Passes every video frame through the identical motion differencing, MediaPipe pose tracking, posture kinematics, and FSM safeguard modules as live monitoring.
- **Multi-Signal Evidence Fusion**: Combines downward velocity, cumulative vertical drop ($\Delta y$), postural tilt ($\theta$), bounding box aspect ratio ($AR = W/H$), motion disturbance bursts, and post-impact floor immobility.
- **Realistic Fall Support**: Successfully verified on complex realistic videos including `istockphoto-2178409204-640_adpp_is.mp4` (a person falling down stairs into a slumped posture at the base of the stairs).
- **Incident Keyframe Capture**: Saves annotated visual snapshot keyframes at the exact moment of each state transition.
- **Streamlit Result Caching**: Avoids re-processing the entire video on ordinary Streamlit UI reruns using signature-based session caching.

---

## 📂 Project Structure

```
elderly-fall-detection/
├── .venv/                         # Local Python 3.12 virtual environment (NumPy 1.26.4 + PyArrow 16.1.0)
├── models/                        # Pretrained model weights
│   └── pose_landmarker_lite.task  # MediaPipe PoseLandmarker model (Float16)
├── src/
│   ├── __init__.py                # Package exports and versioning
│   ├── camera.py                  # Threaded live webcam capture & deduplication
│   ├── pose.py                    # MediaPipe pose estimation, aspect-preserving scaling & occlusion diagnostics
│   ├── motion.py                  # Traditional image processing motion analysis
│   ├── features.py                # Pose-derived features, torso orientation, cumulative drop & kinematics
│   ├── detector.py                # Multi-modal fall detection state machine (5 states) & safeguards
│   ├── video_analysis.py          # Video validation, metadata, frame-by-frame analysis & keyframe capture
│   └── utils.py                   # Rolling FPS calculator, diagnostic HUD & emergency banners
├── tests/
│   ├── __init__.py                # Test package initialization
│   ├── test_camera.py             # CameraCapture unit, deduplication & integration tests (5 tests)
│   ├── test_pose.py               # PoseEstimator inference, topology, occlusion & boundary tests (9 tests)
│   ├── test_utils.py              # FPSCalculator and HUD rendering tests (3 tests)
│   ├── test_motion.py             # Traditional image processing tests on synthetic arrays (6 tests)
│   ├── test_features.py           # Posture geometry, drop, keypoints & kinematics tests (8 tests)
│   ├── test_detector.py           # FSM state lifecycle, stair fall, safeguards & alert tests (14 tests)
│   └── test_video_analysis.py     # Video validation, metadata, timestamps, reset & pipeline tests (11 tests)
├── app.py                         # Streamlit dual-mode live monitoring & video analysis web application
├── requirements.txt               # Compatible pinned package dependencies
└── README.md                      # Milestone documentation, architecture & test results
```

---

## 🧪 Verification & Test Results

The full test suite executes **56 automated unit and integration tests** across 7 test modules:

```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
```

**Result: Ran 56 tests (OK)**:
- `test_camera.py` (5 tests): Default initialization, unavailable camera index recovery, physical camera capture, threaded background capture, and frame deduplication.
- `test_pose.py` (9 tests): Topology verification (33 landmarks, 35 connections), model caching, blank frame processing, invalid frame recovery, landmark extraction, inference scaling, missing landmarks, low-confidence occlusion, and boundary clipping.
- `test_utils.py` (3 tests): Rolling FPS calculator, diagnostic HUD rendering, and model integrity.
- `test_motion.py` (6 tests): Grayscale smoothing, consecutive-frame differencing, moving block detection, morphological noise suppression, reset, and empty frame safety.
- `test_features.py` (8 tests): Upright standing posture, horizontal lying posture, upper-body only view classification, occluded torso landmarks, temporal vertical velocity ($dy/dt$), empty landmark recovery, partial keypoint extraction, and cumulative vertical drop computation.
- `test_detector.py` (14 tests): Full state lifecycle, static horizontal false positive rejection, upper-body view decision suspension, immediate recovery cancellation, duplicate alert suppression, ordinary movement stability, detector reset, stair fall with non-horizontal angle, occluded disappearance (`UNCERTAIN`), normal walking down stairs without false alarms, and partial landmarks fall detection.
- `test_video_analysis.py` (11 tests): Sub-minute and hour timestamp formatting, negative clamp safety, non-existent file rejection, unsupported file extension rejection, empty file rejection, corrupt file header rejection, synthetic video metadata validation, temporal state reset, full video analysis pipeline execution, progress callbacks, and early stop cancellation.

---

## ⚡ Live & Video Processing Performance

- **Live Webcam Pipeline Throughput**: **26.4 - 27.0 FPS** (Rolling smoothed FPS: **29.7 - 30.8 FPS** at the 30.0 FPS camera hardware limit).
- **Video Processing Throughput**: **~28 - 35 processing FPS** on typical $640 \times 480$ / $320 \times 240$ input with MediaPipe Float16 Lite model.
- **Latency Breakdown**:
  - Motion Differencing: ~0.8 ms
  - MediaPipe Pose Tracking (Video Mode): ~10.2 ms
  - Posture Feature Extraction: ~0.1 ms
  - Fall State Machine: ~0.05 ms
  - Rendering & Compression: ~1.0 ms

---

## ⚠️ Camera View & Clinical Reliability Disclaimer

- **Laptop Webcam View**: When seated in front of a laptop webcam, only the head, shoulders, and upper torso are visible.
- **Guard in Place**: The system explicitly classifies this view as `UPPER_BODY_ONLY` and **suspends full-body fall confirmation** to prevent false alarms.
- **Future CCTV Integration**: The modular architecture is designed so that when connected to a wide-angle CCTV camera capturing the full body (including hips, knees, and ankles), full-body fall classification operates automatically without code modification.
- **Prerecorded Video Analysis**: Video analysis mode evaluates recorded footage with full-body or upper-body classification based on observed landmarks and source-video timestamps.
- **Clinical Disclaimer**: This software is an engineering research prototype for computer vision study and does not claim clinical or certified diagnostic reliability.

---

## 🚀 Exact Commands to Verify

All commands must be executed from the project root directory using PowerShell and the project's virtual environment:

### 1. Launch the Dual-Mode Streamlit Application
```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```
Open `http://localhost:8501` in your browser. Use the sidebar **Operating Mode** selector:
- Select **🎥 Live Monitoring** and click **▶ Start** to stream from your live webcam.
- Select **📁 Video Analysis** to upload an MP4 or AVI file and click **🔍 Analyze Video**.

### 2. Run the Full Automated Test Suite (56 Tests)
```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
```

### 3. Run Package & Environment Dependency Check
```powershell
& ".\.venv\Scripts\python.exe" -m pip check
```
