# Mini Golf AI Stroke Detection POC

A modular, production-ready Python application that detects and counts golf ball strokes from an overhead camera video feed. This Proof of Concept (POC) validates the feasibility of using AI-based computer vision and kinematic state estimation for automated game tracking.

---

## 1. Directory Structure

```
mini-golf-ai/
├── assets/
│   ├── input.mp4                # Input video (user-supplied or mock-generated)
│   └── oracle_positions.json   # Mock ground-truth positions for testing
├── config/
│   └── config.yaml              # App configuration (thresholds, topics, scale)
├── models/
│   └── yolov11.pt               # YOLOv11 model weights (auto-downloaded if missing)
├── ai/
│   ├── interfaces.py            # Abstract interfaces decoupling components
│   ├── detector.py              # YOLO and Oracle/Mock object detectors
│   ├── tracker.py               # ByteTrack tracking wrapper
│   └── motion.py                # Kinematics calculations (smoothing, speed, accel)
├── engine/
│   ├── state_machine.py         # Finite State Machine for golf ball transitions
│   └── stroke_engine.py         # Coordinator driving the processing workflow
├── mqtt/
│   ├── publisher.py             # Telemetry publication (paho-mqtt & mock)
│   └── __init__.py
├── visualization/
│   └── overlay.py               # HUD overlay and trajectory drawing using OpenCV
├── exporters/
│   ├── csv_export.py            # Appends stroke events to CSV
│   └── json_export.py           # Appends stroke events to JSON
├── outputs/
│   ├── output.mp4               # Annotated output video showing HUD and trails
│   ├── strokes.csv              # CSV of stroke records
│   └── strokes.json             # JSON array of stroke records
├── main.py                      # Main entrypoint script
├── create_mock_video.py         # Test utility to generate synthetic inputs
├── requirements.txt             # Project library dependencies
└── README.md                    # Setup and usage guide
```

---

## 2. Architecture & Data Flow

The system employs a modular layout adhering to Clean Architecture principles. Major business logic units depend on interfaces (`IBallDetector`, `IBallTracker`, `IMotionAnalyzer`, `IMqttPublisher`) rather than concrete library implementations:

```
           [ Video Feed / Camera ]
                      │ (Frames)
                      ▼
             [ IBallDetector ] (e.g., YOLOv11 / Mock)
                      │ (Detections: Bboxes & Confidences)
                      ▼
             [ IBallTracker ] (e.g., ByteTrack)
                      │ (Tracks: Bboxes & Track IDs)
                      ▼
            [ IMotionAnalyzer ] (Smooths jitter & computes kinematics)
                      │ (Center coordinates, Speed, Acceleration, Travel)
                      ▼
     ┌────────────────┴────────────────┐
     ▼                                 ▼
[ BallState Machine ]         [ HUD Visualizer ] (OpenCV Overlays)
     │ (State: STOPPED/READY/MOVING)   │
     ▼                                 ▼
[ Stroke Trigger Logic ]      [ Annotated Output Video ]
     │ (Event: Stroke detected!)
     ▼
[ Exporters & Telemetry ] ──► (MQTT / JSON / CSV)
```

---

## 3. The Ball State Machine

To prevent camera noise, tracker swap artifacts, and minor setup adjustments from triggering false-positive stroke events, the application runs a **Finite State Machine (FSM)** for every golf ball:

*   **UNKNOWN**: Default state when a track first starts. Transitions to **STOPPED** as soon as speed drops below `stop_speed`.
*   **STOPPED**: Ball is slow/stationary. Transitions to **READY** after remaining below `stop_speed` for `ready_delay_frames` (default: 30 frames).
*   **READY**: The ball is completely stationary, settled, and ready for a shot. Once speed exceeds `moving_speed` (default: 0.8 m/s), we begin tracking cumulative distance. If the distance exceeds `minimum_distance` (default: 15 cm), it triggers a **Stroke Event** and transitions to **MOVING**. If it stops before reaching the minimum distance, it drops back to **STOPPED** without incrementing strokes.
*   **MOVING**: Ball is actively traveling down the lane. Transitions back to **STOPPED** once speed drops below `stop_speed` for `stop_delay_frames` (default: 15 frames).

---

## 4. Mathematics & Spatial Calibration

Motion analytics are calculated from pixels and mapped to real-world units (meters, meters per second) using a spatial calibration factor:

1.  **Coordinates Smoothing**: To mitigate pixel-boundary noise, coordinates are smoothed using a simple moving average window of $N$ frames:
    $$X_{\text{smoothed}, t} = \frac{1}{N} \sum_{i=0}^{N-1} X_{t-i}$$
2.  **Velocity ($v_x, v_y$)**:
    $$v_x = \frac{X_t - X_{t-1}}{\text{pixels\_per\_meter}} \cdot \text{FPS}$$
    $$\text{speed} = \sqrt{v_x^2 + v_y^2} \text{ (m/s)}$$
3.  **Acceleration ($a$)**:
    $$a_x = \frac{v_{x, t} - v_{x, t-1}}{1 / \text{FPS}}$$
    $$\text{acceleration} = \sqrt{a_x^2 + a_y^2} \text{ (m/s}^2\text{)}$$

---

## 5. Setup & Installation

### Requirements
*   Python 3.12+ (tested up to 3.14)
*   MQTT Broker (optional, fallback offline mode is supported automatically)

### Local Virtual Environment Setup
It is highly recommended to isolate your dependencies in a virtual environment:

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate virtual environment
# On Windows (Command Prompt)
venv\Scripts\activate.bat
# On Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# On macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 6. How to Run

### Mode A: Mock Verification (Brokerless & Neural-Net-Free)
To test the tracking, physics math, state transitions, JSON/CSV exports, and output rendering without requiring a GPU or downloading neural network model weights:

```bash
# 1. Generate the synthetic video and oracle position logs
python create_mock_video.py

# 2. Run the main processing using the mock detector and mock MQTT client
python main.py --use-mock-detector --mock-mqtt
```

This will run instantly on any CPU. You can examine `outputs/output.mp4` to see the annotated HUD overlays, and inspect `outputs/strokes.csv` / `outputs/strokes.json` to confirm that exactly **2 strokes** were recorded for **Ball #1**.

### Mode B: Production Run (YOLO & MQTT)
To run the full computer vision detector on a real mini-golf video file:

1.  Configure your MQTT host and credentials in `config/config.yaml`.
2.  Place your video in `assets/input.mp4`.
3.  Run:
    ```bash
    python main.py
    ```

---

## 7. Output Events Format

### MQTT Payload (Topic: `minigolf/stroke`)
```json
{
  "camera_id": "CAM-01",
  "hole": 1,
  "ball_id": 1,
  "stroke": 2,
  "timestamp": "2026-07-01T17:00:00"
}
```

### JSON Export (`outputs/strokes.json`)
```json
[
  {
    "stroke": 1,
    "ball": 1,
    "frame": 90,
    "time": 3.0,
    "speed": 1.12
  }
]
```

### CSV Export (`outputs/strokes.csv`)
```csv
Stroke Number,Ball ID,Frame,Timestamp,Speed,Distance
1,1,90,3.000,1.12,1.50
```
