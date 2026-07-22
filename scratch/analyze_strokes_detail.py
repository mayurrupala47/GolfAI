import cv2
import numpy as np
import yaml
from ai.detector import YoloOnlyBallDetector
from ai.kalman_tracker import KalmanBallTracker
from ai.motion import MotionAnalyzer
from engine.stroke_engine import StrokeEngine
from mqtt.publisher import MockMqttPublisher
from exporters.csv_export import CsvExporter
from exporters.json_export import JsonExporter

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
config["low_fps"] = True
config["strict_course_mode"] = True
config["yolo_detector"]["model_path"] = "models/multicolor_detector_model.pt"

detector = YoloOnlyBallDetector(config)
tracker = KalmanBallTracker(config)
motion_analyzer = MotionAnalyzer(smoothing_window=5)
mqtt_pub = MockMqttPublisher(config)
csv_exp = CsvExporter("outputs/strokes.csv")
json_exp = JsonExporter("outputs/strokes.json")

engine = StrokeEngine(config, detector, tracker, motion_analyzer, mqtt_pub, csv_exp, json_exp)

cap = cv2.VideoCapture("test_video_1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

frame_idx = 0
print("\n========== STROKE & HOLE DETAIL TRACE ==========")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    # Process frame
    active_metrics, active_states, stroke_events = engine.process_frame(
        frame=frame,
        frame_idx=frame_idx,
        fps=fps,
        timestamp=float(frame_idx / fps)
    )
    
    for event in stroke_events:
        track_id, count, event_type = event
        pos = (active_metrics[0]['x'], active_metrics[0]['y']) if active_metrics else (0,0)
        speed = active_metrics[0]['speed'] if active_metrics else 0.0
        print(f"[Frame {frame_idx:04d}] Event: {event_type.upper():<12} | Ball {track_id} | Stroke Count: {count} | Pos: ({pos[0]:.1f}, {pos[1]:.1f}) | Speed: {speed:.2f} m/s")

cap.release()
print("================================================\n")
