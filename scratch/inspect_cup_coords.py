import cv2
import yaml
from ai.detector import YoloOnlyBallDetector

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
config["yolo_detector"]["model_path"] = "models/multicolor_detector_model.pt"

detector = YoloOnlyBallDetector(config)
cap = cv2.VideoCapture("test_video_1.mp4")

frame_idx = 0
print("\n========== ALL DETECTIONS NEAR CUP (X < 150) ==========")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    detections = detector.detect(frame)
    for det in detections:
        x1, y1, x2, y2, conf, color = det
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if cx < 150.0:
            print(f"[Frame {frame_idx:04d}] Cup Region Detection: Center=({cx:.1f}, {cy:.1f}), Conf={conf:.2f}, Color={color}")

cap.release()
print("========================================================\n")
