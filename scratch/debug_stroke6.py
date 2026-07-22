import sys
import os
sys.path.insert(0, os.path.abspath("."))

import cv2
import yaml
import logging
from ai.detector import YoloOnlyBallDetector

logging.basicConfig(level=logging.INFO)

video_path = r"D:\AI Projects\Golf AI\Training Videos\Golf All color videos\orange_right_1.mp4"
model_path = r"models/multicolor_detector_model.pt"

config = {
    "yolo_detector": {
        "model_path": model_path,
        "confidence_threshold": 0.15
    }
}

detector = YoloOnlyBallDetector(config)
cap = cv2.VideoCapture(video_path)

frame_idx = 0
tee_anchor = (244.8, 319.3)

print("--- Inspecting Full-Frame Detections Frames 1050 to 1250 ---")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if 1050 <= frame_idx <= 1250:
        # Full frame search without crop hint
        dets = detector.detect(frame, hint_center=None, hint_moving=False)
        if len(dets) > 0:
            for d in dets:
                cx = (d[0] + d[2]) / 2.0
                cy = (d[1] + d[3]) / 2.0
                dist = ((cx - tee_anchor[0])**2 + (cy - tee_anchor[1])**2)**0.5
                print(f"Frame {frame_idx}: center=({cx:.1f}, {cy:.1f}) conf={d[4]:.2f} color={d[5]} dist_from_anchor={dist:.1f}px")

cap.release()
