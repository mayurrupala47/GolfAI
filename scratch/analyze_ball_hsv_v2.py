import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from ultralytics import YOLO
import numpy as np

def main():
    model = YOLO("models/multicolor_detector_model.pt")
    cap = cv2.VideoCapture("test_video_2.mp4")
    
    frame_idx = 0
    while cap.isOpened() and frame_idx < 100:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, imgsz=320, conf=0.5, verbose=False)
        if len(results) > 0 and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            xyxy = box.xyxy[0].tolist()
            h, w, _ = frame.shape
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
            # Print core crop HSV stats
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # Core crop
            margin_y = int(crop.shape[0] * 0.20)
            margin_x = int(crop.shape[1] * 0.20)
            core = hsv[margin_y:-margin_y, margin_x:-margin_x]
            core_h, core_s, core_v = cv2.split(core)
            
            print(f"Frame {frame_idx}: Core Mean Hue={np.mean(core_h):.1f}, Sat={np.mean(core_s):.1f}, Val={np.mean(core_v):.1f}")
            
        frame_idx += 1
        
    cap.release()

if __name__ == "__main__":
    main()
