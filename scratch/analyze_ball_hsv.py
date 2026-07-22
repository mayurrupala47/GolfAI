import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from ultralytics import YOLO
import numpy as np

def main():
    model = YOLO("models/multicolor_detector_model.pt")
    cap = cv2.VideoCapture("test_video_1.mp4")
    
    frame_idx = 0
    while cap.isOpened() and frame_idx < 30:
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
            
            # Print crop statistics
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            ch, cs, cv_val = cv2.split(hsv)
            
            # Core crop
            margin_y = int(crop.shape[0] * 0.20)
            margin_x = int(crop.shape[1] * 0.20)
            core = hsv[margin_y:-margin_y, margin_x:-margin_x]
            core_h, core_s, core_v = cv2.split(core)
            
            print(f"Frame {frame_idx}:")
            print(f"  Core Mean HSV: H={np.mean(core_h):.1f}, S={np.mean(core_s):.1f}, V={np.mean(core_v):.1f}")
            print(f"  Core Max HSV:  H={np.max(core_h)}, S={np.max(core_s)}, V={np.max(core_v)}")
            
            # Count pixels in green range with low vs high threshold
            green_low = (core_h >= 38) & (core_h < 85) & (core_s > 40) & (core_v > 40)
            green_high = (core_h >= 38) & (core_h < 85) & (core_s > 100) & (core_v > 100)
            print(f"  Green Low Fraction (S>40, V>40):   {np.sum(green_low)/core_h.size:.2f}")
            print(f"  Green High Fraction (S>100, V>100): {np.sum(green_high)/core_h.size:.2f}")
            
        frame_idx += 1
        
    cap.release()

if __name__ == "__main__":
    main()
