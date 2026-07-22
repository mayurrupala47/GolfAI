import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from ultralytics import YOLO
from ai.color_classifier import classify_ball_color

def main():
    model = YOLO("models/multicolor_detector_model.pt")
    cap = cv2.VideoCapture("test_video_1.mp4")
    
    frame_idx = 0
    while cap.isOpened() and frame_idx < 50:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO on the full frame
        results = model(frame, imgsz=320, conf=0.1, verbose=False)
        if len(results) > 0 and len(results[0].boxes) > 0:
            print(f"Frame {frame_idx}:")
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                
                # Crop and classify color
                h, w, _ = frame.shape
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                ball_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                color = classify_ball_color(ball_crop)
                
                print(f"  Det: Class={cls_id}, Conf={conf:.3f}, BBox=[{x1}, {y1}, {x2}, {y2}], Color={color}")
        else:
            print(f"Frame {frame_idx}: No detections")
            
        frame_idx += 1
        
    cap.release()

if __name__ == "__main__":
    main()
