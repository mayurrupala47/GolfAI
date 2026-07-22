import cv2
import os
from ultralytics import YOLO

def main():
    model_path = "models/yolov11.pt"
    video_path = "assets/input.mp4"
    
    if not os.path.exists(model_path):
        print("Model not found!")
        return
    if not os.path.exists(video_path):
        print("Video not found!")
        return
        
    print("Loading model...")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    # Static clusters to ignore (from our previous run)
    ignore_zones = [
        (1456.1, 472.9),
        (778.0, 1209.3),
        (3201.3, 1116.0),
        (708.6, 1180.0),
        (652.5, 1140.4),
        (1339.3, 376.5),
        (606.7, 1096.1),
        (539.9, 1076.3)
    ]
    
    found_moving_detections = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO with very low confidence threshold (0.05)
        results = model(frame, conf=0.05, verbose=False)
        if len(results) > 0:
            boxes = results[0].boxes
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                if cls_id == 32:  # sports ball
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    cx = (xyxy[0] + xyxy[2]) / 2.0
                    cy = (xyxy[1] + xyxy[3]) / 2.0
                    
                    # Check if it is near any ignore zone
                    is_static = False
                    for iz in ignore_zones:
                        dist = ((cx - iz[0])**2 + (cy - iz[1])**2)**0.5
                        if dist < 60:
                            is_static = True
                            break
                            
                    if not is_static:
                        found_moving_detections += 1
                        if found_moving_detections <= 30:
                            print(f"Frame {frame_idx}: Found candidate at ({cx:.1f}, {cy:.1f}) with Conf: {conf:.3f}")
                            
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"Processed {frame_idx} frames...")
            
    cap.release()
    print(f"\nTotal non-static sports ball detections found at conf >= 0.05: {found_moving_detections}")

if __name__ == "__main__":
    main()
