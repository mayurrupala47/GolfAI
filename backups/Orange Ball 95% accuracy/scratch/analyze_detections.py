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
    
    # We will look at a few frames where the ball might be active or detected
    # Let's inspect frames 0, 100, 300, 500, 800, 1200, 1500, 1800
    target_frames = [0, 100, 300, 500, 800, 1200, 1500, 1800]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in target_frames:
            print(f"\n--- Frame {frame_idx} ---")
            results = model(frame, verbose=False)
            if len(results) > 0:
                boxes = results[0].boxes
                for idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    class_name = model.names[cls_id]
                    # We are interested in class 32 (sports ball) or other classes that could be the ball
                    print(f"  Det {idx}: Class {cls_id} ({class_name}), Conf: {conf:.3f}, Bbox: {[round(x, 1) for x in xyxy]}")
                    
        frame_idx += 1
        if frame_idx > max(target_frames):
            break
            
    cap.release()

if __name__ == "__main__":
    main()
