import cv2
import os
import numpy as np
from ultralytics import YOLO
from ai.tracker import ByteBallTracker

def main():
    model_path = "models/yolov11.pt"
    video_path = "assets/input.mp4"
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = ByteBallTracker(fps=int(fps), track_thresh=0.05)
    
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
    
    frame_idx = 0
    total_tracks_found = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, conf=0.05, verbose=False)
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                if cls_id == 32:  # sports ball
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    cx = (xyxy[0] + xyxy[2]) / 2.0
                    cy = (xyxy[1] + xyxy[3]) / 2.0
                    
                    is_static = False
                    for iz in ignore_zones:
                        dist = ((cx - iz[0])**2 + (cy - iz[1])**2)**0.5
                        if dist < 60:
                            is_static = True
                            break
                            
                    if not is_static:
                        detections.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf))
                        
        # Pass to tracker
        tracks = tracker.update(detections, frame)
        if len(tracks) > 0:
            total_tracks_found += len(tracks)
            print(f"Frame {frame_idx}: Tracker returned {len(tracks)} tracks! Tracks: {tracks}")
            
        frame_idx += 1
        if frame_idx > 100:  # Check first 100 frames
            break
            
    cap.release()
    print(f"Total tracker outputs in first 100 frames: {total_tracks_found}")

if __name__ == "__main__":
    main()
