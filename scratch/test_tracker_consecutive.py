import cv2
import os
import numpy as np
from ultralytics import YOLO
import supervision as sv

def main():
    model_path = "models/yolov11.pt"
    video_path = "assets/input.mp4"
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Set minimum_consecutive_frames to 1
    tracker = sv.ByteTrack(
        track_activation_threshold=0.05,
        lost_track_buffer=30,
        minimum_matching_threshold=0.1,
        frame_rate=int(fps),
        minimum_consecutive_frames=1  # Activate track immediately on first detection!
    )
    
    ignore_zones = [
        (1456.1, 472.9), (778.0, 1209.3), (3201.3, 1116.0), (708.6, 1180.0),
        (652.5, 1140.4), (1339.3, 376.5), (606.7, 1096.1), (539.9, 1076.3)
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
                        
        if not detections:
            sv_detections = sv.Detections.empty()
        else:
            xyxy_list = [det[:4] for det in detections]
            conf_list = [det[4] for det in detections]
            cls_list = [0] * len(detections)
            sv_detections = sv.Detections(
                xyxy=np.array(xyxy_list, dtype=np.float32),
                confidence=np.array(conf_list, dtype=np.float32),
                class_id=np.array(cls_list, dtype=np.int32)
            )
            
        tracked = tracker.update_with_detections(sv_detections)
        if len(tracked) > 0 and tracked.tracker_id is not None:
            total_tracks_found += len(tracked)
            for idx in range(len(tracked)):
                box = tracked.xyxy[idx]
                tid = int(tracked.tracker_id[idx])
                print(f"Frame {frame_idx}: Found Track {tid} at bbox {[round(x,1) for x in box]}")
            
        frame_idx += 1
        if frame_idx > 100:  # Check first 100 frames
            break
            
    cap.release()
    print(f"Total tracker outputs in first 100 frames: {total_tracks_found}")

if __name__ == "__main__":
    main()
