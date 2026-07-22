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
    
    # Store ball detections: (frame, xyxy, conf)
    ball_detections = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, verbose=False)
        if len(results) > 0:
            boxes = results[0].boxes
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                if cls_id == 32:  # sports ball
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    ball_detections.append((frame_idx, xyxy, conf))
                    
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"Processed {frame_idx} frames...")
            
    cap.release()
    
    print("\n--- Summary of sports ball detections ---")
    print(f"Total detections: {len(ball_detections)}")
    
    # Group detections by coordinate proximity to see unique locations
    clusters = []
    for frame, xyxy, conf in ball_detections:
        cx = (xyxy[0] + xyxy[2]) / 2.0
        cy = (xyxy[1] + xyxy[3]) / 2.0
        
        # Check if it belongs to an existing cluster
        matched = False
        for cluster in clusters:
            ccx = cluster["cx"]
            ccy = cluster["cy"]
            # Distance threshold of 50 pixels
            dist = ((cx - ccx)**2 + (cy - ccy)**2)**0.5
            if dist < 50:
                cluster["count"] += 1
                cluster["frames"].append(frame)
                cluster["confs"].append(conf)
                # Update cluster average center
                cluster["cx"] = (cluster["cx"] * (cluster["count"] - 1) + cx) / cluster["count"]
                cluster["cy"] = (cluster["cy"] * (cluster["count"] - 1) + cy) / cluster["count"]
                matched = True
                break
                
        if not matched:
            clusters.append({
                "cx": cx,
                "cy": cy,
                "count": 1,
                "frames": [frame],
                "confs": [conf]
            })
            
    print(f"Found {len(clusters)} spatial clusters of detections:")
    for idx, c in enumerate(clusters):
        min_f = min(c["frames"])
        max_f = max(c["frames"])
        avg_conf = sum(c["confs"]) / len(c["confs"])
        print(f"Cluster {idx}: Center ({c['cx']:.1f}, {c['cy']:.1f}), Count: {c['count']}, Frames: {min_f} to {max_f}, Avg Conf: {avg_conf:.3f}")

if __name__ == "__main__":
    main()
