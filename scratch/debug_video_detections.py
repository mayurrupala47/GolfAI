import cv2
import numpy as np
import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python scratch/debug_video_detections.py <path_to_video.mp4> [model_path]")
        return
        
    video_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/combined_ball_detector.pt"
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        return
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return
        
    print(f"Loading model: {model_path}")
    from ultralytics import YOLO
    model = YOLO(model_path, task="detect")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Analyzing {total_frames} frames from {video_path}...")
    
    frame_idx = 0
    max_conf = 0.0
    best_frame = None
    best_bbox = None
    detections_found = 0
    
    # Store confidence distribution
    confidences = []
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        frame_idx += 1
        
        # Auto-rotate portrait frames to landscape to match YOLO's training layout
        h, w = frame.shape[:2]
        if h > w:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            h, w = frame.shape[:2]
            
        # Run inference at 640 (matching training resolution) to avoid losing tiny objects
        results = model(frame, imgsz=640, conf=0.001, verbose=False)
        
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Check for golf-ball class (index 0)
                if cls_id == 0:
                    confidences.append(conf)
                    detections_found += 1
                    
                    if conf > max_conf:
                        max_conf = conf
                        best_frame = frame.copy()
                        best_bbox = box.xyxy[0].tolist()
                        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")
            
    cap.release()
    
    print("\n--- DIAGNOSTIC RESULTS ---")
    print(f"Total frames processed: {frame_idx}")
    print(f"Frames with ball detections (conf > 0.001): {detections_found}")
    
    if len(confidences) > 0:
        conf_arr = np.array(confidences)
        print(f"Max Confidence: {max_conf:.4f}")
        print(f"Mean Confidence: {np.mean(conf_arr):.4f}")
        print(f"Median Confidence: {np.median(conf_arr):.4f}")
        print(f"Detections above 0.35 threshold: {np.sum(conf_arr >= 0.35)}")
        print(f"Detections above 0.20 threshold: {np.sum(conf_arr >= 0.20)}")
        print(f"Detections above 0.10 threshold: {np.sum(conf_arr >= 0.10)}")
        
        if best_frame is not None and best_bbox is not None:
            x1, y1, x2, y2 = map(int, best_bbox)
            cv2.rectangle(best_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(best_frame, f"ball ({max_conf:.2f})", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            output_name = "debug_max_conf_frame.png"
            cv2.imwrite(output_name, best_frame)
            print(f"\nSaved best detection frame to: {output_name}")
    else:
        print("ERROR: Zero ball detections were found in the entire video, even at 0.001 confidence!")

if __name__ == "__main__":
    main()
