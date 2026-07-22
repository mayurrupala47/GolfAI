import cv2
import numpy as np
import os
import sys
import yaml
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    if len(sys.argv) < 2:
        print("Usage: python scratch/auto_label_dataset.py <path_to_video.mp4> [output_dir] [frame_skip]")
        return
        
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "auto_labeled_dataset"
    frame_skip = int(sys.argv[3]) if len(sys.argv) > 3 else 2  # Extract every 2nd frame by default
    color_choice = sys.argv[4].lower() if len(sys.argv) > 4 else "orange"
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return
        
    config = load_config("config/config.yaml")
    
    if color_choice == "white":
        # White golf ball HSV boundaries: low saturation, high brightness (value)
        lower_limit = np.array([0, 0, 180], dtype=np.uint8)
        upper_limit = np.array([180, 45, 255], dtype=np.uint8)
        print(f"Auto-labeling mode: WHITE ball")
    else:
        # Default Orange limits
        classic_cfg = config.get("classic_detector", {})
        lower_limit = np.array(classic_cfg.get("lower_orange", [4, 60, 50]), dtype=np.uint8)
        upper_limit = np.array(classic_cfg.get("upper_orange", [25, 255, 255]), dtype=np.uint8)
        print(f"Auto-labeling mode: ORANGE ball")
    
    # Target folders
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    previews_dir = os.path.join(output_dir, "previews")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(previews_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video file")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {video_path} ({total_frames} frames)")
    print(f"Output directory: {output_dir}")
    print(f"HSV Lower: {lower_limit}, Upper: {upper_limit}")
    
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        frame_idx += 1
        
        # Skip frames to ensure image diversity and prevent redundant data
        if frame_idx % frame_skip != 0:
            continue
            
        h, w = frame.shape[:2]
        
        # Convert to HSV and run the color filter
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_limit, upper_limit)
        
        # Run morphological opening to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_candidate = None
        best_circularity = 0.0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5 or area > 1000:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # solidity = area / hull_area
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Capture the best circular and solid orange candidate
            if circularity >= 0.60 and solidity >= 0.80:
                if circularity > best_circularity:
                    best_circularity = circularity
                    x, y, bw, bh = cv2.boundingRect(contour)
                    ball_candidate = (x, y, bw, bh)
                    
        if ball_candidate is not None:
            bx, by, bw, bh = ball_candidate
            
            # Pad the bounding box slightly to make sure it includes the whole ball
            pad_w = max(4, int(bw * 0.15))
            pad_h = max(4, int(bh * 0.15))
            
            x1 = max(0, bx - pad_w)
            y1 = max(0, by - pad_h)
            x2 = min(w, bx + bw + pad_w)
            y2 = min(h, by + bh + pad_h)
            
            # Recalculate padded width, height, and center
            pw = x2 - x1
            ph = y2 - y1
            cx = x1 + pw / 2.0
            cy = y1 + ph / 2.0
            
            # Normalize to YOLO format (0.0 to 1.0)
            norm_cx = cx / w
            norm_cy = cy / h
            norm_w = pw / w
            
            # Maintain square aspect ratio for YOLO anchor matches
            norm_h = ph / h
            
            # Write Image
            img_filename = f"frame_{frame_idx:06d}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            cv2.imwrite(img_path, frame)
            
            # Write Label in YOLO format (Class 0)
            label_filename = f"frame_{frame_idx:06d}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            with open(label_path, "w") as f:
                f.write(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                
            # Draw preview for verification
            preview = frame.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(preview, "ball", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(previews_dir, img_filename), preview)
            
            saved_count += 1
            
        # Log progress every 200 frames
        if frame_idx % 200 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames... Auto-labeled {saved_count} images.")
            
    cap.release()
    print(f"\nSUCCESS: Processing complete!")
    print(f"Total frames processed: {frame_idx}")
    print(f"Successfully auto-labeled: {saved_count} images.")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Previews (for inspection): {previews_dir}")

if __name__ == "__main__":
    main()
