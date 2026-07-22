import cv2
import numpy as np
import os

def main():
    video_path = "assets/input.mp4"
    if not os.path.exists(video_path):
        print("Video not found!")
        return
        
    cap = cv2.VideoCapture(video_path)
    
    # Initialize MOG2 Background Subtractor
    # detectShadows=True helps identify and separate shadows
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    
    frame_idx = 0
    moving_objects_log = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Downscale frame for faster processing and noise reduction
        # (Processing 4K is very slow and noisy; 1920x1080 is much better)
        small_frame = cv2.resize(frame, (1920, 1080))
        
        # Apply background subtraction
        fgMask = backSub.apply(small_frame)
        
        # Threshold to remove shadows (shadows are gray value 127 in OpenCV MOG2)
        _, fgMask = cv2.threshold(fgMask, 250, 255, cv2.THRESH_BINARY)
        
        # Morphological opening to remove small noise dots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of moving areas
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # A golf ball in 1920x1080 is ~4-6 pixels in diameter, so area is 12-30 pixels.
            # Let's search for area between 5 and 80.
            if 5 <= area <= 100:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if 0.5 <= aspect_ratio <= 2.0:
                    candidates.append((x*2, y*2, w*2, h*2, area))  # Scale back coordinates to 4K
                    
        if len(candidates) > 0 and frame_idx > 60: # Let model warm up first 60 frames
            moving_objects_log.append((frame_idx, candidates))
            if len(moving_objects_log) < 20:
                print(f"Frame {frame_idx}: Found {len(candidates)} moving ball-sized candidates.")
                for idx, c in enumerate(candidates):
                    print(f"  Cand {idx}: Center ({c[0] + c[2]/2.0:.1f}, {c[1] + c[3]/2.0:.1f}), Size: {c[2]}x{c[3]}")
                    
        frame_idx += 1
        if frame_idx > 800:
            break
            
    cap.release()
    print(f"\nProcessed {frame_idx} frames. MOG2 warmup complete.")

if __name__ == "__main__":
    main()
