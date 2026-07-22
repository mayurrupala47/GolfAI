import cv2
import numpy as np
import os

def main():
    video_path = "assets/input.mp4"
    if not os.path.exists(video_path):
        print("Video not found!")
        return
        
    cap = cv2.VideoCapture(video_path)
    
    # Let's inspect a few frames where we know a ball might be present (e.g. frame 500)
    target_frame_idx = 500
    frame = None
    
    frame_idx = 0
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        if frame_idx == target_frame_idx:
            frame = f.copy()
            break
        frame_idx += 1
    cap.release()
    
    if frame is None:
        print("Could not read target frame.")
        return
        
    print(f"Analyzing frame {target_frame_idx} with shape {frame.shape}...")
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define white color range
    # Saturation is low, Value (brightness) is high
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 60, 255])
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} raw contours.")
    
    candidates = []
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # Filter by area: golf ball at 200 px/m is ~8-12 pixels diameter, so area is ~50-110 pixels.
        # Let's check contours with area between 10 and 200.
        if 10 <= area <= 300:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            # Circularity check
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
            
            if 0.5 <= aspect_ratio <= 2.0 and circularity > 0.4:
                candidates.append((x, y, w, h, area, circularity))
                
    print(f"Found {len(candidates)} golf ball candidates after size & circularity filtering:")
    for idx, cand in enumerate(candidates):
        x, y, w, h, area, circ = cand
        cx = x + w/2.0
        cy = y + h/2.0
        print(f"  Candidate {idx}: Center ({cx:.1f}, {cy:.1f}), Size: {w}x{h}, Area: {area:.1f}, Circularity: {circ:.2f}")

if __name__ == "__main__":
    main()
