import cv2
import numpy as np
import os

def main():
    video_path = "assets/input.mp4"
    if not os.path.exists(video_path):
        print("Video not found!")
        return
        
    cap = cv2.VideoCapture(video_path)
    
    # Read first frame as background
    ret, bg_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return
    bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.GaussianBlur(bg_gray, (5, 5), 0)
    
    # We will check frame 500
    target_frame_idx = 500
    frame = None
    frame_idx = 1
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
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Compute absolute difference
    diff = cv2.absdiff(gray, bg_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the diff
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Diff contours: {len(contours)}")
    
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 15 <= area <= 300:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
            
            if 0.5 <= aspect_ratio <= 2.0 and circularity > 0.4:
                candidates.append((x, y, w, h, area, circularity))
                
    print(f"Candidates in diff frame {target_frame_idx}: {len(candidates)}")
    for idx, cand in enumerate(candidates):
        x, y, w, h, area, circ = cand
        print(f"  Candidate {idx}: Center ({x + w/2.0:.1f}, {y + h/2.0:.1f}), Size: {w}x{h}, Area: {area:.1f}, Circ: {circ:.2f}")

if __name__ == "__main__":
    main()
