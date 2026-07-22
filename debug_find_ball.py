"""
Find the exact position of the orange ball in the live frame.
Scans for the best orange candidate and reports its coordinates.
"""
import cv2
import numpy as np
import json
import os

def main():
    source = "rtsp://10.30.7.125:8554/mystream"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("ERROR: Cannot open stream")
        return
    
    # Stabilize
    for _ in range(15):
        cap.grab()
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("ERROR: Cannot read frame")
        return
    
    h, w = frame.shape[:2]
    print(f"Frame: {w}x{h}")
    
    # Work at ORIGINAL resolution (no resize) to get exact coordinates
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Use the broadest orange range
    lower = np.array([3, 75, 25])
    upper = np.array([22, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\nOrange contours found at ORIGINAL {w}x{h} resolution:")
    best = None
    best_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 5:
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx = x + bw // 2
            cy = y + bh // 2
            perimeter = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            roi = hsv[y:y+bh, x:x+bw]
            mean_h, mean_s, mean_v, _ = cv2.mean(roi)
            
            print(f"  pos=({cx}, {cy}) area={area:.1f} circ={circ:.2f} HSV=({mean_h:.0f}, {mean_s:.0f}, {mean_v:.0f}) bbox=({x},{y},{bw},{bh})")
            
            # Score: prefer high saturation + circularity + area in ball range
            score = circ * min(area, 200) * (mean_s / 255)
            if score > best_score:
                best_score = score
                best = (cx, cy, area, circ, mean_h, mean_s, mean_v)
    
    if best:
        cx, cy, area, circ, mh, ms, mv = best
        print(f"\n*** BEST BALL CANDIDATE ***")
        print(f"  Position (original 1920x1080): ({cx}, {cy})")
        print(f"  Area={area:.1f}, Circularity={circ:.2f}, HSV=({mh:.0f}, {ms:.0f}, {mv:.0f})")
        
        # Draw it on the frame
        display = frame.copy()
        cv2.circle(display, (cx, cy), 30, (0, 0, 255), 2)
        cv2.putText(display, f"BALL ({cx},{cy})", (cx + 35, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw current tee point for comparison
        with open("config/calibration.json", "r") as f:
            cal = json.load(f)
        for region in cal.get("ignore_regions", []):
            if "tee" in region.get("name", "").lower():
                tx, ty = region["x"], region["y"]
                cv2.circle(display, (tx, ty), 30, (0, 255, 255), 2)
                cv2.putText(display, f"TEE ({tx},{ty})", (tx + 35, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                dist = ((cx - tx)**2 + (cy - ty)**2)**0.5
                print(f"\n  Current TEE calibration: ({tx}, {ty})")
                print(f"  Distance from ball to TEE: {dist:.1f}px")
                if dist > 60:
                    print(f"  *** TEE IS MISCALIBRATED! Distance {dist:.0f}px > 60px threshold ***")
                    print(f"  *** Tracker will NEVER register this ball because it's too far from TEE ***")
                break
        
        cv2.imwrite("debug_ball_vs_tee.png", display)
        print(f"\nSaved: debug_ball_vs_tee.png")
    else:
        print("\nNo orange ball candidates found!")

if __name__ == "__main__":
    main()
