"""
Grab a single frame from the RTSP stream and save diagnostic images to disk.
No GUI window needed - saves PNG files for analysis.
"""

import cv2
import numpy as np
import sys
import json
import os
import time

def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "rtsp://10.30.7.125:8554/mystream"

    print(f"Connecting to: {source}")
    
    # Try with TCP transport for more reliable RTSP
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("ERROR: Could not open video source!")
        print("Trying with default backend...")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("ERROR: Still could not open. Check network/URL.")
            return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Stream opened: {w}x{h} @ {fps:.1f} FPS")

    # Grab a few frames to let the stream stabilize
    print("Stabilizing stream (grabbing 10 frames)...")
    for i in range(10):
        ret = cap.grab()
        if not ret:
            print(f"  Failed to grab frame {i}")

    # Now read a real frame
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame from stream!")
        cap.release()
        return

    print(f"Got frame: {frame.shape[1]}x{frame.shape[0]}")
    
    # Save raw frame
    cv2.imwrite("debug_raw_frame.png", frame)
    print("Saved: debug_raw_frame.png")

    # Resize for processing (match main.py)
    proc_w = 1280
    if frame.shape[1] != proc_w:
        scale = proc_w / frame.shape[1]
        h_new = int(frame.shape[0] * scale)
        proc_frame = cv2.resize(frame, (proc_w, h_new))
    else:
        proc_frame = frame.copy()

    ph, pw = proc_frame.shape[:2]
    
    # Convert to HSV
    hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)

    # Load calibration
    tee_scaled = None
    cal_path = "config/calibration.json"
    if os.path.exists(cal_path):
        with open(cal_path, "r") as f:
            cal = json.load(f)
        base_res = cal.get("source_resolution", [1920, 1080])
        sx = pw / base_res[0]
        sy = ph / base_res[1]
        for region in cal.get("ignore_regions", []):
            if "tee" in region.get("name", "").lower():
                tee_scaled = (int(region["x"] * sx), int(region["y"] * sy))
                print(f"Tee point (scaled to proc resolution): {tee_scaled}")
                break

    # Test multiple HSV ranges
    ranges = [
        ("current",  np.array([3, 75, 25]),  np.array([22, 255, 255])),
        ("broad",    np.array([0, 30, 30]),   np.array([30, 255, 255])),
        ("xbroad",   np.array([0, 15, 15]),   np.array([40, 255, 255])),
        ("red_wrap", np.array([160, 30, 30]), np.array([180, 255, 255])),  # Red wraps around 0/180
        ("full_orange_red", None, None),  # Combined: 0-30 + 160-180
    ]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    for name, lower, upper in ranges:
        if name == "full_orange_red":
            # Combine orange + red-wrap
            m1 = cv2.inRange(hsv, np.array([0, 15, 15]), np.array([40, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([160, 30, 30]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(m1, m2)
        else:
            mask = cv2.inRange(hsv, lower, upper)
        
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw on a copy
        display = proc_frame.copy()
        ball_count = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= 3:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cx, cy = x + bw // 2, y + bh // 2
                
                color = (0, 255, 0) if 6 <= area <= 200 else (0, 165, 255)
                cv2.drawContours(display, [cnt], -1, color, 2)
                
                # Get mean HSV
                roi = hsv[y:y+bh, x:x+bw]
                if roi.size > 0:
                    mean_h, mean_s, mean_v, _ = cv2.mean(roi)
                    cv2.putText(display, f"A:{area:.0f} H:{mean_h:.0f}S:{mean_s:.0f}V:{mean_v:.0f}",
                                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                
                if 6 <= area <= 200:
                    ball_count += 1

        # Draw tee point
        if tee_scaled:
            cv2.circle(display, tee_scaled, 40, (0, 255, 255), 2)
            cv2.putText(display, "TEE", (tee_scaled[0] - 15, tee_scaled[1] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Label
        cv2.putText(display, f"{name}: contours={len(contours)}, ball-sized={ball_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Create mask vis
        mask_vis = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
        
        combined = np.hstack([display, mask_vis])
        fname = f"debug_mask_{name}.png"
        cv2.imwrite(fname, combined)
        print(f"Saved: {fname} (contours={len(contours)}, ball-sized={ball_count})")

    # Print HSV stats around tee area
    if tee_scaled:
        tx, ty = tee_scaled
        for r in [30, 50, 80]:
            x1 = max(0, tx - r)
            y1 = max(0, ty - r)
            x2 = min(pw, tx + r)
            y2 = min(ph, ty + r)
            roi = hsv[y1:y2, x1:x2]
            if roi.size > 0:
                print(f"\n=== HSV Stats around Tee (radius={r}px) ({x1},{y1})-({x2},{y2}) ===")
                print(f"  Hue:  min={roi[:,:,0].min()}, max={roi[:,:,0].max()}, mean={roi[:,:,0].mean():.1f}, median={np.median(roi[:,:,0]):.0f}")
                print(f"  Sat:  min={roi[:,:,1].min()}, max={roi[:,:,1].max()}, mean={roi[:,:,1].mean():.1f}, median={np.median(roi[:,:,1]):.0f}")
                print(f"  Val:  min={roi[:,:,2].min()}, max={roi[:,:,2].max()}, mean={roi[:,:,2].mean():.1f}, median={np.median(roi[:,:,2]):.0f}")

    # Also scan the ENTIRE frame for any orange-ish blobs
    print("\n=== Full Frame Scan: All contours with area 6-500 across all HSV ranges ===")
    for name, lower, upper in ranges[:3]:  # Skip red_wrap and combined
        mask = cv2.inRange(hsv, lower, upper)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 6 <= area <= 500:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cx, cy = x + bw // 2, y + bh // 2
                roi = hsv[y:y+bh, x:x+bw]
                if roi.size > 0:
                    mean_h, mean_s, mean_v, _ = cv2.mean(roi)
                    print(f"  [{name:8s}] pos=({cx:4d},{cy:4d}) area={area:6.1f} HSV=({mean_h:.0f},{mean_s:.0f},{mean_v:.0f})")

    cap.release()
    print("\nDone! Check the debug_*.png files.")


if __name__ == "__main__":
    main()
