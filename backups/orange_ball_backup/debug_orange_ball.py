import cv2
import numpy as np
import os
import json

def main():
    video_path = "test_orange_ball.mp4"
    calibration_path = "config/calibration.json"
    
    if not os.path.exists(video_path):
        print(f"Video file not found at {video_path}")
        return

    # Load calibration coordinates for masking
    mask = None
    if os.path.exists(calibration_path):
        with open(calibration_path, "r") as f:
            cal = json.load(f)
    else:
        cal = {}

    cap = cv2.VideoCapture(video_path)
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=10, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    frame_idx = 0
    print("Analyzing test_orange_ball.mp4 for orange color contours...")

    # We collect all candidate contours that pass shape checks to summarize the HSV range
    hsv_candidates = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        
        # Initialize mask scaled to frame dimensions
        if mask is None:
            mask = np.ones((h, w), dtype=np.uint8) * 255
            playable_area = cal.get("playable_area", [])
            if len(playable_area) >= 3:
                base_res = cal.get("source_resolution", [3840, 2160])
                scale_x = w / base_res[0]
                scale_y = h / base_res[1]
                pts = np.array([[int(p[0] * scale_x), int(p[1] * scale_y)] for p in playable_area], dtype=np.int32)
                pa_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(pa_mask, [pts], 255)
                mask = cv2.bitwise_and(mask, pa_mask)
                
            # Apply ignore zones (except Tees so we can see the ball at start)
            for region in cal.get("ignore_regions", []):
                if "tee" in region.get("name", "").lower():
                    continue
                if region.get("type") == "polygon" or "points" in region:
                    poly_pts = region["points"]
                    scale_x = w / base_res[0]
                    scale_y = h / base_res[1]
                    scaled_pts = np.array([[int(p[0] * scale_x), int(p[1] * scale_y)] for p in poly_pts], dtype=np.int32)
                    cv2.fillPoly(mask, [scaled_pts], 0)
                else:
                    scale_x = w / base_res[0]
                    scale_y = h / base_res[1]
                    rx = int(region["x"] * scale_x)
                    ry = int(region["y"] * scale_y)
                    rr = int(region["radius"] * min(scale_x, scale_y))
                    cv2.circle(mask, (rx, ry), rr, 0, -1)

        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        fg_mask = back_sub.apply(masked_frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Find moderate-sized contours (potentially our ball)
            if 15 <= area <= 1000:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cx = x + bw / 2.0
                cy = y + bh / 2.0
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
                avg_diam = (bw + bh) / 2.0
                aspect_ratio = float(bw) / bh

                if 0.4 <= aspect_ratio <= 2.2 and circularity >= 0.4:
                    # Get average HSV value of the contour
                    roi = frame[y:y+bh, x:x+bw]
                    if roi.size > 0:
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        mean_hsv = cv2.mean(hsv_roi)
                        
                        hsv_candidates.append({
                            "frame": frame_idx,
                            "area": area,
                            "diam": avg_diam,
                            "circ": circularity,
                            "hsv": [mean_hsv[0], mean_hsv[1], mean_hsv[2]],
                            "pos": [cx, cy]
                        })

        frame_idx += 1

    cap.release()
    print(f"Analysis complete. Found {len(hsv_candidates)} contour candidates.")
    
    # Analyze the HSV footprint of the candidate contours
    if len(hsv_candidates) > 0:
        print("\n--- Detected Contour Profiles (Top 25 sorted by circularity) ---")
        hsv_candidates.sort(key=lambda c: c["circ"], reverse=True)
        for i, c in enumerate(hsv_candidates[:25]):
            print(f"Candidate #{i+1:02d}: Frame={c['frame']:3d} | Pos=({c['pos'][0]:.1f}, {c['pos'][1]:.1f}) | Area={c['area']:5.1f} | Diam={c['diam']:4.1f} | Circ={c['circ']:.2f} | HSV=({c['hsv'][0]:.1f}, {c['hsv'][1]:.1f}, {c['hsv'][2]:.1f})")

        # Group candidates by HUE to find the dominant color range
        hues = [c["hsv"][0] for c in hsv_candidates]
        sats = [c["hsv"][1] for c in hsv_candidates]
        vals = [c["hsv"][2] for c in hsv_candidates]
        
        print("\n--- Summary Statistics for Potential Detections ---")
        print(f"Hue Range       : Min={min(hues):.1f}, Max={max(hues):.1f}, Mean={np.mean(hues):.1f}")
        print(f"Saturation Range: Min={min(sats):.1f}, Max={max(sats):.1f}, Mean={np.mean(sats):.1f}")
        print(f"Value Range     : Min={min(vals):.1f}, Max={max(vals):.1f}, Mean={np.mean(vals):.1f}")
    else:
        print("No candidates detected. MOG2 did not segment any contours matching a ball profile.")

if __name__ == "__main__":
    main()
