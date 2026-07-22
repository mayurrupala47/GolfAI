"""
Live Color Diagnostic Tool
===========================
Connects to the RTSP stream and shows:
  - Raw frame
  - Current orange HSV color mask (with current thresholds)
  - Broadened HSV color mask (wider hue/sat/val range)
  - HSV pixel heatmap around the Tee area

Usage:
  python debug_live_color.py                      # Uses default RTSP URL
  python debug_live_color.py rtsp://192.168.1.5/live
  python debug_live_color.py test_orange_ball.mp4  # Also works with video files

Press 'q' or ESC to quit, 's' to save a snapshot, 'c' to print HSV stats of the tee region.
"""

import cv2
import numpy as np
import sys
import json
import os

def main():
    # Default RTSP URL - change as needed
    source = sys.argv[1] if len(sys.argv) > 1 else "rtsp://admin:CCFCGK@192.168.1.108:554/Streaming/channels/101"

    print(f"Connecting to: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Stream opened: {w}x{h} @ {fps:.1f} FPS")

    # Load calibration to find Tee point
    tee_point = None
    cal_path = "config/calibration.json"
    if os.path.exists(cal_path):
        with open(cal_path, "r") as f:
            cal = json.load(f)
        base_res = cal.get("source_resolution", [1920, 1080])
        for region in cal.get("ignore_regions", []):
            if "tee" in region.get("name", "").lower():
                tee_point = (region["x"], region["y"])
                print(f"Tee point (calibration): {tee_point} at base resolution {base_res}")
                break

    # Current thresholds from opencv_detector.py
    CURRENT_LOWER = np.array([3, 75, 25])
    CURRENT_UPPER = np.array([22, 255, 255])

    # Broadened thresholds for discovery
    BROAD_LOWER = np.array([0, 30, 30])
    BROAD_UPPER = np.array([30, 255, 255])

    # Extra-broad (even yellow-ish / brownish tones)
    XBROAD_LOWER = np.array([0, 15, 15])
    XBROAD_UPPER = np.array([40, 255, 255])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    frame_count = 0
    print("\nControls:")
    print("  q/ESC = quit")
    print("  s     = save current frame snapshot")
    print("  c     = print HSV stats around tee area")
    print("  1     = show current thresholds (narrow)")
    print("  2     = show broadened thresholds")
    print("  3     = show extra-broad thresholds")
    print("")

    mode = 1  # Start with current thresholds
    mode_names = {1: "CURRENT [H:3-22, S:75-255, V:25-255]",
                  2: "BROAD [H:0-30, S:30-255, V:30-255]",
                  3: "X-BROAD [H:0-40, S:15-255, V:15-255]"}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame, retrying...")
            cv2.waitKey(100)
            continue

        # Resize for processing (match what main.py does)
        proc_w = 1280
        if w != proc_w:
            scale = proc_w / w
            h_new = int(h * scale)
            proc_frame = cv2.resize(frame, (proc_w, h_new))
        else:
            proc_frame = frame.copy()

        ph, pw = proc_frame.shape[:2]

        # Convert to HSV
        hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)

        # Select mask based on mode
        if mode == 1:
            lower, upper = CURRENT_LOWER, CURRENT_UPPER
        elif mode == 2:
            lower, upper = BROAD_LOWER, BROAD_UPPER
        else:
            lower, upper = XBROAD_LOWER, XBROAD_UPPER

        color_mask = cv2.inRange(hsv, lower, upper)
        color_mask_clean = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(color_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw on a copy
        display = proc_frame.copy()

        # Scale tee point to processing resolution
        tee_scaled = None
        if tee_point is not None:
            base_res = cal.get("source_resolution", [1920, 1080])
            sx = pw / base_res[0]
            sy = ph / base_res[1]
            tee_scaled = (int(tee_point[0] * sx), int(tee_point[1] * sy))
            # Draw tee circle
            cv2.circle(display, tee_scaled, 40, (0, 255, 255), 2)
            cv2.putText(display, "TEE", (tee_scaled[0] - 15, tee_scaled[1] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw all contours with area info
        ball_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx = x + bw // 2
            cy = y + bh // 2

            if area >= 3:
                color = (0, 255, 0) if 6 <= area <= 200 else (0, 165, 255)
                cv2.drawContours(display, [cnt], -1, color, 2)
                cv2.putText(display, f"A:{area:.0f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

                # Get mean HSV of contour region
                roi = hsv[y:y+bh, x:x+bw]
                if roi.size > 0:
                    mean_h, mean_s, mean_v, _ = cv2.mean(roi)
                    cv2.putText(display, f"H:{mean_h:.0f} S:{mean_s:.0f} V:{mean_v:.0f}",
                                (x, y + bh + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                    if 6 <= area <= 200:
                        ball_candidates.append({
                            "pos": (cx, cy), "area": area,
                            "hsv": (mean_h, mean_s, mean_v)
                        })

        # Mode label
        cv2.putText(display, f"Mode {mode}: {mode_names[mode]}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Contours: {len(contours)} | Ball-sized: {len(ball_candidates)}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Create mask visualization (colored)
        mask_colored = cv2.cvtColor(color_mask_clean, cv2.COLOR_GRAY2BGR)
        mask_colored[:, :, 1] = 0  # Remove green channel for contrast
        mask_colored[:, :, 0] = 0  # Remove blue channel => red mask

        # Stack display and mask side-by-side
        combined = np.hstack([display, mask_colored])

        # Resize combined for display
        disp_w = min(1600, combined.shape[1])
        disp_scale = disp_w / combined.shape[1]
        combined_resized = cv2.resize(combined, (disp_w, int(combined.shape[0] * disp_scale)))

        cv2.imshow("Live Color Diagnostic", combined_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"debug_snapshot_{frame_count:04d}.png"
            cv2.imwrite(fname, combined)
            print(f"Saved snapshot: {fname}")
        elif key == ord('c'):
            # Print HSV stats around tee region
            if tee_scaled:
                tx, ty = tee_scaled
                r = 50
                x1 = max(0, tx - r)
                y1 = max(0, ty - r)
                x2 = min(pw, tx + r)
                y2 = min(ph, ty + r)
                roi = hsv[y1:y2, x1:x2]
                if roi.size > 0:
                    print(f"\n=== HSV Stats around Tee ({x1},{y1})-({x2},{y2}) ===")
                    print(f"  Hue:  min={roi[:,:,0].min()}, max={roi[:,:,0].max()}, mean={roi[:,:,0].mean():.1f}")
                    print(f"  Sat:  min={roi[:,:,1].min()}, max={roi[:,:,1].max()}, mean={roi[:,:,1].mean():.1f}")
                    print(f"  Val:  min={roi[:,:,2].min()}, max={roi[:,:,2].max()}, mean={roi[:,:,2].mean():.1f}")
                    
                    # Also show how many pixels pass each threshold
                    for name, lo, hi in [("Current", CURRENT_LOWER, CURRENT_UPPER),
                                          ("Broad", BROAD_LOWER, BROAD_UPPER),
                                          ("X-Broad", XBROAD_LOWER, XBROAD_UPPER)]:
                        m = cv2.inRange(roi, lo, hi)
                        total = m.size
                        passing = np.count_nonzero(m)
                        print(f"  {name:8s} threshold: {passing}/{total} pixels pass ({passing/total*100:.1f}%)")
                else:
                    print("ROI around tee is empty!")
            else:
                print("No tee point loaded from calibration.")
        elif key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('3'):
            mode = 3

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames total.")


if __name__ == "__main__":
    main()
