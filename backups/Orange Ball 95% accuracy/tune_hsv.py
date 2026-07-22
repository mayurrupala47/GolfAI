import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture("test_orange_ball.mp4")
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=10, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    frame_idx = 0
    ball_hsv_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fg_mask = back_sub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # A true ball contour should have area between 15 and 80 pixels at 1080p
            if 12 <= area <= 120:
                x, y, bw, bh = cv2.boundingRect(cnt)
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
                
                # Check for roundness
                if circularity >= 0.50:
                    roi = hsv_frame[y:y+bh, x:x+bw]
                    if roi.size > 0:
                        mean_hsv = cv2.mean(roi)
                        # We print all rounded small contours to find the ball's actual path
                        ball_hsv_log.append((frame_idx, x + bw/2.0, y + bh/2.0, area, circularity, mean_hsv))

        frame_idx += 1
    cap.release()
    
    # Save the log to a text file for review
    with open("ball_hsv_tuning_log.txt", "w") as f:
        f.write("Frame, X, Y, Area, Circularity, Hue, Saturation, Value\n")
        for log in ball_hsv_log:
            f.write(f"{log[0]}, {log[1]:.1f}, {log[2]:.1f}, {log[3]:.1f}, {log[4]:.2f}, {log[5][0]:.1f}, {log[5][1]:.1f}, {log[5][2]:.1f}\n")
            
    print(f"Logged {len(ball_hsv_log)} candidate points to ball_hsv_tuning_log.txt")

if __name__ == "__main__":
    main()
