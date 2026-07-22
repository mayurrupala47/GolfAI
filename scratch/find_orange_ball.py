import cv2
import os
import numpy as np

frames_dir = r"scratch/stroke6_frames"
files = sorted(os.listdir(frames_dir))

img_path = os.path.join(frames_dir, files[0])
img = cv2.imread(img_path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Search for orange pixels across entire frame
orange_mask = (hsv[:,:,0] >= 5) & (hsv[:,:,0] < 28) & (hsv[:,:,1] > 60) & (hsv[:,:,2] > 60)
contours, _ = cv2.findContours(orange_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} orange contours in frame 1140:")
for c in contours:
    area = cv2.contourArea(c)
    if area > 10:
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w/2.0, y + h/2.0
        print(f"  Orange contour at ({cx:.1f}, {cy:.1f}) area={area:.1f} box=({x},{y},{w},{h})")
