import cv2
import time
import os
import sys

# Add root dir to path so we can import ai module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.tracknet_tracker import TrackNetEngine

engine = TrackNetEngine(weights_path='models/TrackNet_best.pt', conf_threshold=0.1)
cap = cv2.VideoCapture('orange_color_ball.mp4')

print("\nStarting TrackNet inference on video...")
frames = 0
found = 0
t0 = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    
    pos, conf = engine.update(frame)
    if pos:
        found += 1
    
    frames += 1
    if frames % 100 == 0:
        print(f"Processed {frames} frames...")
        
    # Just run 500 frames for a quick local test
    if frames >= 500:
        break

cap.release()
t_el = time.time() - t0
print(f"\n[OK] TrackNet processed {frames} frames in {t_el:.2f}s ({frames/t_el:.1f} fps).")
print(f"[OK] Detected ball in {found}/{frames} frames ({found/frames*100:.1f}%).")
