import cv2
import os

video_path = r"D:\AI Projects\Golf AI\Training Videos\Golf All color videos\orange_right_1.mp4"
out_dir = r"scratch/stroke6_frames"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if 1140 <= frame_idx <= 1180:
        cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:04d}.png"), frame)

cap.release()
print(f"Saved frames 1140-1180 to {out_dir}")
