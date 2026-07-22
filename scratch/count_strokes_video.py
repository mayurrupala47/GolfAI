import cv2
import os

video_path = r"D:\AI Projects\Golf AI\Training Videos\Golf All color videos\orange_right_1.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"Video: {video_path}")
print(f"Total Frames: {total_frames}, FPS: {fps}, Duration: {duration:.1f} seconds")
cap.release()
