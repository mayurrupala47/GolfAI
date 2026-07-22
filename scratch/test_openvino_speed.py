import time
import cv2
from ultralytics import YOLO

print("Loading OpenVINO model targeting Intel GPU (Iris Xe)...")
model = YOLO('models/multicolor_detector_model_openvino_model', task='detect')

cap = cv2.VideoCapture('test_video_1.mp4')

print("Warming up OpenVINO GPU model...")
for _ in range(10):
    ret, frame = cap.read()
    if ret:
        model.predict(frame, device='GPU', verbose=False)

print("Benchmarking 200 frames on Intel Iris Xe GPU...")
start_time = time.time()
frame_count = 0

while frame_count < 200:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, device='GPU', verbose=False)
    frame_count += 1

elapsed = time.time() - start_time
fps = frame_count / elapsed

print(f"\n==========================================")
print(f"Hardware: Intel(R) Core i5-1135G7")
print(f"GPU: Intel(R) Iris(R) Xe Graphics (OpenVINO)")
print(f"Processed: {frame_count} frames in {elapsed:.2f} seconds")
print(f"Average Throughput: {fps:.2f} FPS")
print(f"==========================================\n")
cap.release()
