import time
import cv2
import numpy as np
import openvino as ov

# Initialize OpenVINO runtime
core = ov.Core()
devices = core.available_devices
print(f"OpenVINO Detected Devices: {devices}")

# Select GPU if available, else CPU
target_device = "GPU" if "GPU" in devices else "CPU"
print(f"Targeting Device: {target_device}")

# Load model XML/BIN
model_path = "models/multicolor_detector_model_openvino_model/multicolor_detector_model.xml"
compiled_model = core.compile_model(model_path, target_device)
infer_request = compiled_model.create_infer_request()

cap = cv2.VideoCapture('test_video_1.mp4')

print("Warming up OpenVINO inference pipeline...")
for _ in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    # Preprocess frame to 640x640 float32 normalized
    resized = cv2.resize(frame, (640, 640))
    input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32) / 255.0
    infer_request.infer({0: input_tensor})

print("Benchmarking 200 frames using OpenVINO Native Engine...")
start_time = time.time()
frame_count = 0

while frame_count < 200:
    ret, frame = cap.read()
    if not ret:
        break
    resized = cv2.resize(frame, (640, 640))
    input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32) / 255.0
    
    # Run hardware inference on GPU
    infer_request.infer({0: input_tensor})
    output = infer_request.get_output_tensor(0).data
    frame_count += 1

elapsed = time.time() - start_time
fps = frame_count / elapsed

print(f"\n==========================================")
print(f"Target Device: {target_device}")
print(f"Processed: {frame_count} frames in {elapsed:.2f} seconds")
print(f"Average Inference Speed: {fps:.2f} FPS")
print(f"==========================================\n")
cap.release()
