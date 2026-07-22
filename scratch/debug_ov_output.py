import cv2
import numpy as np
import openvino as ov

core = ov.Core()
xml_path = "models/multicolor_detector_model_openvino_model/multicolor_detector_model.xml"
compiled_model = core.compile_model(xml_path, "GPU" if "GPU" in core.available_devices else "CPU")
infer_request = compiled_model.create_infer_request()

cap = cv2.VideoCapture("test_video_1.mp4")
ret, frame = cap.read()
h, w = frame.shape[:2]

resized = cv2.resize(frame, (640, 640))
input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32) / 255.0

infer_request.infer({0: input_tensor})
outputs = infer_request.get_output_tensor(0).data

print(f"Raw OpenVINO Output Shape: {outputs.shape}")
preds = outputs[0]
if preds.shape[0] < preds.shape[1]:
    preds = np.transpose(preds)
print(f"Transposed Predictions Shape: {preds.shape}")
print(f"First 5 predictions:\n{preds[:5]}")

# Filter high confidence boxes
high_conf = [p for p in preds if p[4] > 0.4]
print(f"High Conf (>0.4) count: {len(high_conf)}")
if len(high_conf) > 0:
    print(f"Top 3 High Conf Boxes:\n{high_conf[:3]}")
