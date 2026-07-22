import cv2
import numpy as np
import os
import json
import time

def main():
    source = "rtsp://10.30.7.125:8554/mystream"
    model_path = "yolo11n.onnx"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} not found in workspace!")
        return
        
    print("Connecting to RTSP stream...")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("ERROR: Cannot open RTSP stream!")
        return
        
    # Skip a few frames to stabilize
    print("Stabilizing camera feed...")
    for _ in range(10):
        cap.grab()
        
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("ERROR: Failed to read frame from stream!")
        return
        
    print(f"Successfully captured frame of size {frame.shape[1]}x{frame.shape[0]}")
    
    # Load ONNX net natively using OpenCV DNN
    print(f"Loading YOLOv11 ONNX model: {model_path}...")
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    h, w, _ = frame.shape
    
    # Let's try 640x640 and 320x320 resolutions
    for res in [640, 320]:
        print(f"\n--- Running inference at resolution {res}x{res} ---")
        t0 = time.time()
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (res, res), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()
        t1 = time.time()
        print(f"Inference took: {t1 - t0:.3f} seconds")
        
        output = outputs[0]
        output = np.transpose(output)  # Shape: (num_candidates, 84)
        
        # Look for class 32 (sports ball) and class 0 (person)
        candidates = []
        for row in output:
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            conf = classes_scores[class_id]
            
            if conf >= 0.01: # Check everything above 1% confidence
                candidates.append((class_id, conf, row[0], row[1], row[2], row[3]))
                
        # Filter and print results
        person_detections = []
        ball_detections = []
        
        for class_id, conf, cx, cy, bw, bh in candidates:
            if class_id == 0: # person
                person_detections.append((conf, cx, cy, bw, bh))
            elif class_id == 32: # sports ball
                ball_detections.append((conf, cx, cy, bw, bh))
                
        print(f"Found {len(person_detections)} person candidates (conf >= 0.01)")
        print(f"Found {len(ball_detections)} sports ball candidates (conf >= 0.01)")
        
        # Print top 5 person candidates
        person_detections.sort(reverse=True, key=lambda x: x[0])
        for i, det in enumerate(person_detections[:5]):
            conf, cx, cy, bw, bh = det
            # Scale back to original frame size
            cx_orig = cx * (w / res)
            cy_orig = cy * (h / res)
            print(f"  Person #{i+1}: Conf={conf:.3f} at ({cx_orig:.1f}, {cy_orig:.1f})")
            
        # Print top 10 sports ball candidates
        ball_detections.sort(reverse=True, key=lambda x: x[0])
        for i, det in enumerate(ball_detections[:10]):
            conf, cx, cy, bw, bh = det
            cx_orig = cx * (w / res)
            cy_orig = cy * (h / res)
            print(f"  Sports Ball #{i+1}: Conf={conf:.3f} at ({cx_orig:.1f}, {cy_orig:.1f})")

if __name__ == "__main__":
    main()
