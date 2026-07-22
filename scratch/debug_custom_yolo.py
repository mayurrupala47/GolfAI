import cv2
import numpy as np
import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python scratch/debug_custom_yolo.py <path_to_image.jpg>")
        return
        
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        return
        
    model_path = "models/custom_ball_detector.onnx"
    # Try different fallbacks for the model path
    if not os.path.exists(model_path):
        for path in ["models/best.onnx", "best.onnx", "yolo11n.onnx"]:
            if os.path.exists(path):
                model_path = path
                break
                
    print(f"Using model: {model_path}")
    if not os.path.exists(model_path):
        print("ERROR: ONNX model file not found!")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Failed to load image: {image_path}")
        return
        
    h, w = frame.shape[:2]
    print(f"Loaded image of size {w}x{h}")
    
    # Load model
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # Run inference at 320x320
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    
    outputs = net.forward()
    output = outputs[0]
    output = np.transpose(output)
    
    print("\nAnalyzing raw detections...")
    print(f"Output shape after transpose: {output.shape}")
    
    detections = []
    for row in output:
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        conf = classes_scores[class_id]
        
        if conf > 0.001:  # Capture even extremely low confidences for diagnostics
            detections.append((class_id, conf, row[0], row[1], row[2], row[3]))
            
    print(f"Total detections with conf > 0.001: {len(detections)}")
    
    # Sort by confidence
    detections.sort(reverse=True, key=lambda x: x[1])
    
    # Draw top 10 on the frame
    display = frame.copy()
    print("\n--- TOP DETECTIONS ---")
    for i, det in enumerate(detections[:10]):
        class_id, conf, cx_s, cy_s, w_s, h_s = det
        
        # Scale back to original frame size
        cx = cx_s * (w / 320.0)
        cy = cy_s * (h / 320.0)
        bw = w_s * (w / 320.0)
        bh = h_s * (h / 320.0)
        
        x1 = int(cx - bw/2.0)
        y1 = int(cy - bh/2.0)
        x2 = int(cx + bw/2.0)
        y2 = int(cy + bh/2.0)
        
        print(f"#{i+1}: Class={class_id}, Conf={conf:.4f} at bbox=({x1},{y1},{x2-x1},{y2-y1})")
        
        # Draw green boxes for top predictions
        color = (0, 255, 0) if i == 0 else (0, 165, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, f"Cls {class_id} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
    output_filename = "debug_custom_yolo_output.png"
    cv2.imwrite(output_filename, display)
    print(f"\nDiagnostic image saved as: {output_filename}")

if __name__ == "__main__":
    main()
