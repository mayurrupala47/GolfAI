import sys
import time
import math
import cv2
import numpy as np
import torch
from collections import deque, Counter

# Import TrackNet utilities
from utils.general import get_model, WIDTH, HEIGHT, draw_traj
from test import predict_location
from predict_fast_track import count_strokes, _extract_ball_pixels, _classify_ball_color

def main():
    print("==================================================")
    print("⛳ TrackNet Live Golf Analytics - IP Stream ⛳")
    print("==================================================")
    
    # 1. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # 2. Load Model
    ckpt_path = "ckpts/TrackNet_best.pt"
    print(f"[INFO] Loading model from {ckpt_path}...")
    
    model = get_model('TrackNet', seq_len=8, bg_mode='concat').to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        sys.exit(1)
        
    model.eval()
    print("[INFO] Model loaded successfully.")

    # 3. Open Camera
    camera_url = "http://10.30.7.125:5000"
    print(f"[INFO] Connecting to camera feed: {camera_url}...")
    cap = cv2.VideoCapture(camera_url)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at {camera_url}.")
        print("Check if you need to add an endpoint like /video_feed")
        sys.exit(1)
        
    # --- Live Tracking State ---
    seq_len = 8
    frame_buffer = deque(maxlen=seq_len)
    raw_buffer = deque(maxlen=seq_len)
    
    traj_len = 8
    pred_queue = deque(maxlen=traj_len)

    # --- Analytics State ---
    # We maintain a growing history of coordinates for the stroke detector
    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    frame_idx = 0
    current_stroke_count = 0
    
    # Color classification state
    color_votes = Counter()
    COLOR_LOCK_VOTES = 15
    color_locked = False
    current_ball_color = "Detecting..."
    
    # Optional: If you ever want to add a hole position to detect Hole-outs,
    # you can set h_pos = (x, y) here.
    h_pos = None 

    print("\n[INFO] Starting live feed. Press 'q' to quit.")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera disconnected or failed to grab frame.")
                break
                
            frame_idx += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_buffer.append(frame)
            
            # Preprocess the frame
            resized = cv2.resize(frame_rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            transposed = np.moveaxis(resized, -1, 0)
            frame_buffer.append(transposed)
            
            # Only run inference if we have a full sequence of 8 frames
            if len(frame_buffer) == seq_len:
                start_time = time.time()
                
                input_tensor = np.concatenate(frame_buffer, axis=0)
                input_tensor = input_tensor.astype(np.float32) / 255.0
                x = torch.from_numpy(input_tensor).unsqueeze(0).to(device)
                
                y_pred = model(x)
                heatmap = (y_pred[0][0].cpu().numpy() > 0.5).astype(np.uint8) * 255
                hx, hy, hw, hh = predict_location(heatmap)
                
                fps = 1.0 / (time.time() - start_time)
                
                display_frame = raw_buffer[-1].copy()
                clean_frame = display_frame.copy()
                orig_h, orig_w = display_frame.shape[:2]
                
                vis = 0
                center_x, center_y = 0, 0
                
                if hw > 0 and hh > 0:
                    vis = 1
                    center_x = int((hx + hw // 2) * (orig_w / WIDTH))
                    center_y = int((hy + hh // 2) * (orig_h / HEIGHT))
                    
                # Update history for strokes
                pred_dict['Frame'].append(frame_idx)
                pred_dict['X'].append(center_x)
                pred_dict['Y'].append(center_y)
                pred_dict['Visibility'].append(vis)
                
                # Run the Stroke Counting logic silently
                import os
                with open(os.devnull, 'w') as devnull:
                    original_stdout = sys.stdout
                    sys.stdout = devnull
                    try:
                        strokes = count_strokes(pred_dict, save_file=None)
                        current_stroke_count = len(strokes)
                    except Exception:
                        pass
                    finally:
                        sys.stdout = original_stdout
                
                # Queue for yellow trajectory tail
                if len(pred_queue) >= traj_len:
                    pred_queue.pop()
                if vis:
                    pred_queue.appendleft([center_x, center_y])
                else:
                    pred_queue.appendleft(None)
                    
                # Draw the trailing yellow line
                display_frame = draw_traj(display_frame, pred_queue, color='yellow')
                
                # Run Color Detection
                if vis and not color_locked:
                    px, bg_px = _extract_ball_pixels(clean_frame, center_x, center_y, radius=9)
                    c = _classify_ball_color(px, bg_px, sat_cut=60, sat_fraction=0.20)
                    if c:
                        color_votes[c] += 1
                        current_ball_color = color_votes.most_common(1)[0][0]
                        if sum(color_votes.values()) >= COLOR_LOCK_VOTES:
                            color_locked = True
                
                # --- DRAW OVERLAYS ---
                # 1. Dark Top-Left Box for Score/Color
                cv2.rectangle(display_frame, (20, 20), (360, 140), (0, 0, 0), -1)
                
                # 2. Score text
                cv2.putText(display_frame, f"STROKES: {current_stroke_count}", (40, 75), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
                            
                # 3. Color text
                cv2.putText(display_frame, f"COLOR: {current_ball_color}", (40, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
                            
                # 4. Ball indicator
                if vis:
                    cv2.circle(display_frame, (center_x, center_y), 10, (0, 0, 255), -1)
                            
                # 5. Bottom left FPS
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, orig_h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                cv2.imshow("TrackNet Live - IP Stream", display_frame)
                
            else:
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Buffering... {len(frame_buffer)}/{seq_len}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("TrackNet Live - IP Stream", display_frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
