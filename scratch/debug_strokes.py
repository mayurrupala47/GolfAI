import cv2
import os
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_strokes")

from ai.opencv_detector import OpenCVBallDetector
from ai.kalman_tracker import KalmanBallTracker
from ai.motion import MotionAnalyzer
from engine.state_machine import BallStateMachine, BallState
from main import load_config

def main():
    config = load_config("config/config.yaml")
    video_path = "assets/input.mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"Video not found at {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Loaded video: {frame_width}x{frame_height} @ {fps} FPS. Total frames: {total_frames}")
    
    # Scale pixels per meter
    resize_width = config.get("processing", {}).get("resize_width", 1280)
    scale = resize_width / frame_width
    original_ppm = config.get("video", {}).get("pixels_per_meter", 200.0)
    scaled_ppm = original_ppm * scale
    if "video" not in config:
        config["video"] = {}
    config["video"]["pixels_per_meter"] = scaled_ppm
    
    detector = OpenCVBallDetector(config)
    tracker = KalmanBallTracker(config, fps=int(fps))
    motion_analyzer = MotionAnalyzer(smoothing_window=5)
    
    state_machines = {}
    
    frame_idx = 0
    while cap.isOpened() and frame_idx < 500:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame
        h_new = int(frame_height * scale)
        proc_frame = cv2.resize(frame, (resize_width, h_new))
        
        # 1. Detect
        detections = detector.detect(proc_frame)
        
        # 2. Track
        track_states = {tid: sm.state.value for tid, sm in state_machines.items()}
        tracks = tracker.update(detections, proc_frame, track_states=track_states)
        
        # 3. Analyze and State transitions
        active_tids = []
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            active_tids.append(track_id)
            
            metrics = motion_analyzer.update(track_id, (x1, y1, x2, y2), fps, scaled_ppm)
            metrics["frame_index"] = frame_idx
            
            if track_id not in state_machines:
                state_machines[track_id] = BallStateMachine(track_id, config)
                
            sm = state_machines[track_id]
            old_state = sm.state
            new_state, stroke_detected = sm.update(metrics)
            
            # Print important events or transitions
            if old_state != new_state or stroke_detected or (frame_idx >= 210 and frame_idx <= 245):
                logger.info(
                    f"Frame {frame_idx:03d} | Ball {track_id} | Detections: {len(detections)} | "
                    f"Pos: ({metrics['x']:.1f}, {metrics['y']:.1f}) | Speed: {metrics['speed']:.3f} m/s | "
                    f"Accel: {metrics['acceleration']:.3f} m/s^2 | Dist: {sm.distance_traveled_since_ready:.3f} m | "
                    f"State: {old_state.value} -> {new_state.value} | Stroke: {stroke_detected}"
                )
                
            if stroke_detected:
                logger.info(f"***** STROKE DETECTED at Frame {frame_idx}! *****")
                
        frame_idx += 1
        
    cap.release()

if __name__ == "__main__":
    main()
