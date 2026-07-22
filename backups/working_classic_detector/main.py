import os
import argparse
import yaml
import logging
import cv2
import gc
from typing import Dict, Any

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")

from ai.detector import YoloBallDetector, MockBallDetector, ClassicBallDetector
from ai.opencv_detector import OpenCVBallDetector
from ai.tracker import ByteBallTracker
from ai.kalman_tracker import KalmanBallTracker
from ai.motion import MotionAnalyzer
from mqtt.publisher import MqttPublisher, MockMqttPublisher
from exporters.csv_export import CsvExporter
from exporters.json_export import JsonExporter
from engine.stroke_engine import StrokeEngine
from visualization.overlay import Visualizer


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads variables from configuration file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}. Using default settings.")
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Error loading yaml config: {e}. Using defaults.")
        return {}

def main():
    # Enable OpenCV CPU/thread optimizations
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)

    parser = argparse.ArgumentParser(description="Mini Golf AI Stroke Detection POC")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml", 
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--mock-mqtt", 
        action="store_true", 
        help="Use a mock MQTT publisher that only prints to console"
    )
    parser.add_argument(
        "--end-frame", 
        type=int, 
        default=-1, 
        help="Frame index to end processing (exclusive)"
    )
    parser.add_argument(
        "--save-debug", 
        action="store_true", 
        help="Draw visual overlays and save all processed frames to disk"
    )
    parser.add_argument(
        "--frame-skip", 
        type=int, 
        default=1, 
        help="Process every N-th frame (1=all, 2=every second frame, etc.)"
    )
    args = parser.parse_args()

    # 1. Load config
    logger.info("Initializing Mini Golf AI Stroke Detection...")
    config = load_config(args.config)

    # 2. Extract configuration fields
    video_input_path = "assets/input.mp4"
    video_output_path = "outputs/output.mp4"
    
    # Ensure correct folders exist
    os.makedirs(os.path.dirname(os.path.abspath(video_output_path)), exist_ok=True)
    
    # Exporters configuration
    csv_output_path = "outputs/strokes.csv"
    json_output_path = "outputs/strokes.json"
    
    # 3. Instantiate concrete implementations based on interfaces (Dependency Injection)
    # Detector setup - Use the classic detector as requested
    detector_type = "classic"
    logger.info("Configuring Classic CV/Contour Sizing Detector...")
    detector = OpenCVBallDetector(config)

    # Motion analyzer setup
    motion_analyzer = MotionAnalyzer(smoothing_window=5)

    # MQTT Publisher setup
    if args.mock_mqtt:
        logger.info("Configuring Mock MQTT Publisher...")
        mqtt_publisher = MockMqttPublisher(config)
    else:
        logger.info("Configuring Production MQTT Publisher...")
        mqtt_publisher = MqttPublisher(config)

    # CSV and JSON Exporters setup
    csv_exporter = CsvExporter(csv_output_path)
    json_exporter = JsonExporter(json_output_path)

    # 4. Open Video Stream
    logger.info(f"Opening input video stream: {video_input_path}")
    if not os.path.exists(video_input_path):
        logger.error(f"Input video file not found at {video_input_path}")
        return

    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_input_path}")
        return

    # Read Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps <= 0:
        fps = 30.0  # Fallback
        
    logger.info(f"Video loaded: {frame_width}x{frame_height} @ {fps:.2f} FPS. Total frames: {total_frames}")

    # Adjust pixels_per_meter based on resizing
    resize_width = config.get("processing", {}).get("resize_width", 1280)
    if resize_width > 0 and frame_width != resize_width:
        scale = resize_width / frame_width
        original_ppm = config.get("video", {}).get("pixels_per_meter", 200.0)
        scaled_ppm = original_ppm * scale
        if "video" not in config:
            config["video"] = {}
        config["video"]["pixels_per_meter"] = scaled_ppm
        logger.info(f"Scaled pixels_per_meter from {original_ppm} to {scaled_ppm} for processing width {resize_width}")

    # Tracker setup (depends on FPS)
    logger.info("Configuring Kalman Ball Tracker...")
    tracker = KalmanBallTracker(config, fps=int(fps))

    # 5. Initialize Stroke Engine (Decoupled Coordinator)
    stroke_engine = StrokeEngine(
        config=config,
        detector=detector,
        tracker=tracker,
        motion_analyzer=motion_analyzer,
        mqtt_publisher=mqtt_publisher,
        csv_exporter=csv_exporter,
        json_exporter=json_exporter
    )

    # 6. Initialize Visualizer Overlay
    visualizer = Visualizer(motion_analyzer=motion_analyzer)

    # 7. Ensure output image directories exist and are cleared of previous data
    frames_dir = "outputs/frames"
    strokes_dir = "outputs/strokes"
    import shutil
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    if os.path.exists(strokes_dir):
        shutil.rmtree(strokes_dir)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(strokes_dir, exist_ok=True)

    # Connect MQTT
    mqtt_publisher.connect()

    frame_idx = 0
    start_time = cv2.getTickCount()

    try:
        while True:
            if args.end_frame != -1 and frame_idx >= args.end_frame:
                break
                
            if frame_idx % int(fps * 2) == 0 or frame_idx == 1:
                progress_info = f"({frame_idx / total_frames * 100:.1f}%)" if total_frames > 0 else ""
                logger.info(f"Processing progress: {frame_idx}/{total_frames} frames {progress_info}")
            
            if args.frame_skip > 1 and frame_idx % args.frame_skip != 0:
                # Use cap.grab() to skip decoding the video frame image data entirely
                ret = cap.grab()
                if not ret:
                    break
                frame_idx += 1
                continue
                
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_idx / fps
            
            # Resize frame for processing if specified
            resize_width = config.get("processing", {}).get("resize_width", 1280)
            if resize_width > 0 and frame_width != resize_width:
                scale = resize_width / frame_width
                h_new = int(frame_height * scale)
                proc_frame = cv2.resize(frame, (resize_width, h_new))
            else:
                proc_frame = frame
                
            # Run core processing pipeline
            active_metrics, active_states, stroke_events = stroke_engine.process_frame(
                frame=proc_frame,
                frame_idx=frame_idx,
                fps=fps / args.frame_skip,
                timestamp=timestamp
            )
            
            # Draw HUD overlays on frame and save only if save_debug is requested
            annotated_frame = None
            if args.save_debug:
                annotated_frame = visualizer.draw(proc_frame, active_metrics)
                if len(active_metrics) > 0:
                    frame_filename = os.path.join(frames_dir, f"frame_{frame_idx:04d}.jpg")
                    cv2.imwrite(frame_filename, annotated_frame)
                
            # Save specific stroke event frames
            for track_id, stroke_count in stroke_events:
                # Render overlay specifically for this frame if it wasn't already drawn
                stroke_annotated_frame = annotated_frame if args.save_debug else visualizer.draw(proc_frame, active_metrics)
                sm = stroke_engine.state_machines.get(track_id)
                frame_label = sm.last_impact_frame if (sm and sm.last_impact_frame is not None) else frame_idx
                stroke_filename = os.path.join(strokes_dir, f"stroke_ball_{track_id}_stroke_{stroke_count}_frame_{frame_label:04d}.jpg")
                cv2.imwrite(stroke_filename, stroke_annotated_frame)
                logger.info(f"Saved stroke event frame: {stroke_filename}")
            
            frame_idx += 1
            if frame_idx % 10 == 0:
                gc.collect()
                
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.error(f"Exception during processing loop: {e}", exc_info=True)
    finally:
        # Cleanup handles
        cap.release()
        mqtt_publisher.disconnect()
        
        # Calculate execution time
        end_time = cv2.getTickCount()
        elapsed_sec = (end_time - start_time) / cv2.getTickFrequency()
        logger.info("Processing complete.")
        
        # Print summary
        logger.info("========================================")
        logger.info("        Execution Summary (POC)         ")
        logger.info("========================================")
        logger.info(f"Processed frames     : {frame_idx}")
        logger.info(f"Processing time      : {elapsed_sec:.2f} seconds")
        logger.info(f"Average FPS          : {frame_idx / elapsed_sec:.2f}" if elapsed_sec > 0 else "N/A")
        logger.info(f"Annotated Frames Output : {os.path.abspath(frames_dir)}")
        logger.info(f"Stroke Events Output    : {os.path.abspath(strokes_dir)}")
        logger.info(f"CSV Logs Output         : {os.path.abspath(csv_output_path)}")
        logger.info(f"JSON Logs Output        : {os.path.abspath(json_output_path)}")
        
        # Show final stroke counts for all tracked balls
        for track_id, sm in stroke_engine.state_machines.items():
            logger.info(f"Ball ID {track_id:2d}: {sm.stroke_count} strokes recorded. Final State: {sm.state.value}")
        logger.info("========================================")


if __name__ == "__main__":
    main()
