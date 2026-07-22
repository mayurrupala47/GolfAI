from typing import Dict, Any, List, Tuple
import logging
import time

from ai.interfaces import IBallDetector, IBallTracker, IMotionAnalyzer, IMqttPublisher
from engine.state_machine import BallStateMachine, BallState
from exporters.csv_export import CsvExporter
from exporters.json_export import JsonExporter

logger = logging.getLogger(__name__)

class StrokeEngine:
    """
    Coordinating engine that drives the detection, tracking, motion analysis, 
    state transitions, and event publishing pipeline.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        detector: IBallDetector,
        tracker: IBallTracker,
        motion_analyzer: IMotionAnalyzer,
        mqtt_publisher: IMqttPublisher,
        csv_exporter: CsvExporter,
        json_exporter: JsonExporter
    ):
        self.config = config
        self.detector = detector
        self.tracker = tracker
        self.motion_analyzer = motion_analyzer
        self.mqtt_publisher = mqtt_publisher
        self.csv_exporter = csv_exporter
        self.json_exporter = json_exporter
        
        # Pull configuration parameters
        self.camera_id = config.get("camera", {}).get("id", "CAM-01")
        self.hole = config.get("camera", {}).get("hole", 1)
        self.pixels_per_meter = config.get("video", {}).get("pixels_per_meter", 200.0)
        
        # State machines mapped by track_id
        self.state_machines: Dict[int, BallStateMachine] = {}

    def process_frame(self, frame, frame_idx: int, fps: float, timestamp: float) -> Tuple[List[Dict[str, Any]], Dict[int, BallState]]:
        """
        Processes a single video frame. Runs detector, tracker, motion analyzer, 
        and updates state machines. Publishes stroke events if detected.
        
        Args:
            frame: Numpy array representing the frame.
            frame_idx: Index of the current frame.
            fps: Frame rate of the video.
            timestamp: Elapsed time in seconds.
            
        Returns:
            A tuple containing:
            1. A list of active ball metrics.
            2. A dictionary of active ball states mapping track_id -> BallState.
        """
        # 1. Object Detection
        detections = self.detector.detect(frame)
        
        # 2. Multi-Object Tracking
        track_states = {tid: sm.state.value for tid, sm in self.state_machines.items()}
        tracks = self.tracker.update(detections, frame, track_states=track_states)
        
        active_ball_metrics = []
        active_ball_states = {}
        stroke_events = []
        
        # Keep track of active track IDs in this frame
        active_track_ids = set()
        
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            active_track_ids.add(track_id)
            
            # 3. Motion Analysis
            metrics = self.motion_analyzer.update(track_id, (x1, y1, x2, y2), fps, self.pixels_per_meter)
            metrics["track_id"] = track_id
            metrics["bbox"] = (x1, y1, x2, y2)
            
            # Pass disappeared count from tracker
            disappeared = 0
            if hasattr(self.tracker, "tracks") and track_id in self.tracker.tracks:
                disappeared = self.tracker.tracks[track_id].get("disappeared", 0)
            metrics["disappeared"] = disappeared
            
            # 4. State Machine & Stroke Detection
            if track_id not in self.state_machines:
                logger.info(f"New ball detected. Initializing State Machine for Ball {track_id}")
                self.state_machines[track_id] = BallStateMachine(track_id, self.config)
                
            # Update state machine
            state_machine = self.state_machines[track_id]
            state, stroke_detected = state_machine.update(metrics)
            
            # Record current status
            active_ball_states[track_id] = state
            metrics["state"] = state
            metrics["stroke_count"] = state_machine.stroke_count
            active_ball_metrics.append(metrics)
            
            # 5. Handle Stroke Events
            if stroke_detected:
                stroke_events.append((track_id, state_machine.stroke_count))
                
                # Format current real-world timestamp
                # Let's construct a readable ISO timestamp or string
                local_time_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
                
                # Publish MQTT Event
                self.mqtt_publisher.publish_stroke(
                    camera_id=self.camera_id,
                    hole=self.hole,
                    ball_id=track_id,
                    stroke_count=state_machine.stroke_count,
                    timestamp=local_time_str
                )
                
                # Write to CSV
                self.csv_exporter.log_stroke(
                    stroke_number=state_machine.stroke_count,
                    ball_id=track_id,
                    frame=frame_idx,
                    timestamp=timestamp,
                    speed=metrics["speed"],
                    distance=state_machine.distance_traveled_since_ready
                )
                
                # Write to JSON
                self.json_exporter.log_stroke(
                    stroke_number=state_machine.stroke_count,
                    ball_id=track_id,
                    frame=frame_idx,
                    timestamp=timestamp,
                    speed=metrics["speed"]
                )
                
        # Clean up stale state machines of balls that have disappeared for a long time
        # (For this POC, we keep them in memory to preserve their stroke counts if they reappear)
        
        return active_ball_metrics, active_ball_states, stroke_events
