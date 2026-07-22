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
        
        # Minimum pixel displacement from last resting position to auto-count a gap stroke
        self._min_stroke_displacement_px = float(
            config.get("stroke", {}).get("min_stroke_displacement_px", 100.0)
        )
        
        # State machines mapped by track_id
        self.state_machines: Dict[int, BallStateMachine] = {}
        self.prev_active_track_ids = set()

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
        # 1. Object Detection — pass last known ball position as a hint so the
        #    OpenCVBallDetector can crop to a small ROI instead of scanning the full frame.
        #    This processes EVERY frame (no frame skip) so gentle strokes are never missed.
        hint_center = None
        hint_moving = False
        if hasattr(self.tracker, "tracks"):
            for tid, track in self.tracker.tracks.items():
                lc = track.get("last_matched_center")
                if lc is not None:
                    hint_center = lc
                    # Mark as 'moving' if the track's state machine is in MOVING state
                    sm = self.state_machines.get(tid)
                    if sm and sm.state == BallState.MOVING:
                        hint_moving = True
                    break  # We only track one ball (Ball ID 1)
        detections = self.detector.detect(frame, hint_center=hint_center, hint_moving=hint_moving)
        
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
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            # --- Determine if this is a new or re-registered track ---
            is_new_track = track_id not in self.state_machines
            is_reregistered = (not is_new_track) and (track_id not in self.prev_active_track_ids)
            
            # 4a. Initialize state machine for genuinely new tracks
            if is_new_track:
                logger.info(f"New ball detected. Initializing State Machine for Ball {track_id}")
                self.state_machines[track_id] = BallStateMachine(track_id, self.config)

            # 4b. Handle re-registration (track was lost and came back)
            elif is_reregistered:
                sm = self.state_machines[track_id]
                logger.info(f"Ball {track_id} re-registered at ({cx:.1f}, {cy:.1f}).")
                # Reset stale motion history so kinematic readings are clean
                self.motion_analyzer.reset_track(track_id)
                # Try to auto-count a gap stroke if ball moved significantly from last rest
                gap_stroke, was_reset = sm.on_ball_reappeared(cx, cy, self._min_stroke_displacement_px)
                if gap_stroke:
                    local_time_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
                    # Update last_stroke_frame so cooldown applies to gap-strokes too
                    sm.last_stroke_frame = frame_idx
                    self.mqtt_publisher.publish_stroke(
                        camera_id=self.camera_id,
                        hole=self.hole,
                        ball_id=track_id,
                        stroke_count=sm.stroke_count,
                        timestamp=local_time_str
                    )
                    self.csv_exporter.log_stroke(
                        stroke_number=sm.stroke_count,
                        ball_id=track_id,
                        frame=frame_idx,
                        timestamp=timestamp,
                        speed=0.0,
                        distance=0.0
                    )
                    self.json_exporter.log_stroke(
                        stroke_number=sm.stroke_count,
                        ball_id=track_id,
                        frame=frame_idx,
                        timestamp=timestamp,
                        speed=0.0
                    )
                    stroke_events.append((track_id, sm.stroke_count, "stroke"))
                elif was_reset:
                    local_time_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
                    logger.info(f"[Ball {track_id}] TEE RESET (re-appeared on tee)! Stroke counter reset to 0.")
                    self.mqtt_publisher.publish_reset(
                        camera_id=self.camera_id,
                        hole=self.hole,
                        ball_id=track_id,
                        timestamp=local_time_str
                    )
                    stroke_events.append((track_id, 0, "reset"))
                elif sm.state not in [BallState.MOVING]:
                    # Ball barely moved and isn't in MOVING — full reset to avoid bad state
                    sm.reset_motion_state()

            # 4c. Detect stroke onset: ball was STOPPED/READY and just jumped far from anchor
            else:
                sm = self.state_machines.get(track_id)
                if sm and sm.state in [BallState.STOPPED, BallState.READY]:
                    anchor = None
                    if hasattr(self.tracker, "tracks") and track_id in self.tracker.tracks:
                        anchor = self.tracker.tracks[track_id].get("resting_anchor")
                    if anchor:
                        disp = ((cx - anchor[0])**2 + (cy - anchor[1])**2)**0.5
                        if disp > self._min_stroke_displacement_px:
                            # Ball jumped far from rest — clear stale smoothing history
                            # so the speed spike registers correctly this frame
                            self.motion_analyzer.reset_track(track_id)

            # 3. Motion Analysis
            metrics = self.motion_analyzer.update(track_id, (x1, y1, x2, y2), fps, self.pixels_per_meter)
            metrics["track_id"] = track_id
            metrics["bbox"] = (x1, y1, x2, y2)
            metrics["frame_index"] = frame_idx
            metrics["fps"] = fps
            
            # Pass disappeared count from tracker
            disappeared = 0
            if hasattr(self.tracker, "tracks") and track_id in self.tracker.tracks:
                disappeared = self.tracker.tracks[track_id].get("disappeared", 0)
            metrics["disappeared"] = disappeared
            
            # 4. State Machine & Stroke Detection
            state_machine = self.state_machines[track_id]
            state, stroke_detected, was_reset, hole_complete = state_machine.update(metrics)
            
            # 4d. Propagate confirmed resting position back to tracker anchor
            if state in [BallState.STOPPED, BallState.READY]:
                if hasattr(self.tracker, "tracks") and track_id in self.tracker.tracks:
                    self.tracker.tracks[track_id]["resting_anchor"] = (metrics["x"], metrics["y"])
            
            # Record current status
            active_ball_states[track_id] = state
            metrics["state"] = state
            metrics["stroke_count"] = state_machine.stroke_count
            active_ball_metrics.append(metrics)
            
            # 5a. Handle Hole Complete Event
            if hole_complete:
                local_time_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
                logger.info(f"[Ball {track_id}] HOLE COMPLETE! Total strokes: {state_machine.stroke_count}")
                self.mqtt_publisher.publish_hole_complete(
                    camera_id=self.camera_id,
                    hole=self.hole,
                    ball_id=track_id,
                    stroke_count=state_machine.stroke_count,
                    timestamp=local_time_str
                )
                stroke_events.append((track_id, state_machine.stroke_count, "hole_complete"))

            # 5b. Handle Tee Reset Events (stroke count reset to 0)
            if was_reset:
                local_time_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
                logger.info(f"[Ball {track_id}] TEE RESET! Stroke counter reset to 0.")
                self.mqtt_publisher.publish_reset(
                    camera_id=self.camera_id,
                    hole=self.hole,
                    ball_id=track_id,
                    timestamp=local_time_str
                )
                stroke_events.append((track_id, 0, "reset"))
                
            if stroke_detected:
                stroke_events.append((track_id, state_machine.stroke_count, "stroke"))
                
                local_time_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
                
                self.mqtt_publisher.publish_stroke(
                    camera_id=self.camera_id,
                    hole=self.hole,
                    ball_id=track_id,
                    stroke_count=state_machine.stroke_count,
                    timestamp=local_time_str
                )
                
                self.csv_exporter.log_stroke(
                    stroke_number=state_machine.stroke_count,
                    ball_id=track_id,
                    frame=frame_idx,
                    timestamp=timestamp,
                    speed=metrics["speed"],
                    distance=state_machine.distance_traveled_since_ready
                )
                
                self.json_exporter.log_stroke(
                    stroke_number=state_machine.stroke_count,
                    ball_id=track_id,
                    frame=frame_idx,
                    timestamp=timestamp,
                    speed=metrics["speed"]
                )
                
        # Clean up stale state machines of balls that have disappeared for a long time
        # (For this POC, we keep them in memory to preserve their stroke counts if they reappear)
        self.prev_active_track_ids = active_track_ids
        
        return active_ball_metrics, active_ball_states, stroke_events
