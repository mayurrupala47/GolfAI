from enum import Enum
import math
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BallState(Enum):
    UNKNOWN = "UNKNOWN"
    STOPPED = "STOPPED"
    READY = "READY"
    MOVING = "MOVING"


class BallStateMachine:
    """
    Manages the state transitions and stroke detection logic for an individual tracked golf ball.
    """
    def __init__(self, track_id: int, config: Dict[str, Any]):
        self.track_id = track_id
        self.state = BallState.UNKNOWN
        self.stroke_count = 0
        
        # Configuration parameters
        motion_cfg = config.get("motion", config.get("stroke", {}))
        self.stop_speed = motion_cfg.get("stop_speed", 0.05)  # m/s, higher to ignore tiny jitters
        self.moving_speed = motion_cfg.get("moving_speed", 0.80)  # m/s
        self.minimum_distance = motion_cfg.get("minimum_distance", 0.20)  # meters, require more travel
        self.last_impact_frame = None  # will store frame index of first high‑speed impact
        self.ready_delay_frames = motion_cfg.get("ready_delay_frames", 30)
        self.stop_delay_frames = motion_cfg.get("stop_delay_frames", 15)
        self.accel_threshold = motion_cfg.get("acceleration_threshold", 1.0)  # m/s^2
        
        # Tracking variables
        self.frames_stopped = 0
        self.frames_below_stop_speed_for_stopping = 0
        self.ready_position: Optional[Tuple[float, float]] = None  # (x, y) center in pixels
        self.has_exceeded_stroke_speed = False
        self.holed = False
        
        # Cumulative distance traveled since ball was in READY state
        self.distance_traveled_since_ready = 0.0  # in meters

        # Load target holes/cups from calibration
        self.target_holes = []
        import os
        import json
        calibration_path = "config/calibration.json"
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, "r") as f:
                    cal = json.load(f)
                for region in cal.get("ignore_regions", []):
                    name = region.get("name", "").lower()
                    if "hole" in name or "cup" in name:
                        rx = region["x"]
                        ry = region["y"]
                        scale_x = 1280.0 / 3840.0
                        scale_y = 720.0 / 2160.0
                        self.target_holes.append((rx * scale_x, ry * scale_y))
                logger.info(f"[Ball {self.track_id}] Loaded target holes/cups: {self.target_holes}")
            except Exception as e:
                logger.error(f"Failed to load target holes in state machine: {e}")

    def update(self, metrics: Dict[str, Any]) -> Tuple[BallState, bool]:
        """
        Updates the state machine based on the latest motion metrics of the ball.
        
        Args:
            metrics: Motion metrics dictionary from IMotionAnalyzer containing:
                     "x", "y", "speed", "cumulative_distance", etc.
                     
        Returns:
            A tuple of (new_state, stroke_detected).
        """
        if self.holed:
            return self.state, False

        current_speed = metrics.get("speed", 0.0)
        curr_x = metrics.get("x", 0.0)
        curr_y = metrics.get("y", 0.0)
        
        # Check if ball has entered a cup or hole
        disappeared = metrics.get("disappeared", 0)
        for hx, hy in self.target_holes:
            dist = ((curr_x - hx)**2 + (curr_y - hy)**2)**0.5
            if dist < 35.0:
                # Ball has entered the hole if it goes missing inside the ignore region
                # (disappeared >= 15 frames) or stops exactly near the center (dist < 20px)
                if (disappeared >= 15 and current_speed < self.stop_speed) or (dist < 20.0 and current_speed < self.stop_speed):
                    self.holed = True
                    logger.info(f"[Ball {self.track_id}] Ball entered cup/hole at ({curr_x:.1f}, {curr_y:.1f})! Disabling further stroke detection.")
                    return self.state, False
        
        stroke_detected = False
        old_state = self.state

        # Transition logic based on current state
        if self.state == BallState.UNKNOWN:
            # Transition to STOPPED if speed is below threshold
            if current_speed < self.stop_speed:
                self.state = BallState.STOPPED
                self.frames_stopped = 1
                logger.info(f"[Ball {self.track_id}] UNKNOWN -> STOPPED (Speed: {current_speed:.4f} m/s)")
            else:
                # If it's already moving, we keep it as UNKNOWN until it stops
                pass

        elif self.state == BallState.STOPPED:
            if current_speed < self.stop_speed:
                self.frames_stopped += 1
                if self.frames_stopped >= self.ready_delay_frames:
                    self.state = BallState.READY
                    self.ready_position = (curr_x, curr_y)
                    self.distance_traveled_since_ready = 0.0
                    self.has_exceeded_stroke_speed = False
                    logger.info(f"[Ball {self.track_id}] STOPPED -> READY at position ({curr_x:.1f}, {curr_y:.1f})")
            else:
                self.frames_stopped = 0

        elif self.state == BallState.READY:
            if current_speed >= self.stop_speed:
                # Calculate physical travel distance from where it was ready
                if self.ready_position is not None:
                    # Retrieve the cumulative distance increment or compute straight-line distance
                    # For stroke verification, cumulative travel distance is preferred.
                    # We can use the delta from the analyzer or compute relative distance from ready position
                    # Let's check how the analyzer computes distance.
                    # If we calculate straight line distance from ready position in pixels and convert to meters:
                    dx = curr_x - self.ready_position[0]
                    dy = curr_y - self.ready_position[1]
                    pixel_distance = math.sqrt(dx**2 + dy**2)
                    
                    pixels_per_meter = metrics.get("pixels_per_meter", 200.0)
                    self.distance_traveled_since_ready = pixel_distance / pixels_per_meter
                
                # Check speed threshold
                if current_speed >= self.moving_speed:
                    self.has_exceeded_stroke_speed = True
                    if self.last_impact_frame is None:
                        self.last_impact_frame = metrics.get("frame_index")
                
                # Stroke detection: was in READY, exceeded stroke speed, traveled minimum distance, and acceleration matches
                if self.has_exceeded_stroke_speed and self.distance_traveled_since_ready >= self.minimum_distance:
                    current_accel = metrics.get("acceleration", 0.0)
                    if current_accel >= self.accel_threshold:
                        if not hasattr(self, "stroke_consecutive_frames"):
                            self.stroke_consecutive_frames = 0
                        self.stroke_consecutive_frames += 1
                        # Ensure at least 2 consecutive moving frames for a stroke
                        if self.stroke_consecutive_frames >= 2:
                            # Calculate Confidence Score
                            score = 0.0
                            # 1. Rest Duration Score (Max 20)
                            if self.frames_stopped >= 48:
                                score += 20.0
                            elif self.frames_stopped >= 30:
                                score += 15.0
                            else:
                                score += 10.0
                            # 2. Player Motion Proximity Score (Max 20)
                            score += 20.0 if metrics.get("player_nearby", True) else 10.0
                            # 3. Acceleration Score (Max 25)
                            if current_accel >= self.accel_threshold:
                                score += 25.0
                            else:
                                score += min((current_accel / max(0.1, self.accel_threshold)) * 25.0, 25.0)
                            # 4. Travel Distance Score (Max 20)
                            if self.distance_traveled_since_ready >= self.minimum_distance:
                                score += 20.0
                            else:
                                score += min((self.distance_traveled_since_ready / max(0.01, self.minimum_distance)) * 20.0, 20.0)
                            # 5. Shape Match Score (Max 15)
                            score += 15.0
                            # 6. Trajectory Consistency Score (Max 10)
                            score += 10.0

                            logger.info(f"[Ball {self.track_id}] Evaluating stroke confidence: {score:.1f}/100.0 (Threshold: 85.0)")

                            if score >= 85.0:
                                self.stroke_count += 1
                                stroke_detected = True
                                self.state = BallState.MOVING
                                self.frames_below_stop_speed_for_stopping = 0
                                self.stroke_consecutive_frames = 0
                                logger.info(f"[Ball {self.track_id}] STROKE VALIDATED #{self.stroke_count}! "
                                            f"Speed: {current_speed:.2f} m/s, Dist: {self.distance_traveled_since_ready:.2f} m, Accel: {current_accel:.2f} m/s^2, Confidence: {score:.1f}")
                                # Reset impact frame after successful validation
                                self.last_impact_frame = None
                            else:
                                logger.warning(f"[Ball {self.track_id}] Stroke event rejected due to low confidence: {score:.1f}")
                                self.stroke_consecutive_frames = 0
                        pass
                    else:
                        if hasattr(self, "stroke_consecutive_frames"):
                            self.stroke_consecutive_frames = 0
                else:
                    if hasattr(self, "stroke_consecutive_frames"):
                        self.stroke_consecutive_frames = 0
            else:
                # Still stopped
                self.distance_traveled_since_ready = 0.0
                self.has_exceeded_stroke_speed = False

        elif self.state == BallState.MOVING:
            if current_speed < self.stop_speed:
                self.frames_below_stop_speed_for_stopping += 1
                if self.frames_below_stop_speed_for_stopping >= self.stop_delay_frames:
                    self.state = BallState.STOPPED
                    self.frames_stopped = 1
                    logger.info(f"[Ball {self.track_id}] MOVING -> STOPPED (Speed: {current_speed:.4f} m/s)")
            else:
                self.frames_below_stop_speed_for_stopping = 0

        if old_state != self.state and self.state != BallState.READY:
            logger.info(f"Ball {self.track_id} state changed: {old_state.value} -> {self.state.value}")

        return self.state, stroke_detected
