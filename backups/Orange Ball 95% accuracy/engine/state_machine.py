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
        self.low_fps = config.get("low_fps", False)
        # Default to True in low-FPS laptop testing mode (to allow testing local files),
        # otherwise require the ball to settle on the Tee first in production.
        self.placed_on_tee = True if self.low_fps else False
        
        # Configuration parameters
        motion_cfg = config.get("motion", config.get("stroke", {}))
        self.stop_speed = motion_cfg.get("stop_speed", 0.12)  # m/s
        self.moving_speed = motion_cfg.get("moving_speed", 0.15)  # m/s
        
        if self.low_fps:
            # Broaden hysteresis threshold in laptop mode to avoid sub-pixel jitter locks
            self.stop_speed = 0.20
            self.moving_speed = 0.35
            
        self.minimum_distance = motion_cfg.get("minimum_distance", 0.05)  # meters
        self.last_impact_frame = None  # will store frame index of first high‑speed impact
        self.ready_delay_frames = motion_cfg.get("ready_delay_frames", 10)
        self.stop_delay_frames = motion_cfg.get("stop_delay_frames", 15)
        self.accel_threshold = motion_cfg.get("acceleration_threshold", 1.0)
        if self.low_fps:
            # Lower acceleration threshold on low-FPS/frame-skip runs because
            # binned time steps smooth out the impact acceleration peak.
            self.accel_threshold = min(self.accel_threshold, 0.15)  # m/s^2
        
        # Minimum frames that must elapse between any two counted strokes
        stroke_cfg = config.get("stroke", {})
        self.stroke_cooldown_frames = stroke_cfg.get("stroke_cooldown_frames", 120)
        
        # Tracking variables
        self.frames_stopped = 0
        self.frames_below_stop_speed_for_stopping = 0
        self.ready_position: Optional[Tuple[float, float]] = None  # (x, y) center in pixels
        self.has_exceeded_stroke_speed = False
        self.holed = False
        
        # Frame index of the last confirmed stroke (used for cooldown guard)
        self.last_stroke_frame: int = -9999
        
        # Last confirmed resting position (updated every STOPPED/READY frame, cleared on stroke)
        # Used by on_ball_reappeared() to detect gap strokes.
        self.last_resting_position: Optional[Tuple[float, float]] = None
        
        # Last known position regardless of state (updated every frame)
        # Fallback for gap-stroke detection when resting position is unknown.
        self.last_known_position: Optional[Tuple[float, float]] = None
        
        # Cumulative distance traveled since ball was in READY state
        self.distance_traveled_since_ready = 0.0  # in meters

        # Load target holes/cups and tee point from calibration
        self.target_holes = []
        self.tee_point_scaled = None  # Tee position in processing resolution
        self.tee_reset_radius = 60.0  # Pixels: proximity threshold for tee reset
        import os
        import json
        calibration_path = "config/calibration.json"
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, "r") as f:
                    cal = json.load(f)
                base_res = cal.get("source_resolution", [3840, 2160])
                resize_width = config.get("processing", {}).get("resize_width", 1280)
                scale_x = resize_width / base_res[0]
                scale_y = (resize_width * (base_res[1] / base_res[0])) / base_res[1]
                
                for region in cal.get("ignore_regions", []):
                    name = region.get("name", "").lower()
                    if "hole" in name or "cup" in name:
                        rx = region["x"]
                        ry = region["y"]
                        self.target_holes.append((rx * scale_x, ry * scale_y))
                    elif "tee" in name:
                        rx = region["x"]
                        ry = region["y"]
                        self.tee_point_scaled = (rx * scale_x, ry * scale_y)
                        
                logger.info(f"[Ball {self.track_id}] Loaded target holes/cups: {self.target_holes}")
                if self.tee_point_scaled:
                    logger.info(f"[Ball {self.track_id}] Loaded tee point (scaled): {self.tee_point_scaled}")
                else:
                    self.placed_on_tee = True  # Default to True if no tee is calibrated to allow tracking
            except Exception as e:
                logger.error(f"Failed to load calibration in state machine: {e}")
                self.placed_on_tee = True

    def reset_motion_state(self):
        """Resets the motion tracking state when a ball is re-registered, preserving stroke count."""
        self.state = BallState.UNKNOWN
        self.frames_stopped = 0
        self.frames_below_stop_speed_for_stopping = 0
        self.ready_position = None
        self.has_exceeded_stroke_speed = False
        self.holed = False
        self.distance_traveled_since_ready = 0.0
        # NOTE: last_resting_position is intentionally NOT cleared here so that
        # on_ball_reappeared() can still use it after a reset.
        self.last_known_position = None
        if hasattr(self, "stroke_consecutive_frames"):
            self.stroke_consecutive_frames = 0

    def on_ball_reappeared(self, new_x: float, new_y: float, min_displacement_px: float = 100.0) -> Tuple[bool, bool]:
        """
        Called when Ball ID 1's track is re-registered after being lost (deleted).
        If the ball reappears significantly far from its last confirmed resting position
        (or last known position as fallback), a stroke is automatically counted.
        If it reappears on the tee, the stroke count is reset.

        Returns (stroke_counted_bool, was_reset_bool).
        """
        # Check if ball reappeared on the tee
        if self.tee_point_scaled is not None:
            dist_to_tee = math.sqrt((new_x - self.tee_point_scaled[0])**2 + (new_y - self.tee_point_scaled[1])**2)
            if dist_to_tee <= self.tee_reset_radius:
                if self.stroke_count > 0:
                    logger.info(f"[Ball {self.track_id}] Ball placed on tee position ({new_x:.1f}, {new_y:.1f}). Resetting strokes.")
                    self.stroke_count = 0
                    self.last_stroke_frame = -9999
                    return False, True
                return False, False
        # Prefer confirmed resting position; fall back to last known position
        reference_pos = self.last_resting_position or self.last_known_position
        reference_label = "last rest" if self.last_resting_position else "last known"

        if reference_pos is not None:
            dx = new_x - reference_pos[0]
            dy = new_y - reference_pos[1]
            displacement = math.sqrt(dx**2 + dy**2)

            if displacement >= min_displacement_px:
                # If the ball was already rolling when it went missing, this is the same stroke!
                if self.state == BallState.MOVING:
                    logger.info(f"[Ball {self.track_id}] Ball reappeared while already in MOVING state. Not counting another stroke.")
                else:
                    self.stroke_count += 1
                    logger.info(f"[Ball {self.track_id}] Ball reappeared far from reference ({displacement:.1f}px). Counting gap stroke.")
                
                self.state = BallState.MOVING
                self.frames_below_stop_speed_for_stopping = 0
                self.last_resting_position = None  # Unknown until ball rests and is confirmed again
                self.last_known_position = (new_x, new_y)
                # last_stroke_frame will be updated by the caller (stroke_engine) via frame_idx
                if hasattr(self, "stroke_consecutive_frames"):
                    self.stroke_consecutive_frames = 0
                logger.info(
                    f"[Ball {self.track_id}] GAP-STROKE #{self.stroke_count}! "
                    f"Re-appeared {displacement:.1f}px from {reference_label} "
                    f"({reference_pos[0]:.1f}, {reference_pos[1]:.1f}) → ({new_x:.1f}, {new_y:.1f}). State → MOVING"
                )
                return True, False
            else:
                # Ball barely moved — just a detection gap, not a stroke
                logger.info(
                    f"[Ball {self.track_id}] Re-appeared near same position ({displacement:.1f}px from {reference_label}) — gap only, no stroke"
                )
                return False, False
        else:
            # No reference position at all — reset normally
            logger.info(f"[Ball {self.track_id}] Re-appeared with no reference position — resetting state")
            return False, False

    def update(self, metrics: Dict[str, Any]) -> Tuple[BallState, bool, bool, bool]:
        """
        Updates the state machine based on the latest motion metrics of the ball.
        
        Args:
            metrics: Motion metrics dictionary from IMotionAnalyzer containing:
                     "x", "y", "speed", "cumulative_distance", etc.
                     
        Returns:
            A tuple of (new_state, stroke_detected_bool, was_reset_bool, hole_complete_bool).
        """
        if self.holed:
            return self.state, False, False, False

        current_speed = metrics.get("speed", 0.0)
        curr_x = metrics.get("x", 0.0)
        curr_y = metrics.get("y", 0.0)
        
        # Dynamically scale frame thresholds based on the actual frame rate to keep time durations constant
        fps = metrics.get("fps", 30.0)
        self.stroke_cooldown_frames = max(5, int(fps * 1.5))  # Exactly 1.5 seconds cooldown
        self.ready_delay_frames = max(2, int(fps * 0.15))     # Snappy 0.15 seconds ready delay
        self.stop_delay_frames = max(2, int(fps * 0.25))       # Snappy 0.25 seconds stop delay
        
        # Always track last known position (every frame, every state)
        self.last_known_position = (curr_x, curr_y)
        
        # Check if ball has entered a cup or hole
        disappeared = metrics.get("disappeared", 0)
        for hx, hy in self.target_holes:
            dist = ((curr_x - hx)**2 + (curr_y - hy)**2)**0.5
            if dist < 35.0:
                # Ball has entered the hole if it goes missing inside the ignore region
                # (disappeared >= 15 frames)
                if disappeared >= 15 and current_speed < self.stop_speed:
                    hole_complete = False
                    if not self.holed:  # Only fire once
                        self.holed = True
                        hole_complete = True
                        logger.info(f"[Ball {self.track_id}] Ball entered cup/hole at ({curr_x:.1f}, {curr_y:.1f})! Hole complete with {self.stroke_count} stroke(s).")
                    return self.state, False, False, hole_complete
        
        stroke_detected = False
        was_reset = False
        hole_complete = False
        old_state = self.state

        # If the ball has disappeared (undetected), do not allow it to transition from MOVING to STOPPED
        if disappeared > 0:
            if self.state == BallState.MOVING:
                return self.state, False, False, False

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

        elif self.state == BallState.STOPPED and current_speed < self.stop_speed:
            self.frames_stopped += 1
            # Continuously update last_resting_position while the ball is confirmed stationary
            self.last_resting_position = (curr_x, curr_y)
            if self.frames_stopped >= self.ready_delay_frames:
                self.state = BallState.READY
                self.ready_position = (curr_x, curr_y)
                self.distance_traveled_since_ready = 0.0
                self.has_exceeded_stroke_speed = False
                
                # Reset stroke counter when ball returns to the tee position
                if self.tee_point_scaled is not None:
                    dist_to_tee = math.sqrt(
                        (curr_x - self.tee_point_scaled[0])**2 + 
                        (curr_y - self.tee_point_scaled[1])**2
                    )
                    if dist_to_tee <= self.tee_reset_radius:
                        self.placed_on_tee = True
                        if self.stroke_count > 0:
                            logger.info(f"[Ball {self.track_id}] Ball at tee position ({curr_x:.1f}, {curr_y:.1f}), "
                                        f"dist={dist_to_tee:.1f}px. Resetting stroke counter from {self.stroke_count} to 0 (new round).")
                            self.stroke_count = 0
                            self.last_stroke_frame = -9999
                            was_reset = True
                
                logger.info(f"[Ball {self.track_id}] STOPPED -> READY at position ({curr_x:.1f}, {curr_y:.1f})")

        elif self.state == BallState.READY or self.state == BallState.STOPPED:
            # Continuously update resting position while waiting for the stroke
            self.last_resting_position = (curr_x, curr_y)
            if current_speed >= self.stop_speed:
                # If we were in STOPPED state and got hit, initialize ready_position to last rest
                if self.ready_position is None:
                    self.ready_position = self.last_resting_position or (curr_x, curr_y)
                
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
                
                # Check speed threshold and record impact metrics
                current_accel = metrics.get("acceleration", 0.0)
                if current_speed >= self.moving_speed:
                    self.has_exceeded_stroke_speed = True
                    if self.last_impact_frame is None:
                        self.last_impact_frame = metrics.get("frame_index")
                        self.max_impact_accel = current_accel
                    else:
                        self.max_impact_accel = max(getattr(self, "max_impact_accel", 0.0), current_accel)
                
                # Stroke detection: was in READY/STOPPED, exceeded stroke speed, traveled minimum distance, and acceleration matches
                if self.placed_on_tee and self.has_exceeded_stroke_speed and self.distance_traveled_since_ready >= self.minimum_distance:
                    if current_accel >= self.accel_threshold or getattr(self, "max_impact_accel", 0.0) >= self.accel_threshold:
                        if not hasattr(self, "stroke_consecutive_frames"):
                            self.stroke_consecutive_frames = 0
                        self.stroke_consecutive_frames += 1
                        # Ensure enough consecutive moving frames for a stroke
                        required_frames = 1 if self.low_fps else 2
                        if self.stroke_consecutive_frames >= required_frames:
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
                            eval_accel = current_accel
                            frame_index = metrics.get("frame_index", 0)
                            if self.last_impact_frame is not None and (frame_index - self.last_impact_frame) <= 5:
                                eval_accel = max(eval_accel, getattr(self, "max_impact_accel", 0.0))
                                
                            if eval_accel >= self.accel_threshold:
                                score += 25.0
                            else:
                                score += min((eval_accel / max(0.1, self.accel_threshold)) * 25.0, 25.0)
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

                            frame_index = metrics.get("frame_index", 0)
                            frames_since_last = frame_index - self.last_stroke_frame
                            cooldown_ok = frames_since_last >= self.stroke_cooldown_frames

                            if score >= 85.0 and cooldown_ok:
                                self.stroke_count += 1
                                stroke_detected = True
                                self.state = BallState.MOVING
                                self.frames_below_stop_speed_for_stopping = 0
                                self.stroke_consecutive_frames = 0
                                self.last_resting_position = None
                                self.last_stroke_frame = frame_index
                                logger.info(f"[Ball {self.track_id}] STROKE VALIDATED #{self.stroke_count}! "
                                            f"Speed: {current_speed:.2f} m/s, Dist: {self.distance_traveled_since_ready:.2f} m, Accel: {eval_accel:.2f} m/s^2, Confidence: {score:.1f}")
                                # Reset impact frame and cached acceleration after successful validation
                                self.last_impact_frame = None
                                self.max_impact_accel = 0.0
                            elif score >= 85.0 and not cooldown_ok:
                                logger.warning(
                                    f"[Ball {self.track_id}] Stroke suppressed by cooldown "
                                    f"({frames_since_last}/{self.stroke_cooldown_frames} frames since last stroke). Score: {score:.1f}"
                                )
                                self.stroke_consecutive_frames = 0
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
            is_untracked = metrics.get("disappeared", 0) > 0
            
            # Mid-roll hit detection: if the ball is rolling slowly (trickling) and gets struck again,
            # detect the speed/acceleration spike as a new stroke, provided the cooldown has expired.
            frame_index = metrics.get("frame_index", 0)
            frames_since_last = frame_index - self.last_stroke_frame
            cooldown_ok = frames_since_last >= self.stroke_cooldown_frames
            
            current_accel = metrics.get("acceleration", 0.0)
            if not is_untracked and cooldown_ok and current_speed >= self.moving_speed and current_accel >= self.accel_threshold:
                self.stroke_count += 1
                stroke_detected = True
                self.frames_below_stop_speed_for_stopping = 0
                self.stroke_consecutive_frames = 0
                self.last_resting_position = None
                self.last_stroke_frame = frame_index
                self.ready_position = None
                logger.info(f"[Ball {self.track_id}] MID-ROLL STROKE VALIDATED #{self.stroke_count}! "
                            f"Speed: {current_speed:.2f} m/s, Accel: {current_accel:.2f} m/s^2")
            
            if not is_untracked and current_speed < self.stop_speed:
                self.frames_below_stop_speed_for_stopping += 1
                if self.frames_below_stop_speed_for_stopping >= self.stop_delay_frames:
                    self.state = BallState.STOPPED
                    self.frames_stopped = 1
                    self.ready_position = None
                    logger.info(f"[Ball {self.track_id}] MOVING -> STOPPED (Speed: {current_speed:.4f} m/s)")
            elif not is_untracked:
                self.frames_below_stop_speed_for_stopping = 0

        if old_state != self.state and self.state != BallState.READY:
            logger.info(f"Ball {self.track_id} state changed: {old_state.value} -> {self.state.value}")

        return self.state, stroke_detected, was_reset, hole_complete
