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
        self.state = BallState.STOPPED
        self.stroke_count = 0
        self.low_fps = config.get("low_fps", False)
        self.strict_course_mode = config.get("strict_course_mode", False)
        # Always require the ball to settle near the Tee first to prevent accidental strokes
        self.placed_on_tee = False
        
        # Configuration parameters
        motion_cfg = config.get("motion", config.get("stroke", {}))
        self.stop_speed = motion_cfg.get("stop_speed", 0.12)  # m/s
        self.moving_speed = motion_cfg.get("moving_speed", 0.15)  # m/s
        
        if self.low_fps:
            # Broaden hysteresis threshold in laptop mode to avoid sub-pixel jitter locks
            self.stop_speed = 0.25
            self.moving_speed = 0.35
            
        self.minimum_distance = motion_cfg.get("minimum_distance", 0.05)  # meters
        # count_strokes state machine parameters from reference code
        self.anchor = None
        self.last_rest_frame = None
        self.last_rest_pos = None
        self.confirm_list = []
        self.transition_prev_rest_frame = None
        self.transition_prev_rest_pos = None
        self.transition_first_frame = None
        self.settle_candidate_start = None
        self.settle_anchor = None
        
        self.rest_radius = 10.0  # Pixels at 640x360
        self.confirm_frames = 4  # Frames to confirm move
        self.min_rest_span = 15  # Frames to confirm rest
        self.ema_alpha = 0.15
        
        self.holed = False
        self.last_seen_x: Optional[float] = None
        self.last_seen_y: Optional[float] = None
        self.last_stroke_frame: int = -9999
        self.last_resting_position: Optional[Tuple[float, float]] = None
        self.last_known_position: Optional[Tuple[float, float]] = None
        self.ready_position: Optional[Tuple[float, float]] = None
        self.distance_traveled_since_ready = 0.0
 
        # Load target holes/cups and tee point from calibration
        self.target_holes = []
        self.tee_point_scaled = None  # Tee position in processing resolution
        self.tee_reset_radius = 20.0  # Default strict radius for exact center matching
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
                
                # Auto-detect left layout to mirror X coordinates
                video_input = config.get("video", {}).get("input", "")
                mirror_x = "left" in os.path.basename(video_input).lower()
                if mirror_x:
                    logger.info(f"[Ball {self.track_id}] Flipped left camera layout detected. Mirroring calibration coordinates.")
                
                for region in cal.get("ignore_regions", []):
                    if "x" not in region or "y" not in region:
                        continue  # Skip polygon ignore regions
                    name = region.get("name", "").lower()
                    cal_r = float(region.get("radius", 10.0)) * scale_x
                    
                    # Mirror raw X coordinate relative to source resolution width (base_res[0])
                    rx = region["x"]
                    if mirror_x:
                        rx = base_res[0] - rx
                        
                    if "hole" in name or "cup" in name:
                        ry = region["y"]
                        # Use exact calibrated cup radius scaled to processing resolution
                        h_r = cal_r
                        self.target_holes.append((rx * scale_x, ry * scale_y, h_r))
                    elif "tee" in name:
                        ry = region["y"]
                        self.tee_point_scaled = (rx * scale_x, ry * scale_y)
                        # Proximity threshold for Tee reset (60px at 640x360 resolution for 99% accuracy)
                        self.tee_reset_radius = 60.0
                        
                logger.info(f"[Ball {self.track_id}] Loaded target holes/cups (strict): {self.target_holes}")
                if self.tee_point_scaled:
                    logger.info(f"[Ball {self.track_id}] Loaded tee point (scaled): {self.tee_point_scaled}, radius={self.tee_reset_radius:.1f}px")
                else:
                    self.placed_on_tee = True  # Default to True if no tee is calibrated to allow tracking
            except Exception as e:
                logger.error(f"Failed to load calibration in state machine: {e}")
                self.placed_on_tee = True
 
    def reset_motion_state(self):
        """Resets the motion tracking state when a ball is re-registered, preserving stroke count."""
        self.state = BallState.STOPPED
        self.anchor = None
        self.last_rest_frame = None
        self.last_rest_pos = None
        self.confirm_list = []
        self.transition_prev_rest_frame = None
        self.transition_prev_rest_pos = None
        self.transition_first_frame = None
        self.settle_candidate_start = None
        self.settle_anchor = None
        self.holed = False
        self.last_known_position = None
        self.ready_position = None
        self.distance_traveled_since_ready = 0.0
 
    def on_ball_reappeared(self, new_x: float, new_y: float, min_displacement_px: float = 45.0) -> Tuple[bool, bool]:
        """
        Called when Ball ID 1's track is re-registered after being lost (deleted).
        If the ball reappears significantly far from its last confirmed resting position
        (or last known position as fallback), a stroke is automatically counted.
        If it reappears on the tee, the stroke count is reset.
 
        Returns (stroke_counted_bool, was_reset_bool).
        """
        # If the ball was previously holed (round completed), reappearing means starting a NEW round!
        if self.holed:
            logger.info(f"[Ball {self.track_id}] Ball reappeared after completing hole! Resetting holed flag and resetting stroke counter for new round.")
            self.holed = False
            self.placed_on_tee = True
            self.stroke_count = 0
            self.last_stroke_frame = -9999
            self.last_resting_position = None
            self.last_known_position = (new_x, new_y)
            self.state = BallState.UNKNOWN
            return False, True

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
                if self.placed_on_tee and self.state in [BallState.READY, BallState.STOPPED]:
                    self.stroke_count += 1
                    logger.info(f"[Ball {self.track_id}] *** STROKE VALIDATED ON REAPPEARANCE *** Ball escaped tee ({displacement:.1f}px from {reference_label}). Stroke #{self.stroke_count}!")
                    self.state = BallState.MOVING
                    self.frames_below_stop_speed_for_stopping = 0
                    self.last_resting_position = None
                    self.last_known_position = (new_x, new_y)
                    self.has_left_tee = True
                    return True, False
                else:
                    logger.info(f"[Ball {self.track_id}] Ball reappeared ({displacement:.1f}px from {reference_label}). Continuing tracking.")
                    self.state = BallState.MOVING
                    self.frames_below_stop_speed_for_stopping = 0
                    self.last_resting_position = None
                    self.last_known_position = (new_x, new_y)
                    if hasattr(self, "stroke_consecutive_frames"):
                        self.stroke_consecutive_frames = 0
                    return False, False
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

    def update(self, metrics: Dict[str, Any]) -> Tuple[BallState, bool, bool, bool, bool]:
        """
        Updates the state machine based on the latest motion metrics of the ball.
        """
        if self.holed:
            return self.state, False, False, False, False

        curr_x = metrics.get("x", 0.0)
        curr_y = metrics.get("y", 0.0)
        frame_idx = metrics.get("frame_index", 0)
        pos = (curr_x, curr_y)
        
        # Initialize anchor on first update
        if self.anchor is None:
            self.anchor = [curr_x, curr_y]
            self.last_rest_frame = frame_idx
            self.last_rest_pos = pos

        # Always track last known position (every frame, every state)
        self.last_known_position = pos
        
        # Maintain backward compatibility metrics for distance and ready positions
        self.ready_position = self.last_rest_pos
        if self.last_rest_pos is not None:
            self.distance_traveled_since_ready = math.hypot(curr_x - self.last_rest_pos[0], curr_y - self.last_rest_pos[1]) / metrics.get("pixels_per_meter", 200.0)
        else:
            self.distance_traveled_since_ready = 0.0

        # Check if ball has entered a cup or hole
        disappeared = metrics.get("disappeared", 0)
        if disappeared == 0:
            self.last_seen_x = curr_x
            self.last_seen_y = curr_y

        eval_x = self.last_seen_x if (disappeared > 0 and self.last_seen_x is not None) else curr_x
        eval_y = self.last_seen_y if (disappeared > 0 and self.last_seen_y is not None) else curr_y

        for item in self.target_holes:
            hx, hy = item[0], item[1]
            h_radius = item[2] if len(item) > 2 else 18.0
            dist_to_cup = math.hypot(eval_x - hx, eval_y - hy)
            if dist_to_cup < max(35.0, h_radius):
                if disappeared >= 10:  # Ball missing inside cup zone = holed!
                    hole_complete = False
                    if not self.holed:
                        self.holed = True
                        hole_complete = True
                        self.placed_on_tee = False  # Disable further strokes until ball is placed back on Tee
                        logger.info(f"[Ball {self.track_id}] Ball entered cup/hole at ({eval_x:.1f}, {eval_y:.1f})! Hole complete with {self.stroke_count} stroke(s). Strokes disabled.")
                    return self.state, False, False, hole_complete, False

        stroke_detected = False
        was_reset = False
        tee_placed_event = False
        old_state = self.state

        def dist_func(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        def ema_func(anc, p, alpha=self.ema_alpha):
            return [anc[0] * (1. - alpha) + p[0] * alpha,
                    anc[1] * (1. - alpha) + p[1] * alpha]

        # Reset stroke count if ball is placed back on Tee
        if self.tee_point_scaled is not None:
            dist_to_tee = dist_func(pos, self.tee_point_scaled)
            if dist_to_tee <= self.tee_reset_radius:
                if not self.placed_on_tee:
                    self.placed_on_tee = True
                    tee_placed_event = True
                if self.stroke_count > 0:
                    logger.info(f"[Ball {self.track_id}] Ball placed on tee position ({curr_x:.1f}, {curr_y:.1f}). Resetting strokes.")
                    self.stroke_count = 0
                    was_reset = True
                    self.state = BallState.STOPPED
                    self.anchor = [curr_x, curr_y]
                    self.last_rest_frame = frame_idx
                    self.last_rest_pos = pos
                    self.confirm_list = []
                    self.settle_candidate_start = None

        # State Machine Transitions (REST -> CONFIRM -> MOVE)
        # STOPPED maps to 'rest', READY maps to 'confirm', MOVING maps to 'move'
        if self.state == BallState.STOPPED:
            d = dist_func(pos, self.anchor)
            if d <= self.rest_radius:
                self.last_rest_frame = frame_idx
                self.last_rest_pos = pos
                self.anchor = ema_func(self.anchor, pos)
                self.last_resting_position = pos
            else:
                self.state = BallState.READY  # Transition to confirmation phase
                self.confirm_list = [(frame_idx, curr_x, curr_y)]
                self.transition_prev_rest_frame = self.last_rest_frame
                self.transition_prev_rest_pos = self.last_rest_pos
                self.transition_first_frame = frame_idx
                logger.info(f"[Ball {self.track_id}] STOPPED -> CONFIRM (READY). Pos: {pos}, Anchor: {self.anchor}, Dist: {d:.2f}")

        elif self.state == BallState.READY:
            d = dist_func(pos, self.anchor)
            if d <= self.rest_radius:
                self.state = BallState.STOPPED
                self.last_rest_frame = frame_idx
                self.last_rest_pos = pos
                self.confirm_list = []
                logger.info(f"[Ball {self.track_id}] Ball returned to rest. READY -> STOPPED")
            else:
                self.confirm_list.append((frame_idx, curr_x, curr_y))
                if len(self.confirm_list) >= self.confirm_frames:
                    if self.placed_on_tee:
                        # Stroke confirmed!
                        self.stroke_count += 1
                        stroke_detected = True
                        self.state = BallState.MOVING
                        self.settle_candidate_start = None
                        self.last_resting_position = None
                        logger.info(f"[Ball {self.track_id}] STROKE VALIDATED #{self.stroke_count}! "
                                    f"Ball moved away from anchor {self.anchor} to {pos} (dist: {d:.2f}px). State -> MOVING")
                    else:
                        # Moved before Tee placement — track movement but do not count a stroke
                        self.state = BallState.MOVING
                        self.settle_candidate_start = None
                        self.last_resting_position = None
                        logger.info(f"[Ball {self.track_id}] Ball moved before Tee placement. Not counting stroke. State -> MOVING")

        elif self.state == BallState.MOVING:
            if self.settle_candidate_start is None:
                self.settle_candidate_start = frame_idx
                self.settle_anchor = pos
            else:
                d = dist_func(pos, self.settle_anchor)
                if d > self.rest_radius:
                    self.settle_candidate_start = frame_idx
                    self.settle_anchor = pos
                else:
                    span = frame_idx - self.settle_candidate_start
                    if span >= self.min_rest_span:
                        self.state = BallState.STOPPED
                        self.anchor = list(self.settle_anchor)
                        self.last_rest_frame = frame_idx
                        self.last_rest_pos = pos
                        self.settle_candidate_start = None
                        logger.info(f"[Ball {self.track_id}] Ball settled. MOVING -> STOPPED at {pos}")

        if old_state != self.state:
            logger.info(f"[Ball {self.track_id}] State changed: {old_state.value} -> {self.state.value}")

        return self.state, stroke_detected, was_reset, False, tee_placed_event
