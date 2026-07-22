from typing import Tuple, Dict, Any, List
import math
import logging
from ai.interfaces import IMotionAnalyzer

logger = logging.getLogger(__name__)

class MotionAnalyzer(IMotionAnalyzer):
    """
    Tracks and analyzes motion characteristics (velocity, acceleration, travel distance) of golf balls.
    Uses running averages to smooth out detection and tracking noise.
    """
    def __init__(self, smoothing_window: int = 5):
        """
        Initializes the motion analyzer.
        
        Args:
            smoothing_window: Number of frames to use for moving average smoothing of coordinates and speed.
        """
        self.smoothing_window = smoothing_window
        # Dictionary storing history for each track_id:
        # { track_id: { "raw_centers": [], "smoothed_centers": [], "velocities": [], "cumulative_distance": 0.0 } }
        self.history: Dict[int, Dict[str, Any]] = {}

    def update(self, track_id: int, bbox: Tuple[float, float, float, float], fps: float, pixels_per_meter: float) -> Dict[str, Any]:
        """
        Updates kinematic state for a track ID and computes metrics.
        """
        # Calculate current center coordinates
        x1, y1, x2, y2 = bbox
        curr_raw_x = (x1 + x2) / 2.0
        curr_raw_y = (y1 + y2) / 2.0
        
        dt = 1.0 / fps if fps > 0 else 0.033

        # Initialize history for new tracks
        if track_id not in self.history:
            self.history[track_id] = {
                "raw_centers": [],
                "smoothed_centers": [],
                "velocities": [],
                "cumulative_distance": 0.0,
                "last_speed": 0.0
            }

        hist = self.history[track_id]
        
        # Append raw center
        hist["raw_centers"].append((curr_raw_x, curr_raw_y))
        if len(hist["raw_centers"]) > self.smoothing_window:
            hist["raw_centers"].pop(0)
            
        # Compute smoothed center (simple moving average)
        num_centers = len(hist["raw_centers"])
        smoothed_x = sum(pt[0] for pt in hist["raw_centers"]) / num_centers
        smoothed_y = sum(pt[1] for pt in hist["raw_centers"]) / num_centers
        
        hist["smoothed_centers"].append((smoothed_x, smoothed_y))
        if len(hist["smoothed_centers"]) > self.smoothing_window:
            hist["smoothed_centers"].pop(0)

        # Kinematics calculations (requires at least 2 points)
        vx, vy, speed = 0.0, 0.0, 0.0
        ax, ay, acceleration = 0.0, 0.0, 0.0
        distance_step = 0.0

        if len(hist["smoothed_centers"]) >= 2:
            prev_x, prev_y = hist["smoothed_centers"][-2]
            
            # Distance in pixels
            dx_px = smoothed_x - prev_x
            dy_px = smoothed_y - prev_y
            
            # Convert to meters
            dx_m = dx_px / pixels_per_meter
            dy_m = dy_px / pixels_per_meter
            distance_step = math.sqrt(dx_m**2 + dy_m**2)
            
            # Accumulate distance
            hist["cumulative_distance"] += distance_step
            
            # Velocities (m/s)
            vx = dx_m / dt
            vy = dy_m / dt
            speed = distance_step / dt

        # Save velocity history
        hist["velocities"].append((vx, vy))
        if len(hist["velocities"]) > self.smoothing_window:
            hist["velocities"].pop(0)

        # Acceleration calculation (requires at least 2 velocities)
        if len(hist["velocities"]) >= 2:
            prev_vx, prev_vy = hist["velocities"][-2]
            ax = (vx - prev_vx) / dt
            ay = (vy - prev_vy) / dt
            acceleration = math.sqrt(ax**2 + ay**2)

        # Smooth out the speed value to prevent transient noise triggers
        # (helpful for stable state machine transitions)
        # We can keep a history of computed speeds and average them
        # self.history[track_id]["last_speed"] can be used or we can compute average speed
        
        # Package and return results
        metrics = {
            "x": smoothed_x,
            "y": smoothed_y,
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "ax": ax,
            "ay": ay,
            "acceleration": acceleration,
            "cumulative_distance": hist["cumulative_distance"],
            "pixels_per_meter": pixels_per_meter
        }
        
        return metrics

    def reset_distance(self, track_id: int) -> None:
        """
        Resets the cumulative distance traveled for the specified track ID.
        Useful when transitioning ball states.
        """
        if track_id in self.history:
            self.history[track_id]["cumulative_distance"] = 0.0
            logger.info(f"Reset motion analyzer distance for Ball {track_id}")
