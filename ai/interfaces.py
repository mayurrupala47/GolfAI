from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np

class IBallDetector(ABC):
    """
    Interface for detecting golf balls in a video frame.
    """
    @abstractmethod
    def detect(self, frame: np.ndarray, hint_center: tuple = None, hint_moving: bool = False) -> List[Tuple[float, float, float, float, float, str]]:
        """
        Detect golf balls in a frame.
        
        Args:
            frame: A numpy array representing the image frame.
            hint_center: Optional (cx, cy) of last known ball position (from tracker).
                         Implementations may use this to restrict search to an ROI for speed.
            hint_moving: If True, the ball was recently in motion — use wider search area.
            
        Returns:
            A list of detections where each detection is a tuple of
            (x1, y1, x2, y2, confidence, color).
        """
        pass


class IBallTracker(ABC):
    """
    Interface for tracking detected golf balls across frames.
    """
    @abstractmethod
    def update(self, detections: List[Tuple[float, float, float, float, float, str]], frame: np.ndarray = None) -> List[Tuple[float, float, float, float, int, str]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence, color) detections.
            frame: Optional frame image (for visual tracking algorithms).
            
        Returns:
            List of active tracked objects as (x1, y1, x2, y2, track_id, color).
        """
        pass


class IMotionAnalyzer(ABC):
    """
    Interface for tracking motion metrics of golf balls.
    """
    @abstractmethod
    def update(self, track_id: int, bbox: Tuple[float, float, float, float], fps: float, pixels_per_meter: float) -> Dict[str, Any]:
        """
        Process the new coordinates and calculate motion parameters.
        
        Args:
            track_id: The tracking ID of the ball.
            bbox: Bounding box as (x1, y1, x2, y2).
            fps: Frame rate of the video.
            pixels_per_meter: Scale factor.
            
        Returns:
            A dictionary containing motion metrics:
            - "x": current center X (pixels)
            - "y": current center Y (pixels)
            - "vx": velocity X (m/s)
            - "vy": velocity Y (m/s)
            - "speed": speed (m/s)
            - "ax": acceleration X (m/s^2)
            - "ay": acceleration Y (m/s^2)
            - "acceleration": acceleration magnitude (m/s^2)
            - "cumulative_distance": total distance traveled in current session (meters)
        """
        pass
    
    @abstractmethod
    def reset_distance(self, track_id: int) -> None:
        """
        Reset cumulative distance tracking for a ball (e.g. when state transitions).
        """
        pass


class IMqttPublisher(ABC):
    """
    Interface for publishing telemetry data to an MQTT broker.
    """
    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to the MQTT broker.
        """
        pass

    @abstractmethod
    def publish_stroke(self, camera_id: str, hole: int, ball_id: int, stroke_count: int, timestamp: str, ball_color: str = "unknown") -> None:
        """
        Publish a stroke event.
        """
        pass

    @abstractmethod
    def publish_hole_complete(self, camera_id: str, hole: int, ball_id: int, stroke_count: int, timestamp: str, ball_color: str = "unknown") -> None:
        """
        Publish a hole-completed event (ball entered the cup).
        """
        pass

    @abstractmethod
    def publish_reset(self, camera_id: str, hole: int, ball_id: int, timestamp: str, ball_color: str = "unknown") -> None:
        """
        Publish a tee-reset event (ball returned to tee, stroke counter cleared).
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Disconnect from the MQTT broker.
        """
        pass
