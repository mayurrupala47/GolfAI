import json
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class JsonExporter:
    """
    Exports stroke events into a JSON file as an array of stroke event objects.
    """
    def __init__(self, output_path: str):
        """
        Initializes the JSON exporter.
        
        Args:
            output_path: Absolute or relative path to the JSON file.
        """
        self.output_path = output_path
        self.events: List[Dict[str, Any]] = []
        
        # Ensure output directory exists
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        
        # Clean/Initialize the file or load existing events if needed
        self._initialize_file()

    def _initialize_file(self) -> None:
        try:
            # Overwrite with empty array initially for a clean POC run
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)
            logger.info(f"Initialized empty JSON file at: {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to initialize JSON file: {e}")

    def log_stroke(self, stroke_number: int, ball_id: int, frame: int, timestamp: float, speed: float) -> None:
        """
        Logs a stroke event, appends it to the list, and writes the entire array to disk.
        
        Args:
            stroke_number: Cumulative stroke number for this ball.
            ball_id: Tracking ID of the ball.
            frame: Frame index when the stroke occurred.
            timestamp: Time offset in seconds.
            speed: Stroke speed in m/s.
        """
        event = {
            "stroke": stroke_number,
            "ball": ball_id,
            "frame": frame,
            "time": round(timestamp, 2),
            "speed": round(speed, 2)
        }
        
        self.events.append(event)
        
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self.events, f, indent=2)
            logger.info(f"Logged stroke {stroke_number} for ball {ball_id} in JSON.")
        except Exception as e:
            logger.error(f"Failed to write stroke to JSON: {e}")
