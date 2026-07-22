import csv
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CsvExporter:
    """
    Exports stroke events into a CSV file.
    """
    def __init__(self, output_path: str):
        """
        Initializes the CSV exporter.
        
        Args:
            output_path: Absolute or relative path to the CSV file.
        """
        self.output_path = output_path
        # Ensure output directory exists
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        
        # Initialize file with header if it doesn't exist
        self._initialize_header()

    def _initialize_header(self) -> None:
        try:
            file_exists = os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0
            if not file_exists:
                with open(self.output_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Stroke Number", 
                        "Ball ID", 
                        "Frame", 
                        "Timestamp", 
                        "Speed", 
                        "Distance"
                    ])
                logger.info(f"Initialized CSV file with headers at: {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to initialize CSV headers: {e}")

    def log_stroke(self, stroke_number: int, ball_id: int, frame: int, timestamp: float, speed: float, distance: float) -> None:
        """
        Appends a stroke record to the CSV file.
        
        Args:
            stroke_number: Count of strokes for this ball.
            ball_id: Tracking ID of the ball.
            frame: Frame index when the stroke was registered.
            timestamp: Seconds relative to video start.
            speed: Stroke speed in m/s.
            distance: Travel distance in meters.
        """
        try:
            with open(self.output_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    stroke_number,
                    ball_id,
                    frame,
                    f"{timestamp:.3f}",
                    f"{speed:.2f}",
                    f"{distance:.2f}"
                ])
            logger.info(f"Logged stroke {stroke_number} for ball {ball_id} in CSV.")
        except Exception as e:
            logger.error(f"Failed to write stroke to CSV: {e}")
