import cv2
import numpy as np
from typing import Dict, Any, List
from engine.state_machine import BallState
from ai.motion import MotionAnalyzer

class Visualizer:
    """
    Renders annotations, HUD metrics, state indicators, and motion trajectories 
    on video frames using OpenCV.
    """
    def __init__(self, motion_analyzer: MotionAnalyzer, config: Dict[str, Any] = None):
        self.motion_analyzer = motion_analyzer
        
        # Color mapping (BGR)
        self.colors = {
            BallState.UNKNOWN: (160, 160, 160),  # Muted Gray
            BallState.STOPPED: (230, 120, 40),   # Cool Blue-Indigo
            BallState.READY: (0, 215, 255),      # Gold / Warm Yellow
            BallState.MOVING: (50, 205, 50)      # Vivid Lime Green
        }
        
        # Load calibration regions for drawing
        self.base_res = [3840, 2160]
        self.tee_region = None
        self.cup_regions = []
        self.mirror_x = False
        
        video_input = config.get("video", {}).get("input", "") if config else ""
        if "left" in os.path.basename(video_input).lower():
            self.mirror_x = True
            
        import os
        import json
        calibration_path = "config/calibration.json"
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, "r") as f:
                    cal = json.load(f)
                self.base_res = cal.get("source_resolution", [3840, 2160])
                for region in cal.get("ignore_regions", []):
                    name = region.get("name", "").lower()
                    if "tee" in name:
                        self.tee_region = region
                    elif "hole" in name or "cup" in name:
                        self.cup_regions.append(region)
            except Exception as e:
                print(f"[Visualizer] Error loading calibration: {e}")

    def draw(self, frame: np.ndarray, active_metrics: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draws boxes, HUD labels, and motion trails on the video frame.
        
        Args:
            frame: Original BGR video frame.
            active_metrics: List of metrics for currently tracked balls.
            
        Returns:
            Annotated frame.
        """
        annotated_frame = frame.copy()
        h, w, _ = annotated_frame.shape
        font = cv2.FONT_HERSHEY_SIMPLEX

        for ball in active_metrics:
            track_id = ball["track_id"]
            bbox = ball["bbox"]
            state = ball["state"]
            speed = ball["speed"]
            stroke_count = ball["stroke_count"]
            curr_x, curr_y = ball["x"], ball["y"]
            
            color = self.colors.get(state, (255, 255, 255))
            x1, y1, x2, y2 = map(int, bbox)

            # 1. Draw Trajectory Trail
            # Pull historical points from the motion analyzer
            hist = self.motion_analyzer.history.get(track_id, {})
            smoothed_centers = hist.get("smoothed_centers", [])
            
            if len(smoothed_centers) > 1:
                # Draw trailing segments with fading thickness/alpha
                num_points = len(smoothed_centers)
                for i in range(1, num_points):
                    pt1 = tuple(map(int, smoothed_centers[i-1]))
                    pt2 = tuple(map(int, smoothed_centers[i]))
                    
                    # Compute fading thickness
                    thickness = int(max(1, (i / num_points) * 5))
                    # Draw directly
                    cv2.line(annotated_frame, pt1, pt2, color, thickness, cv2.LINE_AA)

            # 2. Draw Tight Bounding Box around the ball (clamped to tight 26x26px square)
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w > 26 or box_h > 26:
                x1 = int(curr_x - 13)
                y1 = int(curr_y - 13)
                x2 = int(curr_x + 13)
                y2 = int(curr_y + 13)
                
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            
            # Draw center dot
            cv2.circle(annotated_frame, (int(curr_x), int(curr_y)), 3, color, -1, cv2.LINE_AA)

            # 3. Draw Premium HUD Overlay Panel
            # Build HUD text lines
            color_name = ball.get("color", "unknown").capitalize()
            placed = ball.get("placed_on_tee", True)
            tee_status = "ENABLED" if placed else "REQUIRED (Disabled)"
            lines = [
                f"BALL #{track_id} ({color_name})",
                f"TEE STATUS: {tee_status}",
                f"STATE: {state.value}",
                f"SPEED: {speed:.2f} m/s",
                f"STROKES: {stroke_count}"
            ]
            
            # Position the HUD box just above the ball bounding box (if space permits) or below
            font_scale = 0.45
            thickness = 1
            line_height = 18
            padding = 8
            
            # Calculate panel dimensions
            max_width = 0
            for line in lines:
                (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
                if tw > max_width:
                    max_width = tw
                    
            panel_w = max_width + (padding * 2)
            panel_h = (line_height * len(lines)) + (padding * 2)
            
            # Position HUD relative to ball box
            hud_x = x1
            hud_y = y1 - panel_h - 10
            
            # Boundary checks
            if hud_y < 10:
                hud_y = y2 + 10  # Flip below box
            if hud_x + panel_w > w:
                hud_x = w - panel_w - 10
            if hud_x < 10:
                hud_x = 10
                
            # Draw HUD background card (dark semi-transparent rounded rectangle representation)
            # Create a localized mask for alpha-blending the black background
            hud_bg = annotated_frame.copy()
            cv2.rectangle(
                hud_bg, 
                (hud_x, hud_y), 
                (hud_x + panel_w, hud_y + panel_h), 
                (20, 20, 20), 
                -1
            )
            # Draw a thin border matching ball state color on the HUD card
            cv2.rectangle(
                hud_bg, 
                (hud_x, hud_y), 
                (hud_x + panel_w, hud_y + panel_h), 
                color, 
                1,
                cv2.LINE_AA
            )
            # Blend back into frame
            cv2.addWeighted(hud_bg, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            
            # Write text lines on the HUD
            for j, line in enumerate(lines):
                # Title gets colored text, other metrics get white text
                text_color = color if j == 0 else (245, 245, 245)
                ty = hud_y + padding + (j * line_height) + 12
                tx = hud_x + padding
                cv2.putText(annotated_frame, line, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)

        # 4. Draw global telemetry banner in top-left corner
        # Let's add a neat general status banner
        banner_w, banner_h = 240, 45
        banner_bg = annotated_frame.copy()
        cv2.rectangle(banner_bg, (15, 15), (15 + banner_w, 15 + banner_h), (10, 10, 10), -1)
        cv2.rectangle(banner_bg, (15, 15), (15 + banner_w, 15 + banner_h), (100, 100, 100), 1, cv2.LINE_AA)
        cv2.addWeighted(banner_bg, 0.8, annotated_frame, 0.2, 0, annotated_frame)
        
        cv2.putText(
            annotated_frame, 
            "MINI GOLF AI STROKE POC", 
            (25, 32), 
            font, 0.4, (200, 200, 200), 1, cv2.LINE_AA
        )
        
        # Count total strokes across all active state machines
        active_balls = len(active_metrics)
        calibration_val = active_metrics[0].get("pixels_per_meter", 200.0) if active_balls > 0 else 200.0
        cv2.putText(
            annotated_frame, 
            f"Active Balls: {active_balls} | Calibration: {calibration_val:.0f} px/m", 
            (25, 48), 
            font, 0.35, (120, 180, 120), 1, cv2.LINE_AA
        )

        # 5. Draw real-time "STROKE DETECTED" toast overlay at the top-center
        for ball in active_metrics:
            track_id = ball["track_id"]
            stroke_count = ball["stroke_count"]
            
            # Initialize tracker for stroke changes
            if not hasattr(self, "_last_strokes"):
                self._last_strokes = {}
            if not hasattr(self, "_toast_timer"):
                self._toast_timer = {}
                
            # If a new stroke was detected, trigger the toast timer (90 frames ~ 3 seconds)
            if track_id in self._last_strokes and stroke_count > self._last_strokes[track_id]:
                self._toast_timer[track_id] = 90
            self._last_strokes[track_id] = stroke_count
            
            # Render the toast if the timer is active
            if self._toast_timer.get(track_id, 0) > 0:
                self._toast_timer[track_id] -= 1
                
                # Draw high-visibility toast banner
                toast_text = f"STROKE DETECTED! BALL #{track_id} - Total: {stroke_count}"
                (tw, th), _ = cv2.getTextSize(toast_text, font, 0.8, 2)
                
                toast_w = tw + 40
                toast_h = th + 25
                toast_x = (w - toast_w) // 2
                toast_y = 30
                
                toast_bg = annotated_frame.copy()
                # Cyan/Gold warning box
                cv2.rectangle(toast_bg, (toast_x, toast_y), (toast_x + toast_w, toast_y + toast_h), (0, 10, 0), -1)
                cv2.rectangle(toast_bg, (toast_x, toast_y), (toast_x + toast_w, toast_y + toast_h), (0, 215, 255), 3, cv2.LINE_AA)
                cv2.addWeighted(toast_bg, 0.85, annotated_frame, 0.15, 0, annotated_frame)
                
                cv2.putText(
                    annotated_frame, 
                    toast_text, 
                    (toast_x + 20, toast_y + toast_h - 12), 
                    font, 0.8, (0, 255, 255), 2, cv2.LINE_AA
                )

        # 6. Draw Tee and Cup overlay reference markers on the field
        # Calculate dynamic scaling based on current frame resolution
        scale_x = w / self.base_res[0]
        scale_y = h / self.base_res[1]
        
        # Draw Tee 1 marker if loaded
        if self.tee_region is not None:
            raw_x = self.tee_region["x"]
            if self.mirror_x:
                raw_x = self.base_res[0] - raw_x
            tee_x = int(raw_x * scale_x)
            tee_y = int(self.tee_region["y"] * scale_y)
            tee_r = int(self.tee_region.get("radius", 25.0) * scale_x)
            cv2.circle(annotated_frame, (tee_x, tee_y), tee_r, (0, 215, 255), 2, cv2.LINE_AA)  # Precise Tee circle
            cv2.circle(annotated_frame, (tee_x, tee_y), 2, (0, 215, 255), -1, cv2.LINE_AA)   # Tee center dot
            cv2.putText(annotated_frame, f"TEE ({tee_x}, {tee_y})", (tee_x - 35, tee_y - tee_r - 5), font, 0.3, (0, 215, 255), 1, cv2.LINE_AA)
            
        # Draw Cup/Hole markers if loaded
        for region in self.cup_regions:
            raw_x = region["x"]
            if self.mirror_x:
                raw_x = self.base_res[0] - raw_x
            cup_x = int(raw_x * scale_x)
            cup_y = int(region["y"] * scale_y)
            cup_r = int(region.get("radius", 18.0) * scale_x)
            cv2.circle(annotated_frame, (cup_x, cup_y), cup_r, (255, 0, 255), 2, cv2.LINE_AA)  # Precise Cup circle
            cv2.circle(annotated_frame, (cup_x, cup_y), 2, (255, 0, 255), -1, cv2.LINE_AA)   # Cup center point
            cv2.putText(annotated_frame, f"CUP ({cup_x}, {cup_y})", (cup_x - 30, cup_y - cup_r - 5), font, 0.3, (255, 0, 255), 1, cv2.LINE_AA)

        return annotated_frame
