import cv2
import numpy as np
import os
import json
import logging
from typing import List, Tuple, Dict, Any
from ai.interfaces import IBallDetector

logger = logging.getLogger(__name__)

class OpenCVBallDetector(IBallDetector):
    """
    OpenCV-based classic machine vision ball detector.
    Uses MOG2 background subtraction, ROI/playable area masking, 
    contour/circularity analysis, and HSV color filtering.
    """
    def __init__(self, config: Dict[str, Any], calibration_path: str = "config/calibration.json"):
        classic_cfg = config.get("classic_detector", {})
        self.min_area = classic_cfg.get("min_area", 30.0)
        self.max_area = classic_cfg.get("max_area", 100.0)
        self.circularity_thresh = classic_cfg.get("circularity_thresh", 0.55)
        
        # Load expected diameter bounds
        ball_cfg = config.get("ball", {})
        self.expected_diameter = ball_cfg.get("expected_diameter", 18.0)
        self.diameter_tolerance = ball_cfg.get("diameter_tolerance", 8.0)
        
        self.calibration_path = calibration_path
        self.calibration_data = {}
        self.mask = None
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Load calibration coordinates
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, "r") as f:
                    self.calibration_data = json.load(f)
                logger.info(f"Loaded calibration data from {calibration_path}")
            except Exception as e:
                logger.error(f"Failed to load calibration json: {e}")
        else:
            logger.warning(f"Calibration file not found at {calibration_path}. Processing whole frame.")

    def _initialize_mask(self, h: int, w: int):
        """Creates the combined binary mask scaled to current frame size."""
        self.mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Scale factor relative to the video resolution used during calibration
        base_res = self.calibration_data.get("source_resolution", [3840, 2160])
        scale_x = w / base_res[0]
        scale_y = h / base_res[1]
        
        # 1. Apply Playable Area Mask
        playable_area = self.calibration_data.get("playable_area", [])
        if len(playable_area) >= 3:
            pts = np.array([[int(p[0] * scale_x), int(p[1] * scale_y)] for p in playable_area], dtype=np.int32)
            pa_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(pa_mask, [pts], 255)
            self.mask = cv2.bitwise_and(self.mask, pa_mask)
            
        # 2. Apply Ignore Regions Mask (Holes, Tees, Pipes, Obstacles)
        ignore_regions = self.calibration_data.get("ignore_regions", [])
        for region in ignore_regions:
            if "tee" in region.get("name", "").lower():
                continue
            
            # Handle Polygon Obstacles
            if region.get("type") == "polygon" or "points" in region:
                poly_pts = region["points"]
                scaled_pts = np.array(
                    [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in poly_pts],
                    dtype=np.int32
                )
                cv2.fillPoly(self.mask, [scaled_pts], 0)
                logger.info(f"Applied polygon ignore region mask: {region.get('name')}")
            # Handle Circle Obstacles (Tees, Cups)
            else:
                rx = int(region["x"] * scale_x)
                ry = int(region["y"] * scale_y)
                rr = int(region["radius"] * min(scale_x, scale_y))
                cv2.circle(self.mask, (rx, ry), rr, 0, -1)
                logger.info(f"Applied circle ignore region mask: {region.get('name')}")
            
        logger.info(f"Initialized scaled mask with size {w}x{h}")

    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        h, w, _ = frame.shape
        if self.mask is None:
            self._initialize_mask(h, w)
            
        # 1. Mask Frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=self.mask)
        
        # 2. Motion Segmentation
        fg_mask = self.back_sub.apply(masked_frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        
        # Cheap motion gating early exit
        motion_pixels = cv2.countNonZero(fg_mask)
        if motion_pixels < 30:
            return []
        
        # 3. Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Area filter
            if self.min_area <= area <= self.max_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh
                
                # Check if the center of detection falls inside the ignore mask
                cx = x + bw / 2.0
                cy = y + bh / 2.0
                if self.mask[int(cy), int(cx)] == 0:
                    continue
                if 0.5 <= aspect_ratio <= 2.0:
                    # Circularity Filter
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
                    
                    if circularity >= self.circularity_thresh:
                        # Expected diameter check
                        avg_diam = (bw + bh) / 2.0
                        if abs(avg_diam - self.expected_diameter) <= self.diameter_tolerance:
                            # Color Filter: check if white in HSV
                            roi = frame[y:y+bh, x:x+bw]
                            if roi.size > 0:
                                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                                mean_hsv = cv2.mean(hsv_roi)
                                # Relaxed HSV color filter: allows white/grey balls in dim/tinted indoor lighting
                                if mean_hsv[1] <= 85 and mean_hsv[2] >= 75:
                                    detections.append((float(x), float(y), float(x + bw), float(y + bh), 0.99))
                                    
        return detections
