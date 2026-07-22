from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import os
import json
import logging
import cv2
from ai.interfaces import IBallDetector

logger = logging.getLogger(__name__)

class YoloBallDetector(IBallDetector):
    """
    YOLO-based golf ball detector using Ultralytics YOLO model.
    """
    def __init__(self, model_path: str = "models/yolov11.pt", class_id: int = 32, confidence_threshold: float = 0.25):
        """
        Initializes the YOLO ball detector.
        
        Args:
            model_path: Path to the YOLO model file (.pt). If not found, it downloads a default model.
            class_id: The COCO class index for the ball (default 32 is 'sports ball').
            confidence_threshold: The confidence threshold for detections.
        """
        self.model_path = model_path
        self.class_id = class_id
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        
        # If the file path is default and doesn't exist, we will use a small model (e.g. yolo11n.pt)
        # to download and run fast, otherwise it will try to download yolov11.pt.
        # Note: in Ultralytics YOLOv11 series, the name is 'yolo11n.pt', 'yolo11s.pt', etc.
        # If model_path is "models/yolov11.pt" and doesn't exist, we'll download 'yolo11n.pt'
        # as a lightweight model and save it to models/yolov11.pt.
        actual_model_path = model_path
        if not os.path.exists(model_path):
            if "yolov11.pt" in model_path or "yolo11" in model_path:
                logger.info(f"Model file {model_path} not found. Will download default yolo11n.pt.")
                # We can load yolo11n.pt directly, it will auto-download
                actual_model_path = "yolo11n.pt"
            else:
                logger.info(f"Model file {model_path} not found. Using default coco weight download.")
        
        self.calibration_path = "config/calibration.json"
        self.ignore_regions = []
        if os.path.exists(self.calibration_path):
            try:
                with open(self.calibration_path, "r") as f:
                    cal_data = json.load(f)
                # Load all ignore regions except Tees (since we want to detect the ball on the Tee)
                for region in cal_data.get("ignore_regions", []):
                    if "tee" not in region.get("name", "").lower():
                        self.ignore_regions.append(region)
                logger.info(f"YOLO loaded {len(self.ignore_regions)} ignore regions from calibration layout.")
            except Exception as e:
                logger.error(f"YOLO failed to load calibration: {e}")
        
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from: {actual_model_path}")
            self.model = YOLO(actual_model_path)
            # If we downloaded a temporary file, save it to the desired path
            if actual_model_path == "yolo11n.pt" and not os.path.exists(model_path):
                self.model.save(model_path)
                logger.info(f"Saved default model to {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise e

    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Detects sports balls (golf balls) in the current frame.
        
        Returns:
            List of Tuple (x1, y1, x2, y2, confidence)
        """
        if self.model is None:
            return []
            
        h, w, _ = frame.shape
        
        # Run YOLO inference
        # verbose=False reduces console spam
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Filter by class ID (32: sports ball) and confidence threshold
                if cls_id == self.class_id and conf >= self.confidence_threshold:
                    xyxy = box.xyxy[0].tolist()
                    cx = (xyxy[0] + xyxy[2]) / 2.0
                    cy = (xyxy[1] + xyxy[3]) / 2.0
                    
                    # Filter out custom ignore zones (Holes/obstacles)
                    is_ignored = False
                    
                    # Load calibration resolution to scale coordinates correctly
                    cal_res = [3840, 2160] # default
                    if os.path.exists(self.calibration_path):
                        try:
                            with open(self.calibration_path, "r") as f:
                                cal_res = json.load(f).get("source_resolution", [3840, 2160])
                        except:
                            pass
                    
                    scale_x = w / cal_res[0]
                    scale_y = h / cal_res[1]
                    
                    for region in self.ignore_regions:
                        if region.get("type") == "polygon" or "points" in region:
                            poly_pts = region["points"]
                            scaled_pts = np.array([[int(p[0] * scale_x), int(p[1] * scale_y)] for p in poly_pts], dtype=np.int32)
                            # Check if center of detection falls inside the polygon obstacle
                            if cv2.pointPolygonTest(scaled_pts, (cx, cy), False) >= 0:
                                is_ignored = True
                                break
                        else:
                            rx = region["x"] * scale_x
                            ry = region["y"] * scale_y
                            rr = region["radius"] * min(scale_x, scale_y)
                            dist = ((cx - rx)**2 + (cy - ry)**2)**0.5
                            if dist <= rr:
                                is_ignored = True
                                break
                            
                    if not is_ignored:
                        detections.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf))
                    
        return detections


class MockBallDetector(IBallDetector):
    """
    Mock detector that returns pre-recorded positions.
    Useful for testing without neural network models or GPUs.
    """
    def __init__(self, oracle_path: str = "assets/oracle_positions.json"):
        self.oracle_path = oracle_path
        self.positions: Dict[str, List[List[float]]] = {}
        self.current_frame_idx = 0
        
        if os.path.exists(oracle_path):
            try:
                with open(oracle_path, "r") as f:
                    self.positions = json.load(f)
                logger.info(f"Loaded {len(self.positions)} mock frame coordinates from {oracle_path}")
            except Exception as e:
                logger.error(f"Failed to load mock oracle positions: {e}")
        else:
            logger.warning(f"Mock oracle file not found at {oracle_path}. Mock detector will return empty list.")

    def set_frame_index(self, frame_idx: int) -> None:
        """Sets the active frame index to read from."""
        self.current_frame_idx = frame_idx

    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Returns detections matching the current frame index.
        """
        frame_key = str(self.current_frame_idx)
        if frame_key not in self.positions:
            return []
            
        detections = []
        for box in self.positions[frame_key]:
            # box format: [x1, y1, x2, y2]
            # append mock confidence of 0.99
            if len(box) >= 4:
                detections.append((box[0], box[1], box[2], box[3], 0.99))
                
        return detections


class ClassicBallDetector(IBallDetector):
    """
    Classic computer vision ball detector using HSV thresholding and contour analysis.
    Filters out large circles (holes/Tees) and keeps ball-sized candidates.
    Supports configuration-defined ignore zones.
    """
    def __init__(self, config: Dict[str, Any]):
        classic_cfg = config.get("classic_detector", {})
        self.min_area = classic_cfg.get("min_area", 15.0)
        self.max_area = classic_cfg.get("max_area", 300.0)
        self.circularity_thresh = classic_cfg.get("circularity_thresh", 0.4)
        
        # ignore_zones format: list of [cx, cy, radius]
        self.ignore_zones = classic_cfg.get("ignore_zones", [])
        logger.info(f"ClassicBallDetector initialized. Min area: {self.min_area}, Max area: {self.max_area}, Ignore zones: {len(self.ignore_zones)}")

    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Threshold for white/bright objects
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 80, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
                
                # Check aspect ratio and circularity to filter out lines and irregular shapes
                if 0.5 <= aspect_ratio <= 2.0 and circularity >= self.circularity_thresh:
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    
                    # Filter out static ignore zones (Tees and holes)
                    is_static = False
                    for iz in self.ignore_zones:
                        # iz format: [cx, cy, radius]
                        iz_cx, iz_cy, iz_r = iz
                        dist = ((cx - iz_cx)**2 + (cy - iz_cy)**2)**0.5
                        if dist < iz_r:
                            is_static = True
                            break
                            
                    if not is_static:
                        # Return bounding box and a dummy confidence of 0.99
                        detections.append((float(x), float(y), float(x + w), float(y + h), 0.99))
                    
        return detections
