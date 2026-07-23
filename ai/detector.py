from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import os
import json
import logging
import cv2
import math
from ai.interfaces import IBallDetector
from ai.color_classifier import classify_ball_color

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
                # Automatically add target cups/holes to ignore regions to prevent cup/hole false detections
                for cup in cal_data.get("target_holes", cal_data.get("target_cups", [])):
                    if "center" in cup:
                        self.ignore_regions.append({
                            "name": "auto_cup_ignore",
                            "x": cup["center"][0],
                            "y": cup["center"][1],
                            "radius": cup.get("radius", 50.0)
                        })
                logger.info(f"YOLO loaded {len(self.ignore_regions)} ignore regions from calibration layout.")
            except Exception as e:
                logger.error(f"YOLO failed to load calibration: {e}")
        
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from: {actual_model_path}")
            self.model = YOLO(actual_model_path)
            if hasattr(self.model, "names"):
                if len(self.model.names) == 1:
                    self.class_id = 0
                    logger.info(f"[YOLO] Single-class custom model detected. Automatically overriding class_id to 0 ('{self.model.names[0]}').")
                elif len(self.model.names) > 32 and self.model.names[32] == "sports ball":
                    self.class_id = 32
                    logger.info(f"[YOLO] COCO model detected. Automatically overriding class_id to 32 ('sports ball').")
            # If we downloaded a temporary file, save it to the desired path
            if actual_model_path == "yolo11n.pt" and not os.path.exists(model_path):
                self.model.save(model_path)
                logger.info(f"Saved default model to {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise e

    def detect(self, frame: np.ndarray, **kwargs) -> List[Tuple[float, float, float, float, float, str]]:
        """
        Detects sports balls (golf balls) in the current frame.
        
        Returns:
            List of Tuple (x1, y1, x2, y2, confidence, color)
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
                        # Crop ball and classify color
                        ball_crop = frame[max(0, int(xyxy[1])):min(h, int(xyxy[3])), max(0, int(xyxy[0])):min(w, int(xyxy[2]))]
                        color = classify_ball_color(ball_crop)
                        detections.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, color))
                    
        return detections


class HybridBallDetector(IBallDetector):
    """
    Hybrid Golf Ball Detector:
    Uses TrackNetV2 for sub-pixel, motion-blur resilient positional tracking.
    Uses HSV Heuristics (classify_ball_color) on a tiny cropped ROI for color identification.
    """
    def __init__(self, tracknet_weights: str = "models/TrackNet_best.pt", conf_threshold: float = 0.1):
        from ai.tracknet_tracker import TrackNetEngine
        self.engine = TrackNetEngine(weights_path=tracknet_weights, conf_threshold=conf_threshold)
        
        self.calibration_path = "config/calibration.json"
        self.ignore_regions = []
        if os.path.exists(self.calibration_path):
            try:
                with open(self.calibration_path, "r") as f:
                    cal_data = json.load(f)
                for region in cal_data.get("ignore_regions", []):
                    if "tee" not in region.get("name", "").lower():
                        self.ignore_regions.append(region)
                for cup in cal_data.get("target_holes", cal_data.get("target_cups", [])):
                    if "center" in cup:
                        self.ignore_regions.append({
                            "name": "auto_cup_ignore",
                            "x": cup["center"][0],
                            "y": cup["center"][1],
                            "radius": cup.get("radius", 50.0)
                        })
                logger.info(f"HybridDetector loaded {len(self.ignore_regions)} ignore regions.")
            except Exception as e:
                logger.error(f"HybridDetector failed to load calibration: {e}")

    def detect(self, frame: np.ndarray, **kwargs) -> List[Tuple[float, float, float, float, float, str]]:
        h, w, _ = frame.shape
        pos, conf = self.engine.update(frame)
        
        if pos is None:
            return []
            
        cx, cy = pos
        
        # Filter out custom ignore zones
        is_ignored = False
        cal_res = [3840, 2160]
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
                    
        if is_ignored:
            return []
            
        # Create a faux bounding box (32x32) around the centroid for color extraction
        box_size = 16
        x1 = max(0, int(cx - box_size))
        y1 = max(0, int(cy - box_size))
        x2 = min(w, int(cx + box_size))
        y2 = min(h, int(cy + box_size))
        
        ball_crop = frame[y1:y2, x1:x2]
        color = classify_ball_color(ball_crop)
        
        return [(float(x1), float(y1), float(x2), float(y2), float(conf), color)]


class TrackNetOnlyDetector(IBallDetector):
    """
    TrackNet-Only Detector:
    Uses TrackNetV2 for positional tracking, skips color classification entirely
    to maximize FPS (returns 'UNKNOWN' for color).
    """
    def __init__(self, tracknet_weights: str = "models/TrackNet_best.pt", conf_threshold: float = 0.1):
        from ai.tracknet_tracker import TrackNetEngine
        self.engine = TrackNetEngine(weights_path=tracknet_weights, conf_threshold=conf_threshold)
        
        self.calibration_path = "config/calibration.json"
        self.ignore_regions = []
        if os.path.exists(self.calibration_path):
            try:
                with open(self.calibration_path, "r") as f:
                    cal_data = json.load(f)
                for region in cal_data.get("ignore_regions", []):
                    if "tee" not in region.get("name", "").lower():
                        self.ignore_regions.append(region)
                for cup in cal_data.get("target_holes", cal_data.get("target_cups", [])):
                    if "center" in cup:
                        self.ignore_regions.append({
                            "name": "auto_cup_ignore",
                            "x": cup["center"][0],
                            "y": cup["center"][1],
                            "radius": cup.get("radius", 50.0)
                        })
                logger.info(f"TrackNetOnlyDetector loaded {len(self.ignore_regions)} ignore regions.")
            except Exception as e:
                logger.error(f"TrackNetOnlyDetector failed to load calibration: {e}")

    def detect(self, frame: np.ndarray, **kwargs) -> List[Tuple[float, float, float, float, float, str]]:
        h, w, _ = frame.shape
        pos, conf = self.engine.update(frame)
        
        if pos is None:
            return []
            
        cx, cy = pos
        
        # Filter out custom ignore zones
        is_ignored = False
        cal_res = [3840, 2160]
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
                    
        if is_ignored:
            return []
            
        box_size = 16
        x1 = max(0, int(cx - box_size))
        y1 = max(0, int(cy - box_size))
        x2 = min(w, int(cx + box_size))
        y2 = min(h, int(cy + box_size))
        
        return [(float(x1), float(y1), float(x2), float(y2), float(conf), "UNKNOWN")]


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

    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float, str]]:
        """
        Returns detections matching the current frame index.
        """
        frame_key = str(self.current_frame_idx)
        if frame_key not in self.positions:
            return []
            
        detections = []
        for box in self.positions[frame_key]:
            # box format: [x1, y1, x2, y2]
            # append mock confidence of 0.99 and color 'unknown'
            if len(box) >= 4:
                detections.append((box[0], box[1], box[2], box[3], 0.99, "unknown"))
                
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

    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float, str]]:
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
                        # Return bounding box, dummy confidence, and color
                        ball_crop = frame[int(y):int(y + h), int(x):int(x + w)]
                        color = classify_ball_color(ball_crop)
                        detections.append((float(x), float(y), float(x + w), float(y + h), 0.99, color))
                    
        return detections


from ai.opencv_detector import OpenCVBallDetector

class HybridYoloBallDetector(OpenCVBallDetector):
    """
    Hybrid YOLO + OpenCV ball detector.
    Uses OpenCV's DNN module to run YOLO and detect 'person' objects (class 0) 
    in the frame, running asynchronously in a background thread to prevent lag.
    Draws black boxes over all detected persons in the playable area mask,
    then runs the 90% accurate OpenCV ball detector on the remaining frame.
    
    This keeps the main tracking loop running at full speed (0 lag) while
    updating the player mask 3-5 times per second in the background.
    """
    def __init__(self, config: Dict[str, Any], model_path: str = "models/yolo11n.onnx"):
        # Initialize the base OpenCV Ball Detector first
        super().__init__(config)
        self.yolo_net = None
        self.person_class_id = 0  # COCO class 0 is 'person'
        
        # Threading state for async person masking
        import threading
        self.lock = threading.Lock()
        self.latest_person_boxes = []  # List of [x1, y1, x2, y2]
        self.thread_running = False
        
        logger.info(f"[Hybrid YOLO] Loading YOLO ONNX model for Person Masking from: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"[Hybrid YOLO] ONNX model file not found at {model_path}. Please copy it from the laptop.")
            raise FileNotFoundError(f"ONNX Model file not found: {model_path}")
            
        try:
            # Load ONNX net natively using OpenCV DNN
            self.yolo_net = cv2.dnn.readNet(model_path)
            # RPi optimization: use CPU target (FP32) since RPi has no GPU
            self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("[Hybrid YOLO] Net loaded successfully. Ready for Async Person Masking.")
        except Exception as e:
            logger.error(f"[Hybrid YOLO] Failed to load YOLO ONNX: {e}")
            raise e

    def _run_yolo_async(self, frame_copy: np.ndarray):
        """Runs YOLO detection on a background thread."""
        try:
            h, w, _ = frame_copy.shape
            
            # Using 320x320 instead of 640x640:
            # - 4x fewer pixels, which makes inference ~3-4x faster on RPi 3 CPU!
            # - Since a human body is a large object, 320x320 is extremely accurate for person detection.
            blob = cv2.dnn.blobFromImage(frame_copy, 1/255.0, (320, 320), swapRB=True, crop=False)
            
            # Thread-safe forward pass: DNN read/write needs to be isolated
            with self.lock:
                self.yolo_net.setInput(blob)
                outputs = self.yolo_net.forward()
                
            output = outputs[0]
            output = np.transpose(output)
            
            boxes_list = []
            confidences_list = []
            
            for row in output:
                classes_scores = row[4:]
                class_id = np.argmax(classes_scores)
                conf = classes_scores[class_id]
                
                # Detect 'person' (Class 0)
                if class_id == self.person_class_id and conf >= 0.25:
                    cx_scale, cy_scale, w_scale, h_scale = row[0], row[1], row[2], row[3]
                    
                    # Scale coordinates from 320x320 input back to original frame size
                    x1 = int((cx_scale - w_scale / 2.0) * (w / 320.0))
                    y1 = int((cy_scale - h_scale / 2.0) * (h / 320.0))
                    bw = int(w_scale * (w / 320.0))
                    bh = int(h_scale * (h / 320.0))
                    
                    boxes_list.append([x1, y1, bw, bh])
                    confidences_list.append(float(conf))
            
            indices = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.25, 0.45)
            
            temp_boxes = []
            if len(indices) > 0:
                flat_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                for idx in flat_indices:
                    x1, y1, bw, bh = boxes_list[idx]
                    x2, y2 = x1 + bw, y1 + bh
                    
                    # Expand the box slightly (35 pixels) to fully catch hands/shoulders/shadows
                    x1 = max(0, x1 - 35)
                    y1 = max(0, y1 - 35)
                    x2 = min(w, x2 + 35)
                    y2 = min(h, y2 + 35)
                    temp_boxes.append((x1, y1, x2, y2))
                    
            with self.lock:
                self.latest_person_boxes = temp_boxes
                
        except Exception as e:
            logger.error(f"[Hybrid YOLO] Background inference failed: {e}")
        finally:
            self.thread_running = False

    def detect(
        self,
        frame: np.ndarray,
        hint_center: tuple = None,
        hint_moving: bool = False
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Detects the ball by using background-threaded YOLO to mask out players/hands,
        running the main OpenCV detection pipeline at full speed with 0 lag.
        """
        h, w, _ = frame.shape
        if self.mask is None:
            self._initialize_mask(h, w)
            
        # 1. Trigger background YOLO inference if the thread is idle
        if not self.thread_running and self.yolo_net is not None:
            self.thread_running = True
            # Copy frame to prevent background thread access violations
            frame_copy = frame.copy()
            import threading
            t = threading.Thread(target=self._run_yolo_async, args=(frame_copy,), daemon=True)
            t.start()
            
        # 2. Create copy of playable mask and apply the LATEST computed person boxes (0 lag!)
        dynamic_mask = self.mask.copy()
        with self.lock:
            current_boxes = list(self.latest_person_boxes)
            
        for x1, y1, x2, y2 in current_boxes:
            cv2.rectangle(dynamic_mask, (x1, y1), (x2, y2), 0, -1)
            
        # 3. Swap the active mask to the dynamic masked version
        original_mask = self.mask
        self.mask = dynamic_mask
        
        # 4. Run standard high-accuracy OpenCV detection on the masked frame (takes <2ms!)
        try:
            detections = super().detect(frame, hint_center, hint_moving)
        finally:
            self.mask = original_mask
            
        return detections


class YoloOnlyBallDetector(IBallDetector):
    """
    Pure YOLO ball detector.
    Uses PyTorch/Ultralytics backend for high-speed (30+ FPS) inference if installed (laptop),
    and falls back to OpenCV DNN for lightweight CPU inference on Raspberry Pi.
    """
    def __init__(self, config: Dict[str, Any]):
        yolo_cfg = config.get("yolo_detector", {})
        model_path = yolo_cfg.get("model_path", "models/yolo11n.onnx")
        self.class_id = yolo_cfg.get("class_id", 32)  # Default to 32 (sports ball) for COCO
        self.confidence_threshold = yolo_cfg.get("confidence_threshold", 0.05)
        self.model_path = model_path
        
        self.use_ultralytics = False
        self.model = None
        self.yolo_net = None
        
        # Try loading via Ultralytics (PyTorch) first for 30+ FPS speed on laptop
        try:
            from ultralytics import YOLO
            logger.info(f"[YOLO Only] Ultralytics detected. Loading model via PyTorch: {model_path}")
            self.model = YOLO(model_path, task="detect")
            self.use_ultralytics = True
            
            # Auto-detect class mapping: override class_id to 32 (sports ball) for COCO models, and 0 for custom single-class models
            if hasattr(self.model, "names"):
                if len(self.model.names) == 1:
                    self.class_id = 0
                    logger.info(f"[YOLO Only] Detected single-class custom model. Automatically overriding class_id to 0 ('{self.model.names[0]}').")
                elif len(self.model.names) > 32 and self.model.names[32] == "sports ball":
                    self.class_id = 32
                    logger.info(f"[YOLO Only] Detected COCO model. Automatically overriding class_id to 32 ('sports ball').")
                
            logger.info("[YOLO Only] PyTorch Model loaded successfully.")
        except ImportError:
            # Fallback to OpenCV DNN for RPi
            logger.info(f"[YOLO Only] Ultralytics not installed. Falling back to OpenCV DNN: {model_path}")
            if not os.path.exists(model_path):
                # Try root directory fallback
                if os.path.exists("yolo11n.onnx"):
                    model_path = "yolo11n.onnx"
                else:
                    logger.error(f"[YOLO Only] ONNX model file not found at {model_path}.")
                    raise FileNotFoundError(f"ONNX Model file not found: {model_path}")
            try:
                self.yolo_net = cv2.dnn.readNet(model_path)
                self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logger.info("[YOLO Only] ONNX model loaded via OpenCV DNN successfully.")
            except Exception as e:
                logger.error(f"[YOLO Only] Failed to load YOLO ONNX: {e}")
                raise e

    def detect(
        self,
        frame: np.ndarray,
        hint_center: tuple = None,
        hint_moving: bool = False
    ) -> List[Tuple[float, float, float, float, float, str]]:
        h, w, _ = frame.shape
        
        if self.use_ultralytics and self.model is not None:
            crop_active = False
            crop_x1, crop_y1 = 0, 0
            
            # If moving fast, expand crop size to prevent the ball from escaping the window
            crop_size = 320 if hint_moving else 200
            
            if hint_center is not None:
                cx, cy = hint_center
                crop_x1 = max(0, int(cx - crop_size / 2))
                crop_y1 = max(0, int(cy - crop_size / 2))
                crop_x2 = min(w, int(cx + crop_size / 2))
                crop_y2 = min(h, int(cy + crop_size / 2))
                
                if (crop_x2 - crop_x1) > 20 and (crop_y2 - crop_y1) > 20:
                    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    # Fast 160px inference on the small crop (takes < 8ms for 100+ FPS speed)
                    results = self.model(cropped_frame, imgsz=160, conf=0.15, verbose=False)
                    crop_active = True
                    
            detections = []
            if crop_active:
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        if (cls_id == self.class_id or len(results[0].names) == 1) and conf >= 0.15:
                            xyxy = box.xyxy[0].tolist()
                            bcx = (crop_x1 + xyxy[0] + crop_x1 + xyxy[2]) / 2.0
                            bcy = (crop_y1 + xyxy[1] + crop_y1 + xyxy[3]) / 2.0
                            if bcx > 600.0 or bcx < 10.0 or bcy < 10.0 or bcy > 340.0:
                                continue  # Ignore player hands/sleeves at camera edge border
                            # Crop ball and classify color
                            ball_crop = cropped_frame[max(0, int(xyxy[1])):min(cropped_frame.shape[0], int(xyxy[3])), max(0, int(xyxy[0])):min(cropped_frame.shape[1], int(xyxy[2]))]
                            color = classify_ball_color(ball_crop)
                            # Offset crop coordinates back to full frame
                            detections.append((crop_x1 + xyxy[0], crop_y1 + xyxy[1], crop_x1 + xyxy[2], crop_y1 + xyxy[3], conf, color))
                    if len(detections) > 1:
                        if hint_center is not None:
                            hc_x, hc_y = hint_center
                            detections.sort(key=lambda d: (
                                0 if d[5] != "unknown" else 1,
                                math.sqrt(((d[0] + d[2]) / 2.0 - hc_x)**2 + ((d[1] + d[3]) / 2.0 - hc_y)**2)
                            ))
                        else:
                            detections.sort(key=lambda d: (0 if d[5] != "unknown" else 1, -d[4]))
                
                # If we missed inside the crop, fall back to full frame search
                if len(detections) == 0:
                    crop_active = False
                    
            if not crop_active:
                # Initialize throttling counter if not present
                if not hasattr(self, "full_frame_counter"):
                    self.full_frame_counter = 0
                
                self.full_frame_counter += 1
                search_interval = 2 if hint_moving else 5
                
                if self.full_frame_counter % search_interval == 0:
                    eval_conf = 0.40 if not hint_moving and hint_center is not None else 0.15
                    results = self.model(frame, imgsz=320, conf=eval_conf, verbose=False)
                    if len(results) > 0:
                        boxes = results[0].boxes
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            if (cls_id == self.class_id or len(results[0].names) == 1) and conf >= eval_conf:
                                xyxy = box.xyxy[0].tolist()
                                bcx = (xyxy[0] + xyxy[2]) / 2.0
                                bcy = (xyxy[1] + xyxy[3]) / 2.0
                                if bcx > 600.0 or bcx < 10.0 or bcy < 10.0 or bcy > 340.0:
                                    continue  # Ignore player hands/sleeves at camera edge border
                                if not hint_moving and hint_center is not None:
                                    dist = math.sqrt((bcx - hint_center[0])**2 + (bcy - hint_center[1])**2)
                                    if dist > 80.0:
                                        continue
                                # Crop ball and classify color
                                ball_crop = frame[max(0, int(xyxy[1])):min(h, int(xyxy[3])), max(0, int(xyxy[0])):min(w, int(xyxy[2]))]
                                color = classify_ball_color(ball_crop)
                                detections.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, color))
            return detections
            
        else:
            # Fallback to OpenCV DNN
            if self.yolo_net is None:
                return []
                
            # Inference at 320x320 to achieve ~4 FPS on RPi 3 CPU
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
            self.yolo_net.setInput(blob)
            
            try:
                outputs = self.yolo_net.forward()
                output = outputs[0]
                output = np.transpose(output)
                
                detections = []
                for row in output:
                    classes_scores = row[4:]
                    class_id = np.argmax(classes_scores)
                    conf = classes_scores[class_id]
                    
                    # Detect 'sports ball' (Class 32)
                    if class_id == self.class_id and conf >= self.confidence_threshold:
                        cx_scale, cy_scale, w_scale, h_scale = row[0], row[1], row[2], row[3]
                        
                        # Scale coordinates back to original frame size
                        x1 = (cx_scale - w_scale / 2.0) * (w / 320.0)
                        y1 = (cy_scale - h_scale / 2.0) * (h / 320.0)
                        x2 = (cx_scale + w_scale / 2.0) * (w / 320.0)
                        y2 = (cy_scale + h_scale / 2.0) * (h / 320.0)
                        
                        # Crop ball and classify color
                        ball_crop = frame[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
                        color = classify_ball_color(ball_crop)
                        detections.append((float(x1), float(y1), float(x2), float(y2), float(conf), color))
                        
                return detections
                
                return detections
                
            except Exception as e:
                logger.error(f"[YOLO Only] Inference failed: {e}")
                return []


class OpenVINOOnlyBallDetector(IBallDetector):
    """
    Native OpenVINO golf ball detector targeting Intel Iris Xe / Intel iGPU hardware.
    Bypasses PyTorch wrappers for maximum throughput (30+ FPS).
    """
    def __init__(self, config: Dict[str, Any]):
        yolo_cfg = config.get("yolo_detector", {})
        model_dir = yolo_cfg.get("model_path", "models/multicolor_detector_model_openvino_model")
        if not model_dir.endswith("_openvino_model"):
            model_dir = "models/multicolor_detector_model_openvino_model"
            
        xml_path = os.path.join(model_dir, "multicolor_detector_model.xml")
        self.confidence_threshold = yolo_cfg.get("confidence_threshold", 0.15)
        
        import openvino as ov
        self.core = ov.Core()
        devices = self.core.available_devices
        self.target_device = "GPU" if "GPU" in devices else "CPU"
        logger.info(f"[OpenVINO Native] Loading OpenVINO model from {xml_path} targeting {self.target_device}...")
        
        compiled_model = self.core.compile_model(xml_path, self.target_device)
        self.infer_request = compiled_model.create_infer_request()
        self.output_layer = compiled_model.output(0)
        logger.info(f"[OpenVINO Native] Loaded custom single-class model from {xml_path} targeting {self.target_device} (Class 0: 'golf-ball'). COCO sports ball (Class 32) is NOT used.")

    def detect(
        self,
        frame: np.ndarray,
        hint_center: tuple = None,
        hint_moving: bool = False
    ) -> List[Tuple[float, float, float, float, float, str]]:
        h, w, _ = frame.shape
        resized = cv2.resize(frame, (640, 640))
        input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32) / 255.0
        
        self.infer_request.infer({0: input_tensor})
        outputs = self.infer_request.get_output_tensor(0).data
        
        # YOLOv8/v11 output shape: [1, 5, 8400] or [1, 6, 8400] (cx, cy, w, h, class_scores...)
        predictions = outputs[0]  # shape (5, 8400) or (6, 8400)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = np.transpose(predictions)
            
        detections = []
        scale_x = w / 640.0
        scale_y = h / 640.0
        
        boxes_list = []
        confidences = []
        
        for pred in predictions:
            cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
            raw_score = pred[4]
            conf = float(1.0 / (1.0 + np.exp(-raw_score))) if (raw_score < -0.1 or raw_score > 1.1) else float(raw_score)
            
            if conf >= 0.15:
                x1 = int((cx - bw / 2.0) * scale_x)
                y1 = int((cy - bh / 2.0) * scale_y)
                w_px = int(bw * scale_x)
                h_px = int(bh * scale_y)
                boxes_list.append([x1, y1, w_px, h_px])
                confidences.append(conf)
                
        indices = cv2.dnn.NMSBoxes(boxes_list, confidences, 0.15, 0.45)
        detections = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
            for idx in flat_indices:
                x1, y1, bw_p, bh_p = boxes_list[idx]
                x2, y2 = x1 + bw_p, y1 + bh_p
                conf = confidences[idx]
                
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w, x2), min(h, y2)
                
                if (x2_c - x1_c) > 5 and (y2_c - y1_c) > 5:
                    ball_crop = frame[y1_c:y2_c, x1_c:x2_c]
                    color = classify_ball_color(ball_crop)
                    detections.append((float(x1), float(y1), float(x2), float(y2), float(conf), color))
                    
        return detections



class CustomYoloOpenCVHybridDetector(OpenCVBallDetector):
    """
    Option A: Custom YOLO for Lock-On + OpenCV for Tracking.
    Uses custom-trained YOLO model (best.onnx) asynchronously in the background
    to find the ball when lost or during startup.
    Once locked on, uses ultra-fast OpenCV tracking (30 FPS) on the main thread.
    """
    def __init__(self, config: Dict[str, Any]):
        # Initialize the base OpenCV Ball Detector
        super().__init__(config)
        
        yolo_cfg = config.get("yolo_detector", {})
        model_path = yolo_cfg.get("model_path", "models/custom_ball_detector.onnx")
        self.class_id = yolo_cfg.get("class_id", 0)  # Custom models have class 0 for golf-ball
        self.confidence_threshold = yolo_cfg.get("confidence_threshold", 0.70)
        
        self.yolo_net = None
        import threading
        self.lock = threading.Lock()
        self.yolo_result = None  # Stores (cx, cy, box) of detected ball
        self.thread_running = False
        self.use_ultralytics = False
        self.model = None
        self.yolo_net = None
        
        logger.info(f"[YOLO Hybrid] Loading custom YOLO model from: {model_path}")
        if not os.path.exists(model_path):
            if os.path.exists("models/best.onnx"):
                model_path = "models/best.onnx"
            elif os.path.exists("best.onnx"):
                model_path = "best.onnx"
            else:
                logger.error(f"[YOLO Hybrid] Custom model not found at {model_path}")
                raise FileNotFoundError(f"Custom Model not found: {model_path}")
                
        if model_path.endswith(".pt"):
            try:
                from ultralytics import YOLO
                logger.info(f"[YOLO Hybrid] Loading via PyTorch/Ultralytics: {model_path}")
                self.model = YOLO(model_path, task="detect")
                self.use_ultralytics = True
                if hasattr(self.model, "names") and len(self.model.names) == 1:
                    self.class_id = 0
                logger.info("[YOLO Hybrid] Custom PyTorch model loaded successfully.")
            except Exception as e:
                logger.error(f"[YOLO Hybrid] Failed to load PyTorch model: {e}")
                raise e
        else:
            try:
                self.yolo_net = cv2.dnn.readNet(model_path)
                self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logger.info("[YOLO Hybrid] Custom YOLO ONNX loaded successfully. Ready for Option A.")
            except Exception as e:
                logger.error(f"[YOLO Hybrid] Failed to load custom YOLO ONNX: {e}")
                raise e

    def _run_yolo_background(self, frame_copy: np.ndarray):
        """Runs custom YOLO on a background thread to find the ball."""
        try:
            h, w, _ = frame_copy.shape
            
            if self.use_ultralytics and self.model is not None:
                results = self.model(frame_copy, imgsz=320, conf=self.confidence_threshold, verbose=False)
                best_candidate = None
                if len(results) > 0 and len(results[0].boxes) > 0:
                    best_conf = 0.0
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        if (cls_id == self.class_id or len(results[0].names) == 1) and conf >= self.confidence_threshold:
                            if conf > best_conf:
                                best_conf = conf
                                xyxy = box.xyxy[0].tolist()
                                x1, y1, x2, y2 = xyxy
                                bw = x2 - x1
                                bh = y2 - y1
                                cx = x1 + bw/2.0
                                cy = y1 + bh/2.0
                                best_candidate = (cx, cy, bw, bh, conf)
                    with self.lock:
                        self.yolo_result = best_candidate
                return
                
            # Fallback to OpenCV DNN
            if self.yolo_net is None:
                return
            blob = cv2.dnn.blobFromImage(frame_copy, 1/255.0, (320, 320), swapRB=True, crop=False)
            
            with self.lock:
                self.yolo_net.setInput(blob)
                outputs = self.yolo_net.forward()
                
            output = outputs[0]
            output = np.transpose(output)
            
            best_conf = 0.0
            best_candidate = None
            
            for row in output:
                classes_scores = row[4:]
                class_id = np.argmax(classes_scores)
                conf = classes_scores[class_id]
                
                if class_id == self.class_id and conf >= self.confidence_threshold:
                    if conf > best_conf:
                        best_conf = conf
                        cx_scale, cy_scale, w_scale, h_scale = row[0], row[1], row[2], row[3]
                        
                        # Scale back to original frame size
                        cx = cx_scale * (w / 320.0)
                        cy = cy_scale * (h / 320.0)
                        bw = w_scale * (w / 320.0)
                        bh = h_scale * (h / 320.0)
                        
                        best_candidate = (cx, cy, bw, bh, conf)
                        
            with self.lock:
                self.yolo_result = best_candidate
                
        except Exception as e:
            logger.error(f"[YOLO Hybrid] Background lock-on failed: {e}")
        finally:
            self.thread_running = False

    def detect(
        self,
        frame: np.ndarray,
        hint_center: tuple = None,
        hint_moving: bool = False
    ) -> List[Tuple[float, float, float, float, float, str]]:
        h, w, _ = frame.shape
        if self.mask is None:
            self._initialize_mask(h, w)
            
        # ===== PHASE 1: No confirmed ball yet -> Run YOLO in background to lock-on =====
        if self._own_center is None:
            if not self.thread_running:
                self.thread_running = True
                frame_copy = frame.copy()
                import threading
                t = threading.Thread(target=self._run_yolo_background, args=(frame_copy,), daemon=True)
                t.start()
                logger.info("[YOLO Hybrid] Started background custom YOLO search...")
                
            # Check if background thread has found the ball
            with self.lock:
                result = self.yolo_result
                self.yolo_result = None  # Reset after reading
                
            if result is not None:
                cx, cy, bw, bh, conf = result
                logger.info(f"[YOLO Hybrid] Lock-on SUCCESS! Ball found by custom YOLO at ({cx:.1f}, {cy:.1f}) with conf={conf:.2f}")
                
                # Mock a candidate dict to capture profile
                candidate = {
                    "cx": cx, "cy": cy,
                    "box": (cx - bw/2.0, cy - bh/2.0, cx + bw/2.0, cy + bh/2.0),
                    "area": bw * bh,
                    "circularity": 1.0,  # Custom model is trained on ball, so it is circular
                    "solidity": 1.0
                }
                
                # Convert frame to HSV to capture profile
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                x1, y1, x2, y2 = candidate["box"]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                roi_hsv = hsv[max(0,y1):max(1,y2), max(0,x1):max(1,x2)]
                roi_mask = cv2.inRange(roi_hsv, self.lower_orange, self.upper_orange)
                ball_pixels = roi_hsv[roi_mask > 0]
                
                if len(ball_pixels) > 0:
                    mean_hsv = np.mean(ball_pixels, axis=0)
                else:
                    mean_hsv = np.array([14.0, 180.0, 180.0])
                    
                profile = {
                    "area": candidate["area"],
                    "mean_h": float(mean_hsv[0]),
                    "mean_s": float(mean_hsv[1]),
                    "mean_v": float(mean_hsv[2]),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                    "circularity": 1.0,
                }
                
                self._own_center = (cx, cy)
                self._ball_profile = profile
                self._lost_frames = 0
                b = candidate["box"]
                ball_crop = frame[max(0, int(b[1])):min(h, int(b[3])), max(0, int(b[0])):min(w, int(b[2]))]
                color = classify_ball_color(ball_crop)
                return [(b[0], b[1], b[2], b[3], conf, color)]
                    
            return []
            
        # ===== PHASE 2: Ball is confirmed -> Run ultra-fast OpenCV tracking (30 FPS) =====
        return super().detect(frame, hint_center, hint_moving)


