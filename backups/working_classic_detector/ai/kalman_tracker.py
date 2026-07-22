import numpy as np
import logging
from typing import List, Tuple, Dict, Any
from ai.interfaces import IBallTracker

logger = logging.getLogger(__name__)

class KalmanFilter:
    """
    Simple 2D Kalman Filter for tracking position (x, y) and velocity (vx, vy).
    """
    def __init__(self, dt: float = 1.0/60.0):
        # State: [x, y, vx, vy]
        self.state = np.zeros(4, dtype=np.float32)
        # Covariance matrix
        self.P = np.eye(4, dtype=np.float32) * 500.0
        # Transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        # Measurement noise
        self.R = np.eye(2, dtype=np.float32) * 2.0
        # Process noise
        self.Q = np.eye(4, dtype=np.float32) * 0.2

    def initialize(self, x: float, y: float):
        self.state = np.array([x, y, 0, 0], dtype=np.float32)

    def predict(self) -> Tuple[float, float]:
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.state[0]), float(self.state[1])

    def update(self, x: float, y: float):
        z = np.array([x, y], dtype=np.float32)
        y_err = z - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y_err)
        self.P = np.dot(np.eye(4, dtype=np.float32) - np.dot(K, self.H), self.P)


class KalmanBallTracker(IBallTracker):
    """
    SORT-like Kalman Filter tracker for tracking multiple golf balls.
    Restricts detection matching to a search window around predicted Kalman position.
    """
    def __init__(self, config: Dict[str, Any], fps: int = 60):
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 60.0
        
        tracking_cfg = config.get("tracking", {})
        self.max_lost_frames = tracking_cfg.get("max_lost_frames", 20)
        self.search_window = tracking_cfg.get("search_window", 40.0)
        
        self.next_track_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}
        
        import os
        import json
        self.tee_point = None
        calibration_path = "config/calibration.json"
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, "r") as f:
                    cal = json.load(f)
                for region in cal.get("ignore_regions", []):
                    if "tee" in region.get("name", "").lower():
                        rx = region["x"]
                        ry = region["y"]
                        resize_width = config.get("processing", {}).get("resize_width", 1280)
                        scale_x = resize_width / 3840.0
                        scale_y = (resize_width * 2160 / 3840) / 2160.0
                        self.tee_point = (rx * scale_x, ry * scale_y)
                        logger.info(f"Loaded scaled Tee point for tracker gating: {self.tee_point}")
                        break
            except Exception as e:
                logger.error(f"Failed to load Tee point for tracker gating: {e}")
                
        # Pre-register Ball ID 1 at the Tee point on initialization
        if self.tee_point is not None:
            kf = KalmanFilter(self.dt)
            kf.initialize(self.tee_point[0], self.tee_point[1])
            x1 = self.tee_point[0] - 9.0
            y1 = self.tee_point[1] - 9.0
            x2 = self.tee_point[0] + 9.0
            y2 = self.tee_point[1] + 9.0
            self.tracks[1] = {
                "kf": kf,
                "bbox": (x1, y1, x2, y2),
                "disappeared": 0
            }
            self.next_track_id = 2
            logger.info(f"Pre-registered Ball ID 1 at Tee point: {self.tee_point}")
        else:
            self.next_track_id = 1
            
        logger.info(f"KalmanBallTracker initialized. Search Window: {self.search_window}px, Max Lost: {self.max_lost_frames}")

    def update(self, detections: List[Tuple[float, float, float, float, float]], frame: np.ndarray = None, track_states: Dict[int, Any] = None) -> List[Tuple[float, float, float, float, int]]:
        # Compute predicted position for all current tracks
        predictions = {}
        for tid, track in self.tracks.items():
            pred_x, pred_y = track["kf"].predict()
            predictions[tid] = (pred_x, pred_y)
            
        # Parse new detections into centroids
        input_dets = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            input_dets.append({"bbox": (x1, y1, x2, y2), "center": (cx, cy)})
            
        # Match tracks with detections within search window
        used_detections = set()
        matched_tids = set()
        
        for tid, pred in predictions.items():
            px, py = pred
            
            # Determine dynamic search window based on track's Kalman velocity, disappeared state, and engine state
            kf = self.tracks[tid]["kf"]
            vx = float(kf.state[2])
            vy = float(kf.state[3])
            speed_px = (vx**2 + vy**2)**0.5
            
            disappeared = self.tracks[tid].get("disappeared", 0)
            
            # Fixed search window of 200px at all times to bridge player occlusion during hits.
            window = 200.0
            
            best_det_idx = -1
            best_dist = window
            
            for idx, det in enumerate(input_dets):
                if idx in used_detections:
                    continue
                    
                cx, cy = det["center"]
                dist = ((cx - px)**2 + (cy - py)**2)**0.5
                
                # Check search window limit
                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = idx
                    
            if best_det_idx != -1:
                det = input_dets[best_det_idx]
                self.tracks[tid]["kf"].update(det["center"][0], det["center"][1])
                self.tracks[tid]["bbox"] = det["bbox"]
                self.tracks[tid]["disappeared"] = 0
                used_detections.add(best_det_idx)
                matched_tids.add(tid)
                
        # Handle disappeared tracks
        for tid in list(self.tracks.keys()):
            if tid not in matched_tids:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_lost_frames:
                    del self.tracks[tid]
                else:
                    # Decay velocity components in Kalman filter state to model friction/stopping (5% decay per frame)
                    self.tracks[tid]["kf"].state[2] *= 0.95
                    self.tracks[tid]["kf"].state[3] *= 0.95
                    
                    # Get current position from state
                    px = float(self.tracks[tid]["kf"].state[0])
                    py = float(self.tracks[tid]["kf"].state[1])
                    
                    x1, y1, x2, y2 = self.tracks[tid]["bbox"]
                    bw = x2 - x1
                    bh = y2 - y1
                    self.tracks[tid]["bbox"] = (px - bw/2.0, py - bh/2.0, px + bw/2.0, py + bh/2.0)
            
        # Register new tracks for unmatched detections
        # (Disabled for single-ball POC since Ball ID 1 is pre-registered at Frame 0)
        pass
                
        # Format active tracks
        active_tracks = []
        for tid, track in self.tracks.items():
            x1, y1, x2, y2 = track["bbox"]
            active_tracks.append((x1, y1, x2, y2, tid))
                
        return active_tracks
