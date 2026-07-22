import numpy as np
import logging
import math
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
    def __init__(self, config: Dict[str, Any], fps: int = 60, frame_width: int = 3840, low_fps: bool = False):
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 60.0
        self.low_fps = low_fps
        
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
                        self.tee_point = (rx, ry)
                        logger.info(f"Loaded Tee point (reference only): {self.tee_point}")
                        break
            except Exception as e:
                logger.error(f"Failed to load Tee point: {e}")
                
        # Do NOT pre-register Ball ID 1 — wait for the first valid orange detection
        # This avoids locking onto noise near a potentially miscalibrated tee point
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
            pred_x, pred_y = pred
            
            # Determine dynamic search window based on track's Kalman velocity, disappeared state, and engine state
            kf = self.tracks[tid]["kf"]
            vx = float(kf.state[2])
            vy = float(kf.state[3])
            speed_px = (vx**2 + vy**2)**0.5
            
            # Clip prediction to last matched position if it drifts too far (>300px) during missing detections
            last_cx, last_cy = self.tracks[tid].get("last_matched_center", (pred_x, pred_y))
            if math.sqrt((pred_x - last_cx)**2 + (pred_y - last_cy)**2) > 300.0:
                pred_x, pred_y = last_cx, last_cy
            px, py = pred_x, pred_y
            
            disappeared = self.tracks[tid].get("disappeared", 0)
            
            # Get track state from engine
            state = None
            if track_states and tid in track_states:
                state = track_states[tid]
                
            if self.low_fps and disappeared > 0:
                # Low-FPS laptop testing mode: expand search window immediately on occlusion
                window = 1000.0
            elif state in ["STOPPED", "READY"]:
                # Safe narrow window to prevent static ball from latching to nearby noise
                window = 300.0
            elif disappeared > 0:
                # Expanded window to catch high-speed jumps and reappearing ball
                window = 1000.0
            else:
                # Standard window to track moving ball
                window = 350.0
            
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
                
                # If we were stationary (STOPPED/READY) and jumped > 40px, initialize physical motion vector
                if state in ["STOPPED", "READY"]:
                    dx = det["center"][0] - last_cx
                    dy = det["center"][1] - last_cy
                    jump_dist = (dx**2 + dy**2)**0.5
                    if jump_dist > 40.0:
                        self.tracks[tid]["stroke_angle"] = math.atan2(dy, dx)
                        self.tracks[tid]["stroke_speed_px"] = jump_dist
                        logger.info(f"[Ball {tid}] Captured physical stroke motion vector: angle={math.degrees(self.tracks[tid]['stroke_angle']):.1f}\u00b0, speed={jump_dist:.1f}px/frame")
                
                self.tracks[tid]["kf"].update(det["center"][0], det["center"][1])
                self.tracks[tid]["bbox"] = det["bbox"]
                self.tracks[tid]["disappeared"] = 0
                self.tracks[tid]["last_matched_center"] = det["center"]
                self.tracks[tid]["frames_on_anchor"] = 0  # Reset: got a real detection
                # Update resting anchor whenever the ball is confirmed at a stationary position
                if state in ["STOPPED", "READY"]:
                    self.tracks[tid]["resting_anchor"] = det["center"]
                else:
                    # Ball is moving — clear the anchor so it doesn't snap back
                    self.tracks[tid].pop("resting_anchor", None)
                used_detections.add(best_det_idx)
                matched_tids.add(tid)
                
        # Handle disappeared tracks
        for tid in list(self.tracks.keys()):
            if tid in matched_tids:
                continue
            
            # Get track state
            state = None
            if track_states and tid in track_states:
                state = track_states[tid]

            # --- Anchor injection for tracks with a resting anchor ---
            # If the track has a resting anchor, use it to freeze the position and velocity
            # during any frame where it is undetected, regardless of the state machine state.
            # This prevents Kalman filter drift and allows stable state machine transitions.
            # BUT: if stuck on anchor too long (3 seconds), drop the track for re-detection.
            anchor = self.tracks[tid].get("resting_anchor")
            if anchor:
                frames_on_anchor = self.tracks[tid].get("frames_on_anchor", 0) + 1
                self.tracks[tid]["frames_on_anchor"] = frames_on_anchor
                
                # If anchored for > 3 seconds without real detection, drop the track
                anchor_timeout = int(self.fps * 3)  # 3 seconds
                if frames_on_anchor > anchor_timeout:
                    logger.info(f"[Tracker] Ball {tid} stuck on anchor at {anchor} for {frames_on_anchor} frames "
                                f"(>{anchor_timeout}). Dropping track for re-detection.")
                    # Reuse the same Ball ID on re-registration to preserve identity
                    self.next_track_id = tid
                    del self.tracks[tid]
                    continue
                
                ax, ay = anchor
                # Zero velocity so the state machine sees speed = 0
                self.tracks[tid]["kf"].state[2] = 0.0
                self.tracks[tid]["kf"].state[3] = 0.0
                self.tracks[tid]["kf"].update(ax, ay)
                r = 9.0  # half of expected ball diameter
                self.tracks[tid]["bbox"] = (ax - r, ay - r, ax + r, ay + r)
                self.tracks[tid]["disappeared"] = 0
                matched_tids.add(tid)
                continue

            # --- Normal disappeared handling for MOVING / UNKNOWN tracks ---
            self.tracks[tid]["disappeared"] += 1
            
            # If the ball was moving and goes missing, assume it has stopped at its last known position
            # and transition it to a stationary anchor to prevent losing the track (e.g. behind obstacles).
            # Set to exactly 3.0 seconds dynamically using the stream's FPS.
            if state == "MOVING" and self.tracks[tid]["disappeared"] >= int(self.fps * 3.0):
                last_center = self.tracks[tid].get("last_matched_center")
                if last_center:
                    ax, ay = last_center
                    self.tracks[tid]["resting_anchor"] = last_center
                    self.tracks[tid]["disappeared"] = 0
                    # Zero out velocity and update Kalman Filter state
                    self.tracks[tid]["kf"].state[2] = 0.0
                    self.tracks[tid]["kf"].state[3] = 0.0
                    self.tracks[tid]["kf"].update(ax, ay)
                    r = 9.0
                    self.tracks[tid]["bbox"] = (ax - r, ay - r, ax + r, ay + r)
                    logger.info(f"[Tracker] Ball {tid} went missing while MOVING. Anchoring at last known position {last_center} and zeroing velocity.")
                    matched_tids.add(tid)
                    continue
                    
            if self.tracks[tid]["disappeared"] > self.max_lost_frames:
                self.next_track_id = tid  # Reuse Ball ID on re-detection
                del self.tracks[tid]
            else:
                # Decay physical motion vector speed if tracking is lost
                if "stroke_speed_px" in self.tracks[tid]:
                    self.tracks[tid]["stroke_speed_px"] *= 0.95
                    
                # Decay velocity components in Kalman filter state to model friction/stopping (30% decay per frame)
                self.tracks[tid]["kf"].state[2] *= 0.70
                self.tracks[tid]["kf"].state[3] *= 0.70
                
                # Get current position from state
                px = float(self.tracks[tid]["kf"].state[0])
                py = float(self.tracks[tid]["kf"].state[1])
                
                x1, y1, x2, y2 = self.tracks[tid]["bbox"]
                bw = x2 - x1
                bh = y2 - y1
                self.tracks[tid]["bbox"] = (px - bw/2.0, py - bh/2.0, px + bw/2.0, py + bh/2.0)
            
        # Register first valid detection as Ball ID 1 when no active tracks exist.
        # The strict color filter (H:3-22, S:120+) already ensures only the orange ball passes.
        if len(self.tracks) == 0:
            for idx, det in enumerate(input_dets):
                if idx not in used_detections:
                    cx, cy = det["center"]
                    
                    kf = KalmanFilter(self.dt)
                    kf.initialize(cx, cy)
                    
                    tid = self.next_track_id
                    self.tracks[tid] = {
                        "kf": kf,
                        "bbox": det["bbox"],
                        "disappeared": 0,
                        "last_matched_center": (cx, cy)
                    }
                    logger.info(f"Registered new Ball ID {tid} at detected position ({cx:.1f}, {cy:.1f})")
                    self.next_track_id += 1
                    break  # Only register one track at a time
                
        # Format active tracks
        active_tracks = []
        for tid, track in self.tracks.items():
            x1, y1, x2, y2 = track["bbox"]
            active_tracks.append((x1, y1, x2, y2, tid))
                
        return active_tracks
