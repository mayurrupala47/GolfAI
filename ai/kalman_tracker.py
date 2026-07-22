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
    def __init__(self, config: Dict[str, Any], fps: int = 60, frame_width: int = 3840, low_fps: bool = False, resize_width: int = 640):
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
                base_res = cal.get("source_resolution", [1920, 1080])
                scale_x = resize_width / base_res[0]
                scale_y = resize_width / base_res[0]  # maintain aspect ratio
                for region in cal.get("ignore_regions", []):
                    if "tee" in region.get("name", "").lower():
                        rx = region["x"]
                        ry = region["y"]
                        self.tee_point = (rx, ry)  # raw calibration coords
                        # Scale to processing resolution
                        scaled_tx = rx * scale_x
                        scaled_ty = ry * scale_y
                        logger.info(f"Loaded Tee point (reference only): {self.tee_point}")
                        
                        # Pre-register Ball ID 1 at the scaled tee position.
                        # This ensures tracking starts at the tee from frame 0 so that:
                        # 1. Color is identified when ball is placed on tee
                        # 2. First stroke is counted when ball moves away from tee
                        kf = KalmanFilter(self.dt)
                        kf.initialize(scaled_tx, scaled_ty)
                        r = 12.0  # approximate ball radius in pixels at processing resolution
                        self.tracks[1] = {
                            "kf": kf,
                            "bbox": (scaled_tx - r, scaled_ty - r, scaled_tx + r, scaled_ty + r),
                            "disappeared": 0,
                            "last_matched_center": (scaled_tx, scaled_ty),
                            "color": "unknown",
                            "color_votes": {},
                            "resting_anchor": (scaled_tx, scaled_ty),
                            "frames_on_anchor": 0,
                            "tee_preregistered": True,  # flag so state machine knows this is virtual
                        }
                        self.next_track_id = 2
                        logger.info(f"[Tracker] Pre-registered Ball ID 1 at scaled tee ({scaled_tx:.1f}, {scaled_ty:.1f}). "
                                    f"Watching for ball placement at tee.")
                        break
            except Exception as e:
                logger.error(f"Failed to load Tee point: {e}")
            
        logger.info(f"KalmanBallTracker initialized. Search Window: {self.search_window}px, Max Lost: {self.max_lost_frames}")

    def update(self, detections: List[Tuple[float, float, float, float, float, str]], frame: np.ndarray = None, track_states: Dict[int, Any] = None) -> List[Tuple[float, float, float, float, int, str]]:
        # Compute predicted position for all current tracks
        predictions = {}
        for tid, track in self.tracks.items():
            pred_x, pred_y = track["kf"].predict()
            predictions[tid] = (pred_x, pred_y)
            
        # Parse new detections into centroids
        input_dets = []
        for det in detections:
            x1, y1, x2, y2, conf, color = det
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            input_dets.append({"bbox": (x1, y1, x2, y2), "center": (cx, cy), "color": color})
            
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
            
            locked_color = self.tracks[tid].get("color", "unknown")
            for idx, det in enumerate(input_dets):
                if idx in used_detections:
                    continue
                    
                cx, cy = det["center"]
                dist = ((cx - px)**2 + (cy - py)**2)**0.5
                
                # A resting ball cannot move > 30px away from its locked resting anchor position!
                anchor = self.tracks[tid].get("resting_anchor")
                if state in ["STOPPED", "READY"] and anchor is not None:
                    dist_from_anchor = math.sqrt((cx - anchor[0])**2 + (cy - anchor[1])**2)
                    if dist_from_anchor > 30.0:
                        continue  # Discard shoes/shadows/noise creeping away from resting anchor
                        
                # A resting ball cannot physically jump > 35px in a single frame (regardless of color)
                if state in ["STOPPED", "READY"] and dist > 35.0:
                    continue
                    
                # Color match weighting: Detections matching the locked ball color are prioritized over putter heads/shadows
                det_color = det.get("color", "unknown")
                effective_dist = dist
                
                # A resting ball cannot associate with non-matching color objects (shoes/shadows) > 15px away
                if state in ["STOPPED", "READY"] and locked_color != "unknown" and det_color != locked_color and dist > 15.0:
                    continue
                    
                if locked_color != "unknown" and det_color != locked_color:
                    if disappeared > 0:
                        continue  # Discard! A lost ball cannot re-appear as a completely different color (shoe/turf)
                    effective_dist += 150.0  # Penalize non-matching objects (putter heads/shadows)
                    
                if effective_dist < best_dist:
                    best_dist = effective_dist
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
                
                # --- Handle tee pre-registration: first real YOLO detection ---
                was_preregistered = self.tracks[tid].pop("tee_preregistered", False)
                if was_preregistered:
                    det_cx, det_cy = det["center"]
                    dist_from_prereg = math.sqrt((det_cx - last_cx)**2 + (det_cy - last_cy)**2)
                    if dist_from_prereg <= 100.0:
                        # Ball placed at tee — reset motion history so position jump doesn't spike speed
                        self.tracks[tid]["first_real_tee_detection"] = True
                        logger.info(f"[Tracker] Ball {tid}: First real detection at tee ({det_cx:.1f}, {det_cy:.1f}) "
                                    f"({dist_from_prereg:.1f}px from pre-reg). Flagging for motion reset.")
                    else:
                        # Ball first detected far from tee — stroke already happened, count it
                        self.tracks[tid]["anchor_just_escaped"] = True
                        logger.info(f"[Tracker] Ball {tid}: First detection {dist_from_prereg:.1f}px from pre-reg tee "
                                    f"— ball already in motion. Flagging as anchor escape.")

                # Maintain running vote for ball color to prevent occlusion/cup anomalies
                if "color_votes" not in self.tracks[tid]:
                    self.tracks[tid]["color_votes"] = {}
                c = det["color"]
                if c != "unknown" and not self.tracks[tid].get("color_locked", False):
                    self.tracks[tid]["color_votes"][c] = self.tracks[tid]["color_votes"].get(c, 0) + 1
                    # Majority vote
                    best_color = max(self.tracks[tid]["color_votes"], key=self.tracks[tid]["color_votes"].get)
                    self.tracks[tid]["color"] = best_color
                    # Lock color once a single color reaches 5 confident votes
                    if self.tracks[tid]["color_votes"][best_color] >= 5:
                        self.tracks[tid]["color_locked"] = True
                        logger.info(f"[Tracker] Ball {tid} color LOCKED as '{best_color.upper()}' after 5 consistent votes.")
                else:
                    self.tracks[tid]["color"] = self.tracks[tid].get("color", "unknown")
                self.tracks[tid]["last_matched_center"] = det["center"]
                self.tracks[tid]["frames_on_anchor"] = 0  # Reset: got a real detection
                # Lock resting anchor ONCE when stationary so it does not drift with transient noise/shoes
                if state in ["STOPPED", "READY"]:
                    if "resting_anchor" not in self.tracks[tid]:
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
                # --- TWO-ZONE ANCHOR LOGIC ---
                # Zone 1: detection within 100px of anchor → ball placed/resting near tee
                #         Update anchor to real position + accumulate color votes. No stroke.
                # Zone 2: detection beyond 100px of anchor → ball was HIT
                #         Clear anchor, signal stroke engine to count stroke.
                best_near_det = None   # closest detection within 100px (ball at or near anchor)
                best_near_dist = 100.0
                best_far_det = None    # furthest detection beyond 100px (ball hit)
                best_far_dist = 100.0

                for idx, det in enumerate(input_dets):
                    if idx in used_detections:
                        continue
                    ecx, ecy = det["center"]
                    dist_from_anchor = math.sqrt((ecx - anchor[0])**2 + (ecy - anchor[1])**2)
                    det_color = det.get("color", "unknown")
                    
                    # Color check: if color is locked AND candidate has a known color,
                    # it MUST match the locked color to be used for Zone 1 (resting update).
                    # Zone 2 (ball escaped far away) is exempt - it's a new positional hit, not a resting match.
                    if dist_from_anchor <= 100.0:
                        if locked_color != "unknown" and det_color != "unknown" and det_color != locked_color:
                            continue  # Reject non-matching color within anchor zone (shoe/putter/turf)

                    if dist_from_anchor <= 100.0 and dist_from_anchor < best_near_dist:
                        best_near_dist = dist_from_anchor
                        best_near_det = (idx, det)
                    elif dist_from_anchor > 100.0 and dist_from_anchor > best_far_dist:
                        best_far_dist = dist_from_anchor
                        best_far_det = (idx, det)

                # Zone 1: ball near anchor — update anchor to real position, accumulate color
                if best_near_det is not None:
                    idx, det = best_near_det
                    ecx, ecy = det["center"]
                    was_preregistered = self.tracks[tid].pop("tee_preregistered", False)
                    if was_preregistered:
                        # First real detection at tee — signal stroke_engine to reset motion analyzer
                        # so the jump from pre-registered position doesn't create a false speed spike.
                        self.tracks[tid]["first_real_tee_detection"] = True
                        logger.info(f"[Tracker] Ball {tid}: First real detection at tee ({ecx:.1f}, {ecy:.1f}). "
                                    f"Replacing pre-registration anchor.")
                    # Update anchor to the actual detected position
                    self.tracks[tid]["resting_anchor"] = (ecx, ecy)
                    self.tracks[tid]["kf"].update(ecx, ecy)
                    self.tracks[tid]["bbox"] = det["bbox"]
                    self.tracks[tid]["disappeared"] = 0
                    self.tracks[tid]["frames_on_anchor"] = 0
                    self.tracks[tid]["last_matched_center"] = (ecx, ecy)
                    # Accumulate color votes
                    if "color_votes" not in self.tracks[tid]:
                        self.tracks[tid]["color_votes"] = {}
                    c = det["color"]
                    if c != "unknown" and not self.tracks[tid].get("color_locked", False):
                        self.tracks[tid]["color_votes"][c] = self.tracks[tid]["color_votes"].get(c, 0) + 1
                        best_color = max(self.tracks[tid]["color_votes"], key=self.tracks[tid]["color_votes"].get)
                        self.tracks[tid]["color"] = best_color
                        if self.tracks[tid]["color_votes"][best_color] >= 5:
                            self.tracks[tid]["color_locked"] = True
                            logger.info(f"[Tracker] Ball {tid} color LOCKED as '{best_color.upper()}' "
                                        f"after 5 consistent votes at tee ({ecx:.1f}, {ecy:.1f}).")
                    used_detections.add(idx)
                    matched_tids.add(tid)
                    continue

                # Zone 2: ball escaped anchor — it was HIT. Count stroke.
                if best_far_det is not None and state in ["STOPPED", "READY"]:
                    idx, det = best_far_det
                    ecx, ecy = det["center"]
                    logger.info(f"[Tracker] Ball {tid} ESCAPED anchor at {anchor} -> ({ecx:.1f}, {ecy:.1f}) "
                                f"({best_far_dist:.1f}px). Stroke from tee detected!")
                    self.tracks[tid].pop("resting_anchor", None)
                    self.tracks[tid]["frames_on_anchor"] = 0
                    self.tracks[tid]["anchor_just_escaped"] = True  # Signal stroke engine to count stroke
                    # Initialize Kalman velocity towards new position
                    dx = ecx - float(self.tracks[tid]["kf"].state[0])
                    dy = ecy - float(self.tracks[tid]["kf"].state[1])
                    self.tracks[tid]["kf"].state[2] = dx
                    self.tracks[tid]["kf"].state[3] = dy
                    self.tracks[tid]["kf"].update(ecx, ecy)
                    self.tracks[tid]["bbox"] = det["bbox"]
                    self.tracks[tid]["disappeared"] = 0
                    # Update color vote
                    if "color_votes" not in self.tracks[tid]:
                        self.tracks[tid]["color_votes"] = {}
                    c = det["color"]
                    if c != "unknown" and not self.tracks[tid].get("color_locked", False):
                        self.tracks[tid]["color_votes"][c] = self.tracks[tid]["color_votes"].get(c, 0) + 1
                        self.tracks[tid]["color"] = max(self.tracks[tid]["color_votes"], key=self.tracks[tid]["color_votes"].get)
                    self.tracks[tid]["last_matched_center"] = (ecx, ecy)
                    used_detections.add(idx)
                    matched_tids.add(tid)
                    continue

                frames_on_anchor = self.tracks[tid].get("frames_on_anchor", 0) + 1
                self.tracks[tid]["frames_on_anchor"] = frames_on_anchor
                
                # If anchored for > 3 seconds without real detection, drop the track
                # EXCEPT if it's the pre-registered ball that hasn't seen a real detection yet
                is_preregistered = self.tracks[tid].get("tee_preregistered", False)
                anchor_timeout = int(self.fps * 3)  # 3 seconds
                if frames_on_anchor > anchor_timeout and not is_preregistered:
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
                        "last_matched_center": (cx, cy),
                        "color": det["color"],
                        "color_votes": {det["color"]: 1} if det["color"] != "unknown" else {}
                    }
                    logger.info(f"Registered new {det['color'].capitalize()} Ball ID {tid} at detected position ({cx:.1f}, {cy:.1f})")
                    self.next_track_id += 1
                    break  # Only register one track at a time
                
        # Format active tracks
        active_tracks = []
        for tid, track in self.tracks.items():
            x1, y1, x2, y2 = track["bbox"]
            color = track.get("color", "unknown")
            active_tracks.append((x1, y1, x2, y2, tid, color))
                
        return active_tracks
