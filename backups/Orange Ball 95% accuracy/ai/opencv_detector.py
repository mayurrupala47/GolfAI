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
    OpenCV-based ball detector with Appearance Lock-On and Vacancy Check.

    Key mechanisms:
    1. Captures ball fingerprint at tee (size, color, shape)
    2. Vacancy Check: before switching to a new object, verifies the ball's
       old position is actually EMPTY. Prevents hand-hijacking.
    3. Strict re-acquisition: after ball is lost, requires profile match +
       stability before accepting a new candidate.
    """
    def __init__(self, config: Dict[str, Any], calibration_path: str = "config/calibration.json"):
        classic_cfg = config.get("classic_detector", {})
        self.min_area = classic_cfg.get("min_area", 30.0)
        self.max_area = classic_cfg.get("max_area", 100.0)
        self.circularity_thresh = classic_cfg.get("circularity_thresh", 0.55)
        
        ball_cfg = config.get("ball", {})
        self.expected_diameter = ball_cfg.get("expected_diameter", 18.0)
        self.diameter_tolerance = ball_cfg.get("diameter_tolerance", 8.0)
        
        self.calibration_path = calibration_path
        self.calibration_data = {}
        self.mask = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        self.lower_orange = np.array([4, 60, 50], dtype=np.uint8)
        self.upper_orange = np.array([25, 255, 255], dtype=np.uint8)
        
        proc_cfg = config.get("processing", {})
        self.roi_radius = proc_cfg.get("roi_radius", 120)
        self.roi_moving_radius = proc_cfg.get("roi_moving_radius", 220)
        
        # --- Ball Appearance Profile ---
        self._ball_profile = None
        
        # --- Lock-on state ---
        self._own_center = None
        self._pending_center = None
        self._pending_count = 0
        self._pending_candidate = None
        self._LOCK_FRAMES = 5
        self._SWITCH_DIST = 20.0
        self._VACANCY_RADIUS = 35.0  # Radius to check if old position is vacant
        self._lost_frames = 0        # How many frames since we last saw the ball
        self._MAX_LOST = 90          # After 90 lost frames (~3 sec), reset entirely
        
        # Load calibration
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
        base_res = self.calibration_data.get("source_resolution", [3840, 2160])
        scale_x = w / base_res[0]
        scale_y = h / base_res[1]
        
        playable_area = self.calibration_data.get("playable_area", [])
        if len(playable_area) >= 3:
            pts = np.array([[int(p[0] * scale_x), int(p[1] * scale_y)] for p in playable_area], dtype=np.int32)
            pa_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(pa_mask, [pts], 255)
            self.mask = cv2.bitwise_and(self.mask, pa_mask)
            
        ignore_regions = self.calibration_data.get("ignore_regions", [])
        for region in ignore_regions:
            if "tee" in region.get("name", "").lower():
                continue
            if region.get("type") == "polygon" or "points" in region:
                poly_pts = region["points"]
                scaled_pts = np.array(
                    [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in poly_pts],
                    dtype=np.int32
                )
                cv2.fillPoly(self.mask, [scaled_pts], 0)
                logger.info(f"Applied polygon ignore region mask: {region.get('name')}")
            else:
                rx = int(region["x"] * scale_x)
                ry = int(region["y"] * scale_y)
                rr = int(region["radius"] * min(scale_x, scale_y))
                cv2.circle(self.mask, (rx, ry), rr, 0, -1)
                logger.info(f"Applied circle ignore region mask: {region.get('name')}")
            
        logger.info(f"Initialized scaled mask with size {w}x{h}")

    def _capture_profile(self, candidate: dict, hsv_frame: np.ndarray) -> dict:
        """Capture the ball's appearance fingerprint from a confirmed candidate."""
        x1, y1, x2, y2 = candidate["box"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        roi_hsv = hsv_frame[max(0,y1):max(1,y2), max(0,x1):max(1,x2)]
        roi_mask = cv2.inRange(roi_hsv, self.lower_orange, self.upper_orange)
        ball_pixels = roi_hsv[roi_mask > 0]
        
        if len(ball_pixels) > 0:
            mean_hsv = np.mean(ball_pixels, axis=0)
        else:
            mean_hsv = np.array([14.0, 180.0, 180.0])
            
        # Strict validation checks during profile capture to reject hands and bracelets
        # 1. Saturation check: Skin is low saturation (20-60). Golf ball is high (85+).
        # 2. Solidity check: Hands/bracelets have low solidity. Golf ball is highly solid (>0.82).
        if mean_hsv[1] < 85.0 or candidate.get("solidity", 1.0) < 0.82:
            logger.warning(
                f"[DETECTOR] Rejecting profile capture: Candidate not ball-like. "
                f"Sat={mean_hsv[1]:.1f} (min 85), Solidity={candidate.get('solidity', 1.0):.2f} (min 0.82)"
            )
            return None
        
        profile = {
            "area": candidate["area"],
            "mean_h": float(mean_hsv[0]),
            "mean_s": float(mean_hsv[1]),
            "mean_v": float(mean_hsv[2]),
            "width": int(x2 - x1),
            "height": int(y2 - y1),
            "circularity": candidate["circularity"],
        }
        logger.info(
            f"[BALL PROFILE CAPTURED] Area={profile['area']:.0f}px, "
            f"HSV=({profile['mean_h']:.1f}, {profile['mean_s']:.1f}, {profile['mean_v']:.1f}), "
            f"Size={profile['width']}x{profile['height']}px"
        )
        return profile

    def _profile_match_score(self, candidate: dict, hsv_frame: np.ndarray) -> float:
        """
        Score how closely a candidate matches the captured ball profile.
        Returns 0.0 (no match) to 1.0 (perfect match).
        Wide tolerances for motion blur and lighting changes.
        """
        if self._ball_profile is None:
            return 0.5
        
        bp = self._ball_profile
        
        # 1. Area: log-ratio (tolerant of 4x motion blur stretch)
        if candidate["area"] > 0 and bp["area"] > 0:
            log_ratio = abs(np.log(candidate["area"] / bp["area"]))
            area_score = max(0.0, 1.0 - (log_ratio / 1.6))
        else:
            area_score = 0.0
        
        # 2. Color: tolerant of lighting shifts
        x1, y1, x2, y2 = candidate["box"]
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        roi_hsv = hsv_frame[max(0,y1i):max(1,y2i), max(0,x1i):max(1,x2i)]
        roi_mask = cv2.inRange(roi_hsv, self.lower_orange, self.upper_orange)
        pixels = roi_hsv[roi_mask > 0]
        
        if len(pixels) > 0:
            mean_h = float(np.mean(pixels[:, 0]))
            mean_s = float(np.mean(pixels[:, 1]))
        else:
            mean_h, mean_s = bp["mean_h"], bp["mean_s"]
        
        h_score = max(0.0, 1.0 - (abs(mean_h - bp["mean_h"]) / 15.0))
        s_score = max(0.0, 1.0 - (abs(mean_s - bp["mean_s"]) / 120.0))
        color_score = h_score * 0.6 + s_score * 0.4
        
        # 3. Min-dimension (robust to motion blur stretch)
        cw = max(1, int(x2) - int(x1))
        ch = max(1, int(y2) - int(y1))
        ball_min = min(bp["width"], bp["height"])
        cand_min = min(cw, ch)
        if ball_min > 0 and cand_min > 0:
            dim_ratio = min(cand_min, ball_min) / max(cand_min, ball_min)
        else:
            dim_ratio = 0.0
        
        return 0.50 * area_score + 0.25 * color_score + 0.25 * dim_ratio

    def _adapt_profile(self, candidate: dict, hsv_frame: np.ndarray):
        """Slowly adapt the ball profile to current lighting conditions."""
        if self._ball_profile is None:
            return
        alpha = 0.05  # 5% new, 95% old
        x1, y1, x2, y2 = candidate["box"]
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        roi_hsv = hsv_frame[max(0,y1i):max(1,y2i), max(0,x1i):max(1,x2i)]
        roi_mask = cv2.inRange(roi_hsv, self.lower_orange, self.upper_orange)
        pix = roi_hsv[roi_mask > 0]
        if len(pix) > 0:
            self._ball_profile["mean_h"] = (1-alpha) * self._ball_profile["mean_h"] + alpha * float(np.mean(pix[:, 0]))
            self._ball_profile["mean_s"] = (1-alpha) * self._ball_profile["mean_s"] + alpha * float(np.mean(pix[:, 1]))
            self._ball_profile["mean_v"] = (1-alpha) * self._ball_profile["mean_v"] + alpha * float(np.mean(pix[:, 2]))
        # Only adapt area when ball is NOT blurred
        area_ratio = candidate["area"] / self._ball_profile["area"] if self._ball_profile["area"] > 0 else 1.0
        if 0.7 <= area_ratio <= 1.5:
            self._ball_profile["area"] = (1-alpha) * self._ball_profile["area"] + alpha * candidate["area"]

    def _find_candidates(self, frame: np.ndarray, hsv: np.ndarray):
        """Extract all valid orange contour candidates from the frame."""
        color_mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        color_mask = cv2.bitwise_and(color_mask, color_mask, mask=self.mask)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, self.kernel)
        
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5.0 <= area <= 500.0:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh
                if 0.4 <= aspect_ratio <= 2.5:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * (area / (perimeter * perimeter))
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        solidity = (float(area) / hull_area) if hull_area > 0 else 0
                        if circularity > 0.3 and solidity > 0.7:
                            candidates.append({
                                "box": (float(x), float(y), float(x + bw), float(y + bh)),
                                "area": area,
                                "circularity": circularity,
                                "solidity": solidity,
                                "cx": float(x + bw / 2.0),
                                "cy": float(y + bh / 2.0),
                            })
        return candidates

    def _dist(self, c1, c2):
        """Euclidean distance between two (cx, cy) tuples."""
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def _candidate_near(self, candidates, center, radius):
        """Find the best candidate within radius of a center point."""
        near = [(c, self._dist((c["cx"], c["cy"]), center)) for c in candidates]
        near = [(c, d) for c, d in near if d <= radius]
        if near:
            near.sort(key=lambda x: x[1])
            return near[0][0]
        return None

    def detect(
        self,
        frame: np.ndarray,
        hint_center: tuple = None,
        hint_moving: bool = False
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Detects the golf ball with vacancy-check anti-hijack logic.
        
        Core rule: NEVER switch to a new object unless the old position is VACANT.
        This prevents the hand from hijacking the tracker while the ball is still visible.
        """
        h, w, _ = frame.shape
        if self.mask is None:
            self._initialize_mask(h, w)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        candidates = self._find_candidates(frame, hsv)

        if not candidates:
            if self._own_center is not None:
                self._lost_frames += 1
                if self._lost_frames > self._MAX_LOST:
                    # Ball has been gone too long — full reset
                    logger.info("[DETECTOR] Ball lost for too long. Resetting lock-on.")
                    self._own_center = None
                    self._ball_profile = None
                    self._pending_center = None
                    self._pending_count = 0
                    self._lost_frames = 0
            return []

        # ===== PHASE 1: No confirmed ball yet (startup / after full reset) =====
        if self._own_center is None:
            # Pick the candidate that is most "ball-like": smallest, roundest, lowest
            best = max(candidates, key=lambda c: (
                0.40 * min(c["circularity"], 1.0) +
                0.35 * (c["cy"] / float(h)) +
                0.25 * c["solidity"]
            ))
            best_center = (best["cx"], best["cy"])
            
            # Require stability before accepting
            if self._pending_center is not None:
                if self._dist(best_center, self._pending_center) <= self._SWITCH_DIST:
                    self._pending_count += 1
                    self._pending_candidate = best
                else:
                    self._pending_center = best_center
                    self._pending_count = 1
                    self._pending_candidate = best
            else:
                self._pending_center = best_center
                self._pending_count = 1
                self._pending_candidate = best
            
            if self._pending_count >= self._LOCK_FRAMES:
                # Stable for 5 frames → confirmed as ball! Capture profile.
                profile = self._capture_profile(self._pending_candidate, hsv)
                if profile is not None:
                    self._own_center = self._pending_center
                    self._ball_profile = profile
                    self._pending_center = None
                    self._pending_count = 0
                    self._pending_candidate = None
                    self._lost_frames = 0
                    b = best["box"]
                    return [(b[0], b[1], b[2], b[3], 0.99)]
                else:
                    # Failed validation (hand/bracelet rejected). Reset pending state.
                    self._pending_center = None
                    self._pending_count = 0
                    self._pending_candidate = None
                    return []
            return []

        # ===== PHASE 2: Ball is confirmed — track with vacancy check =====
        
        # Step 1: Is there still a candidate at/near the confirmed position?
        ball_at_old_pos = self._candidate_near(candidates, self._own_center, self._VACANCY_RADIUS)
        
        if ball_at_old_pos is not None:
            # Ball is still at its old position! Track it. Ignore everything else.
            self._own_center = (ball_at_old_pos["cx"], ball_at_old_pos["cy"])
            self._lost_frames = 0
            self._pending_center = None
            self._pending_count = 0
            self._adapt_profile(ball_at_old_pos, hsv)
            b = ball_at_old_pos["box"]
            return [(b[0], b[1], b[2], b[3], 0.99)]
        
        # Step 2: Old position is VACANT — the ball has genuinely moved!
        # Now find the best candidate that matches the ball profile.
        # This is where we re-acquire the ball at its new resting position.
        self._lost_frames += 1
        
        if self._lost_frames > self._MAX_LOST:
            # Too long — full reset
            logger.info("[DETECTOR] Ball lost for too long after vacancy. Resetting.")
            self._own_center = None
            self._ball_profile = None
            self._pending_center = None
            self._pending_count = 0
            self._lost_frames = 0
            return []
        
        # Score all candidates by profile match
        scored = []
        for c in candidates:
            ps = self._profile_match_score(c, hsv)
            scored.append((ps, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        
        best_ps, best_c = scored[0]
        best_center = (best_c["cx"], best_c["cy"])
        
        # Only consider candidates that reasonably match the ball profile
        if best_ps < 0.3:
            # Nothing looks like the ball — it's all hands/noise. Keep waiting.
            return []
        # Check if the candidate passes the strict audits (solidity > 0.82, saturation > 85)
        # If it does, we can re-acquire it instantly without waiting for stability (helps track fast rolling balls!)
        profile = self._capture_profile(best_c, hsv)
        if profile is not None:
            logger.info(
                f"[DETECTOR] Ball re-acquired instantly at ({best_center[0]:.0f}, {best_center[1]:.0f}) "
                f"after {self._lost_frames} lost frames (Audits Passed)."
            )
            self._own_center = best_center
            self._ball_profile = profile
            self._lost_frames = 0
            self._pending_center = None
            self._pending_count = 0
            b = best_c["box"]
            return [(b[0], b[1], b[2], b[3], best_ps)]
            
        # Fallback to stability check (only for objects that might not pass the strict audits but are stable)
        if self._pending_center is not None:
            if self._dist(best_center, self._pending_center) <= self._SWITCH_DIST:
                self._pending_count += 1
            else:
                self._pending_center = best_center
                self._pending_count = 1
        else:
            self._pending_center = best_center
            self._pending_count = 1
        
        if self._pending_count >= self._LOCK_FRAMES:
            # Check profile again during stable lock
            profile = self._capture_profile(best_c, hsv)
            if profile is not None:
                logger.info(
                    f"[DETECTOR] Ball re-acquired at ({best_center[0]:.0f}, {best_center[1]:.0f}) "
                    f"after {self._lost_frames} lost frames (Stable). Profile match: {best_ps:.2f}"
                )
                self._own_center = self._pending_center
                self._ball_profile = profile
                self._pending_center = None
                self._pending_count = 0
                self._lost_frames = 0
                b = best_c["box"]
                return [(b[0], b[1], b[2], b[3], best_ps)]
            else:
                self._pending_center = None
                self._pending_count = 0
                return []
        
        return []
