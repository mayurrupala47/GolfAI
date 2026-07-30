import cv2
import numpy as np

def classify_ball_color(crop_bgr: np.ndarray) -> str:
    """
    Classifies the color of the ball crop in BGR format into:
    'red', 'yellow', 'green', 'orange', or 'white' using dynamic background discounting.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
        
    ph, pw = crop_bgr.shape[:2]
    cy, cx = ph // 2, pw // 2
    
    # Calculate radius based on crop dimensions (reversing reference logic)
    radius = min(ph, pw) / 2.2
    
    yy, xx = np.ogrid[:ph, :pw]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

    # Extract ball pixels (center core)
    ball_mask = dist2 <= (radius * 0.75) ** 2
    ball_pixels = crop_bgr[ball_mask]
    
    # Extract background pixels (outer ring)
    bg_mask = (dist2 >= (radius * 1.3) ** 2) & (dist2 <= (radius * 2.0) ** 2)
    bg_pixels = crop_bgr[bg_mask]
    
    if ball_pixels.size < 8:
        return "unknown"
        
    def _to_hsv(px):
        s = px.reshape(-1, 1, 3).astype('uint8')
        h = cv2.cvtColor(s, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
        return h[:, 0], h[:, 1], h[:, 2]

    hue, sat, val = _to_hsv(ball_pixels)

    keep = val > 50
    if keep.sum() >= 8:
        hue, sat, val = hue[keep], sat[keep], val[keep]

    def _hue_dist(a, b):
        d = np.abs(a - b) % 180
        return np.minimum(d, 180 - d)

    bg_hue = None
    if bg_pixels.size >= 8:
        bhue, bsat, bval = _to_hsv(bg_pixels)
        bkeep = (bval > 50) & (bsat > 40)
        if bkeep.sum() >= 8:
            bg_hue = np.median(bhue[bkeep])

    colorful = sat > 60  # sat_cut
    if bg_hue is not None:
        # Discount any pixels that match the dynamic background turf hue
        matches_bg = colorful & (_hue_dist(hue, bg_hue) < 18)
        colorful = colorful & ~matches_bg

    colorful_frac = colorful.sum() / len(sat)

    if colorful_frac < 0.20:  # sat_fraction
        return "white"

    med_hue = np.median(hue[colorful])
    refs = {'orange': 15, 'yellow': 32, 'green': 60, 'pink': 155, 'red': 2}

    def hue_dist(a, b):
        d = abs(a - b) % 180
        return min(d, 180 - d)

    return min(refs, key=lambda k: hue_dist(med_hue, refs[k]))
