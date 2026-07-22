import cv2
import numpy as np

def classify_ball_color(crop_bgr: np.ndarray) -> str:
    """
    Classifies the color of the ball crop in BGR format into:
    'red', 'yellow', 'green', 'orange', or 'white'.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
        
    h_orig, w_orig = crop_bgr.shape[:2]
    # Crop to the center core (60% width and height) to minimize background turf
    margin_y = int(h_orig * 0.20)
    margin_x = int(w_orig * 0.20)
    core_crop = crop_bgr[margin_y:h_orig-margin_y, margin_x:w_orig-margin_x]
    if core_crop.size > 0:
        crop_bgr = core_crop
        
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Check for Orange pixels explicitly (Hue 5-28 with decent saturation/value)
    orange_pixels_mask = (h >= 5) & (h < 28) & (s > 70) & (v > 70)
    orange_fraction = np.sum(orange_pixels_mask) / h.size if h.size > 0 else 0.0
    if orange_fraction > 0.12:
        return "orange"
        
    # Check for Red pixels explicitly (Hue < 5 or > 165)
    red_pixels_mask = ((h < 5) | (h > 165)) & (s > 70) & (v > 70)
    if np.sum(red_pixels_mask) / h.size > 0.15:
        return "red"
        
    # Check for Yellow pixels explicitly (Hue 28-38)
    yellow_pixels_mask = (h >= 28) & (h < 38) & (s > 70) & (v > 70)
    if np.sum(yellow_pixels_mask) / h.size > 0.15:
        return "yellow"
        
    # Filter out green background turf pixels (Hue between 35 and 85)
    non_green_mask = (h < 35) | (h > 85)
    
    # Check if this is genuinely a green ball (fluorescent green ball with high saturation non-turf pixels)
    green_pixels_mask = (h >= 38) & (h < 85) & (s > 130) & (v > 130)
    green_fraction = np.sum(green_pixels_mask) / h.size if h.size > 0 else 0.0
    
    # If non-green pixels exist, prioritize non-green pixels (red/orange/yellow/white ball on green turf)
    if np.sum(non_green_mask) > 0.15 * h.size:
        valid_h = h[non_green_mask]
        valid_s = s[non_green_mask]
        valid_v = v[non_green_mask]
    else:
        # Genuine green ball on green turf
        if green_fraction > 0.40:
            return "green"
        valid_h, valid_s, valid_v = h.flatten(), s.flatten(), v.flatten()
        
    avg_s = np.mean(valid_s)
    avg_v = np.mean(valid_v)
    
    # Handle red wrap-around for averaging
    wrapped_h = np.where(valid_h > 90, valid_h - 180, valid_h)
    avg_h = np.mean(wrapped_h)
    if avg_h < 0:
        avg_h += 180
    
    # White classification: low saturation and high brightness
    if avg_s < 55 and avg_v > 120:
        return "white"
        
    # Classification based on Hue per AGENTS.md Rule 5:
    if avg_h < 8 or avg_h > 165:
        return "red"
    elif 8 <= avg_h < 25:
        return "orange"
    elif 25 <= avg_h < 38:
        return "yellow"
    elif 38 <= avg_h < 85:
        return "green"
        
    return "unknown"
