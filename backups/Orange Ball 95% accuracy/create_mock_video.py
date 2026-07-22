import cv2
import numpy as np
import json
import os
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock_generator")


def smoothstep(edge0, edge1, x):
    # Scale, bias and saturate x to 0..1 range
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    # Evaluate polynomial
    return x * x * (3.0 - 2.0 * x)


def clamp(val, minval, maxval):
    if val < minval:
        return minval
    if val > maxval:
        return maxval
    return val


def main():
    # Video properties
    width, height = 640, 480
    fps = 30
    duration_sec = 10
    total_frames = fps * duration_sec  # 300 frames
    
    output_video_dir = "assets"
    os.makedirs(output_video_dir, exist_ok=True)
    video_path = os.path.join(output_video_dir, "input.mp4")
    oracle_path = os.path.join(output_video_dir, "oracle_positions.json")

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        logger.error("Failed to open VideoWriter!")
        return

    # Ball properties
    radius = 8
    ball_color = (255, 255, 255)  # White BGR
    green_color = (34, 139, 34)   # Forest Green BGR

    # Ball position timeline
    # Frame 0 to 59: Stationary at (100, 100)
    # Frame 60 to 119: Stroke 1. Moves from (100, 100) to (400, 100)
    # Frame 120 to 179: Stationary at (400, 100)
    # Frame 180 to 239: Stroke 2. Moves from (400, 100) to (400, 350)
    # Frame 240 to 299: Stationary at (400, 350)
    
    oracle_positions = {}

    logger.info(f"Generating synthetic video: {video_path}...")
    for frame_idx in range(total_frames):
        # Create green field
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = green_color
        
        # Calculate current position (x, y)
        if frame_idx < 60:
            x, y = 100.0, 100.0
        elif frame_idx < 120:
            # Stroke 1: smoothstep interpolation
            t = (frame_idx - 60) / 60.0  # 0.0 to 1.0
            factor = smoothstep(0.0, 1.0, t)
            x = 100.0 + (400.0 - 100.0) * factor
            y = 100.0
        elif frame_idx < 180:
            x, y = 400.0, 100.0
        elif frame_idx < 240:
            # Stroke 2: smoothstep interpolation
            t = (frame_idx - 180) / 60.0  # 0.0 to 1.0
            factor = smoothstep(0.0, 1.0, t)
            x = 400.0
            y = 100.0 + (350.0 - 100.0) * factor
        else:
            x, y = 400.0, 350.0

        # Draw ball
        cv2.circle(frame, (int(x), int(y)), radius, ball_color, -1, cv2.LINE_AA)
        
        # Draw some reference grid lines (looks neat and mimics a real lane)
        # Border lines
        cv2.rectangle(frame, (30, 30), (width - 30, height - 30), (50, 80, 50), 2)
        # Tee area
        cv2.circle(frame, (100, 100), 20, (60, 100, 60), 1)
        # Target cup (hole)
        cv2.circle(frame, (400, 350), 12, (20, 20, 20), -1)

        # Write frame to video
        writer.write(frame)

        # Record bounding box [x1, y1, x2, y2]
        # In actual object detectors, we get the bounding box surrounding the object
        x1 = x - radius
        y1 = y - radius
        x2 = x + radius
        y2 = y + radius
        oracle_positions[str(frame_idx)] = [[x1, y1, x2, y2]]

    writer.release()
    logger.info("Mock video generation complete.")

    # Write oracle positions to file
    with open(oracle_path, "w") as f:
        json.dump(oracle_positions, f, indent=2)
    logger.info(f"Mock oracle locations written to {oracle_path}.")


if __name__ == "__main__":
    main()
