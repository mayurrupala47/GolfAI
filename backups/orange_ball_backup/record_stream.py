import cv2
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", type=str, default="rtsp://10.30.7.125:8554/mystream", help="RTSP URL")
    parser.add_argument("--output", type=str, default="test_orange_ball.mp4", help="Output filename")
    parser.add_argument("--duration", type=int, default=20, help="Duration to record in seconds")
    args = parser.parse_args()

    print(f"Connecting to RTSP Stream: {args.rtsp}")
    cap = cv2.VideoCapture(args.rtsp)

    if not cap.isOpened():
        print("Failed to open stream.")
        return

    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    print(f"Stream resolution: {width}x{height} @ {fps} FPS")
    print(f"Recording to {args.output} for {args.duration} seconds...")

    # Define Video Writer using MP4V codec (highly compatible on Windows/Mac)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    start_time = time.time()
    frames_recorded = 0

    try:
        while (time.time() - start_time) < args.duration:
            ret, frame = cap.read()
            if not ret:
                print("Dropped frame, retrying...")
                time.sleep(0.01)
                continue
            
            out.write(frame)
            frames_recorded += 1
            if frames_recorded % 30 == 0:
                print(f"Recorded {frames_recorded} frames...")
                
    except KeyboardInterrupt:
        print("Recording interrupted by user.")
    finally:
        cap.release()
        out.release()
        print(f"Finished! Saved {frames_recorded} frames to {args.output}")

if __name__ == "__main__":
    main()
