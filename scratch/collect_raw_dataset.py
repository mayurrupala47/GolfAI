import cv2
import time
import os
import argparse
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("collect_dataset")

def main():
    parser = argparse.ArgumentParser(description="Capture raw camera frames to build a custom YOLO dataset.")
    parser.add_argument(
        "--video",
        type=str,
        default="rtsp://10.30.7.125:8554/mystream",
        help="Video source (RTSP URL, video file, or camera index)."
    )
    parser.add_argument(
        "--rpicam",
        action="store_true",
        help="Use Raspberry Pi hardware camera (rpicam-vid subprocess + GStreamer)."
    )
    parser.add_argument(
        "--gst-port",
        type=int,
        default=5000,
        help="UDP/TCP port to listen on for Pi camera stream. Default: 5000."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Time interval in seconds between frame captures. Default: 2.0."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/raw_images",
        help="Directory to save raw JPG images. Default: dataset/raw_images."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rpicam_process = None

    # Handle Raspberry Pi Camera source
    if args.rpicam:
        import subprocess
        logger.info("[RPICAM] Starting hardware camera subprocess...")
        cmd = [
            "rpicam-vid", "-t", "0", 
            "--mode", "4608:2592:10", 
            "--width", "1280", "--height", "720",  # Capture at higher resolution for better training datasets!
            "--framerate", "30", "--inline", 
            "--codec", "h264", "--profile", "baseline",
            "--listen", "-o", f"tcp://127.0.0.1:{args.gst_port}"
        ]
        rpicam_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pipeline = f"tcpclientsrc host=127.0.0.1 port={args.gst_port} ! h264parse ! v4l2h264dec ! videoconvert ! appsink drop=true max-buffers=1 sync=false emit-signals=false"
        time.sleep(1.5)
        logger.info("[RPICAM] Connecting to hardware stream via GStreamer...")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        # Standard RTSP, USB, or File stream
        source = args.video
        is_rtsp = source.startswith("rtsp://") or source.startswith("rtsps://")
        
        logger.info(f"Connecting to video source: {source} ...")
        if is_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        elif source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error("ERROR: Cannot open video source!")
        if rpicam_process:
            rpicam_process.terminate()
        return

    logger.info(f"Successfully connected to stream.")
    logger.info(f"Saving raw frames to '{args.output_dir}' every {args.interval}s.")
    logger.info("Press Ctrl+C to stop recording.")

    try:
        last_capture_time = 0
        while cap.isOpened():
            # For RTSP/Pi camera streams, we must continually grab frames 
            # to empty the buffer, preventing old/cached frames from piling up!
            if args.rpicam or is_rtsp:
                cap.grab()
                
            current_time = time.monotonic()
            if current_time - last_capture_time >= args.interval:
                # Retrieve and decode the frame
                ret, frame = cap.read() if not (args.rpicam or is_rtsp) else cap.retrieve()
                
                if not ret or frame is None:
                    logger.warning("Failed to retrieve frame from stream.")
                    time.sleep(0.1)
                    continue

                timestamp = int(time.time() * 1000)
                img_name = f"frame_{timestamp}.jpg"
                img_path = os.path.join(args.output_dir, img_name)
                
                cv2.imwrite(img_path, frame)
                logger.info(f"Captured: {img_name}")
                last_capture_time = current_time
            
            # Short sleep to prevent CPU pegging in the tight grab loop
            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("\nCapture stopped by user.")
    finally:
        cap.release()
        if rpicam_process:
            logger.info("Stopping Pi camera subprocess...")
            rpicam_process.terminate()
            rpicam_process.wait()
        logger.info("Collector cleaned up. Happy labeling!")

if __name__ == "__main__":
    main()
