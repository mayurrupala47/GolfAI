import cv2
import time
import argparse
import os
import threading

def start_pi_camera_via_ssh(ip: str, port: int) -> bool:
    """
    Connects to the Raspberry Pi via SSH, kills any running rpicam-vid stream,
    and starts a new H264 stream in the background.
    """
    try:
        import paramiko
    except ImportError:
        print("[SSH Auto-Start] Installing paramiko SSH library...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko"])
            import paramiko
        except Exception as e:
            print(f"[SSH Auto-Start] Failed to install paramiko: {e}")
            return False

    try:
        print(f"[SSH Auto-Start] Connecting to Raspberry Pi at {ip} via SSH...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=ip, username="admin", password="admin", timeout=5.0)
        
        # Kill any active camera locks on the Pi
        ssh.exec_command("pkill -f rpicam-vid")
        time.sleep(0.5)
        
        # Run rpicam-vid in the background (nohup ensures it detaches and keeps running)
        cmd = (
            "nohup rpicam-vid -t 0 --mode 2304:1296:10 --width 960 --height 540 "
            "--framerate 30 --inline --codec h264 --profile baseline --listen "
            f"-o tcp://0.0.0.0:{port} > /dev/null 2>&1 &"
        )
        print(f"[SSH Auto-Start] Starting Pi camera stream: {cmd}")
        ssh.exec_command(cmd)
        
        time.sleep(3.0)
        ssh.close()
        print("[SSH Auto-Start] SSH command completed successfully. Pi is now streaming!")
        return True
    except Exception as e:
        print(f"[SSH Auto-Start] Failed to control Pi camera via SSH: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", type=str, default="tcp://10.30.7.125:5000", help="Camera URL (RTSP or TCP)")
    parser.add_argument("--output", type=str, default="recorded_multicolor.mp4", help="Output filename")
    parser.add_argument("--duration", type=int, default=60, help="Duration to record in seconds")
    args = parser.parse_args()

    # Remote start Pi camera if using tcp://
    if args.rtsp.startswith("tcp://"):
        try:
            parts = args.rtsp.replace("tcp://", "").split(":")
            pi_ip = parts[0]
            pi_port = int(parts[1]) if len(parts) > 1 else 5000
            start_pi_camera_via_ssh(pi_ip, pi_port)
        except Exception as e:
            print(f"[SSH Auto-Start] Failed to parse TCP URL {args.rtsp}: {e}")

    print(f"Connecting to Camera Stream: {args.rtsp}")
    cap = cv2.VideoCapture(args.rtsp)

    if not cap.isOpened():
        print("Failed to open stream. Retrying in 2 seconds...")
        time.sleep(2.0)
        cap = cv2.VideoCapture(args.rtsp)
        if not cap.isOpened():
            print("Could not open video stream.")
            return

    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 60:
        fps = 30.0

    print(f"Stream resolution: {width}x{height} @ {fps} FPS")
    print(f"Recording to {args.output} for {args.duration} seconds...")

    # Define Video Writer using MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    start_time = time.time()
    frames_recorded = 0

    try:
        while (time.time() - start_time) < args.duration:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            out.write(frame)
            frames_recorded += 1
            if frames_recorded % 30 == 0:
                elapsed = time.time() - start_time
                print(f"Recorded {frames_recorded} frames ({elapsed:.1f}s / {args.duration}s)...")
                
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    finally:
        cap.release()
        out.release()
        print(f"Finished! Saved {frames_recorded} frames to {args.output}")

if __name__ == "__main__":
    main()
