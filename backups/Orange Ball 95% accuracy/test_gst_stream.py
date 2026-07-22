"""
GStreamer RTP Stream Tester for Raspberry Pi 3
===============================================
Run this script on the RPi BEFORE starting main.py with --gst,
to verify the GStreamer pipeline can receive frames from your IP camera.

Usage:
    python test_gst_stream.py [--port 5000] [--sw]

Options:
    --port PORT   UDP port to listen on (default: 5000)
    --sw          Force software decode (avdec_h264) even if v4l2h264dec is available

What to configure on your IP camera:
    Set the camera to push H264 RTP unicast to:
      Destination IP  : <this RPi's IP address>
      Destination port: 5000 (or whatever --port you choose)
      Codec           : H264
      Payload type    : 96 (standard dynamic payload for H264)

On camera web UIs this is usually called:
    "Unicast Push" / "RTP Stream" / "H264 over UDP"
"""
import cv2
import time
import argparse
import subprocess
import sys


def check_gst_element(name: str) -> bool:
    """Returns True if a GStreamer element/plugin is available on this system."""
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", name],
            capture_output=True, timeout=3
        )
        return result.returncode == 0
    except Exception:
        return False


def build_pipeline(port: int, force_sw: bool) -> str:
    """Builds the appropriate GStreamer pipeline string."""
    if not force_sw and check_gst_element("v4l2h264dec"):
        decoder = "v4l2h264dec"
        print(f"[OK] v4l2h264dec (hardware VideoCore decoder) found — using HW decode")
    else:
        decoder = "avdec_h264 max-threads=2"
        if force_sw:
            print("[INFO] Forced software decode (avdec_h264)")
        else:
            print("[WARN] v4l2h264dec not found — using software decode (avdec_h264). Install gstreamer1.0-plugins-bad for HW decode.")

    pipeline = (
        f"udpsrc port={port} "
        f"caps=\"application/x-rtp,encoding-name=H264,payload=96\" ! "
        f"rtph264depay ! h264parse ! "
        f"{decoder} ! "
        f"videoconvert ! "
        f"appsink drop=true max-buffers=1 sync=false emit-signals=false"
    )
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="GStreamer RTP stream tester for RPi3")
    parser.add_argument("--port", type=int, default=5000, help="UDP port to receive RTP on (default: 5000)")
    parser.add_argument("--sw", action="store_true", help="Force software H264 decode")
    parser.add_argument("--duration", type=int, default=15, help="Test duration in seconds (default: 15)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  GStreamer RTP Stream Test")
    print("="*60)
    print(f"\nListening for H264 RTP on UDP port {args.port}")
    print(f"Configure your IP camera to push to: <this RPi's IP>:{args.port}")
    print("\nChecking required GStreamer plugins...")

    # Check required plugins
    required = {
        "udpsrc": "gstreamer1.0-plugins-good",
        "rtph264depay": "gstreamer1.0-plugins-good",
        "h264parse": "gstreamer1.0-plugins-bad",
        "videoconvert": "gstreamer1.0-plugins-base",
    }
    all_ok = True
    for element, package in required.items():
        found = check_gst_element(element)
        status = "[OK]" if found else "[MISSING]"
        print(f"  {status} {element} (from {package})")
        if not found:
            all_ok = False

    if not all_ok:
        print("\n[ERROR] Missing GStreamer plugins. Install them with:")
        print("  sudo apt-get install gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly")
        print("  sudo apt-get install gstreamer1.0-libav  # for avdec_h264 software decode")
        sys.exit(1)

    pipeline_str = build_pipeline(args.port, args.sw)
    print(f"\nPipeline:\n  {pipeline_str}\n")
    print(f"Opening pipeline (waiting up to 10s for first frame)...\n")

    cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[ERROR] Failed to open GStreamer pipeline.")
        print("  - Make sure OpenCV was compiled with GStreamer support")
        print("  - Check: python3 -c \"import cv2; print(cv2.getBuildInformation())\" | grep GStreamer")
        sys.exit(1)

    print("[OK] Pipeline opened successfully!\n")

    start = time.monotonic()
    frame_count = 0
    first_frame_time = None

    print(f"Receiving frames for {args.duration} seconds. Press Ctrl+C to stop early.\n")

    try:
        while (time.monotonic() - start) < args.duration:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            if first_frame_time is None:
                first_frame_time = time.monotonic() - start
                h, w = frame.shape[:2]
                print(f"[OK] First frame received! {w}x{h} pixels | Latency from start: {first_frame_time:.2f}s")

            frame_count += 1

            if frame_count % 30 == 0:
                elapsed = time.monotonic() - start
                fps = frame_count / elapsed
                print(f"  {frame_count} frames | {fps:.1f} fps | elapsed: {elapsed:.1f}s")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    cap.release()

    elapsed = time.monotonic() - start
    avg_fps = frame_count / elapsed if elapsed > 0 else 0

    print("\n" + "="*60)
    print("  Test Results")
    print("="*60)
    print(f"  Frames received : {frame_count}")
    print(f"  Duration        : {elapsed:.1f}s")
    print(f"  Average FPS     : {avg_fps:.1f}")
    print(f"  First frame at  : {first_frame_time:.2f}s" if first_frame_time else "  First frame     : NONE (check camera config)")
    print("="*60)

    if frame_count == 0:
        print("\n[FAIL] No frames received.")
        print("  1. Verify your camera is configured to push H264 RTP to this IP")
        print(f"  2. Check firewall: sudo ufw allow {args.port}/udp")
        print("  3. Test with: gst-launch-1.0 udpsrc port={args.port} ! fakesink")
        sys.exit(1)
    else:
        print(f"\n[PASS] GStreamer pipeline working. Run main.py with:")
        print(f"  python main.py --gst --gst-port {args.port} --gst-width <W> --gst-height <H> --mock-mqtt")


if __name__ == "__main__":
    main()
