import os
import argparse
import yaml
import logging
import cv2
from typing import Dict, Any

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")

from ai.detector import YoloBallDetector, MockBallDetector, ClassicBallDetector, HybridYoloBallDetector, YoloOnlyBallDetector, CustomYoloOpenCVHybridDetector
from ai.opencv_detector import OpenCVBallDetector
from ai.tracker import ByteBallTracker
from ai.kalman_tracker import KalmanBallTracker
from ai.motion import MotionAnalyzer
from mqtt.publisher import MqttPublisher, MockMqttPublisher
from exporters.csv_export import CsvExporter
from exporters.json_export import JsonExporter
from engine.stroke_engine import StrokeEngine
from visualization.overlay import Visualizer
import threading
import time
from app import app, push_frame, push_stroke_event, push_stroke_frame

class RtspStream:
    """
    Background-thread RTSP reader using GStreamer (default backend on RPi3).
    Drains the camera buffer continuously so the main loop always gets the newest frame.
    Auto-reconnects if the IP camera drops the connection.
    """
    def __init__(self, src: str, rtsp_transport: str = "tcp"):
        self.src = src
        self.rtsp_transport = rtsp_transport
        self._lock = threading.Lock()
        self._frame = None
        self._grabbed = False
        self._stopped = False
        self._reconnect_delay = 2.0
        self._cap = None
        self._connect()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        time.sleep(0.5)

    def _connect(self) -> bool:
        global STATUS_MESSAGE
        STATUS_MESSAGE = f"Connecting to stream: {self.src}..."
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        cap = cv2.VideoCapture(self.src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            self._cap = cap
            logger.info(f"[RtspStream] Connected to {self.src}")
            STATUS_MESSAGE = "Live feed connected! Warming up..."
            return True
        logger.warning(f"[RtspStream] Could not connect to {self.src}")
        self._cap = cap
        return False

    def _reader_loop(self):
        while not self._stopped:
            # If the stream is not open, sleep to prevent busy-spinning the CPU with connect attempts
            if self._cap is None or not self._cap.isOpened():
                time.sleep(2.0)
                self._connect()
                continue
                
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._grabbed = True
            else:
                global STATUS_MESSAGE
                STATUS_MESSAGE = "Stream lost. Reconnecting..."
                logger.warning("[RtspStream] Frame read failed. Attempting reconnect...")
                self._connect()
                time.sleep(1.0)

    def read(self):
        with self._lock:
            if self._grabbed and self._frame is not None:
                self._grabbed = False  # Reset flag to allow next frame decoding
                return True, self._frame.copy()
            # If no new frame is decoded, return the last frame to prevent blocking
            return self._frame is not None, self._frame.copy() if self._frame is not None else None

    def isOpened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get(self, propId):
        return self._cap.get(propId) if self._cap else 0

    def release(self):
        self._stopped = True
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info("[RtspStream] Released.")

STATUS_MESSAGE = "Initializing Mini Golf AI..."


def start_pi_camera_via_ssh(ip: str, port: int) -> bool:
    """
    Connects to the Raspberry Pi via SSH, kills any running rpicam-vid stream,
    and starts a new H264 stream in the background.
    """
    global STATUS_MESSAGE
    try:
        import paramiko
    except ImportError:
        STATUS_MESSAGE = "Installing paramiko SSH library on laptop..."
        logger.warning("[SSH Auto-Start] 'paramiko' is not installed. Trying to install it automatically...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko"])
            import paramiko
            logger.info("[SSH Auto-Start] 'paramiko' installed successfully.")
        except Exception as e:
            STATUS_MESSAGE = "SSH Error: paramiko installation failed"
            logger.error(f"[SSH Auto-Start] Failed to install 'paramiko': {e}. Please run 'pip install paramiko' manually.")
            return False

    try:
        STATUS_MESSAGE = f"SSH Connecting to Pi at {ip}..."
        logger.info(f"[SSH Auto-Start] Connecting to Raspberry Pi at {ip} via SSH...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=ip, username="admin", password="admin", timeout=5.0)
        
        STATUS_MESSAGE = "SSH: Stopping existing camera process..."
        # Kill any active camera locks on the Pi
        ssh.exec_command("pkill -f rpicam-vid")
        time.sleep(0.5)
        
        STATUS_MESSAGE = "SSH: Starting H264 camera stream..."
        # Run rpicam-vid in the background (nohup ensures it detaches and keeps running)
        cmd = (
            f"nohup rpicam-vid -t 0 --mode 2304:1296:10 --width 960 --height 540 "
            f"--framerate 30 --inline --codec h264 --profile baseline --listen "
            f"-o tcp://0.0.0.0:{port} > /dev/null 2>&1 &"
        )
        logger.info(f"[SSH Auto-Start] Starting Pi camera stream: {cmd}")
        ssh.exec_command(cmd)
        
        STATUS_MESSAGE = "SSH: Waiting 3 seconds for camera warmup..."
        time.sleep(3.0)
        ssh.close()
        STATUS_MESSAGE = "SSH: Stream started successfully! Connecting..."
        logger.info("[SSH Auto-Start] SSH command completed successfully. Pi is now streaming!")
        return True
    except Exception as e:
        STATUS_MESSAGE = f"SSH Autostart bypassed: {e}"
        logger.warning(f"[SSH Auto-Start] Failed to control Pi camera via SSH: {e}. If the stream is already running, this is fine.")
        return False


class GStreamerStream:
    """
    Zero-latency direct RTP/UDP stream reader using GStreamer's hardware H264 decoder
    on Raspberry Pi 3.

    Why this is faster than RTSP:
    ─────────────────────────────
    RTSP path:  Camera → TCP session → RTSP server → RTSP client → software H264 decode → frame
    GST path:   Camera → UDP packet  → GStreamer   → v4l2h264dec (VideoCore GPU) → frame

    Eliminating RTSP removes:
      • TCP handshake & session negotiation overhead
      • RTSP control-plane round-trips
      • Software H264 decode (moves to RPi's GPU, freeing ~30% CPU)

    How to configure your camera:
    ─────────────────────────────
    On the IP camera side, configure it to push H264 RTP to UDP port 5000 on the RPi's IP.
    Most cameras call this "unicast push" or "RTP stream". No RTSP server needed.

    Pipeline used:
      udpsrc port=<PORT> → RTP H264 depay → H264 parse → v4l2h264dec (HW) → videoconvert → appsink
    """

    # GStreamer pipeline templates
    # v4l2h264dec = VideoCore IV hardware decoder on RPi3 (zero CPU cost)
    # appsink drop=true max-buffers=1 sync=false = always latest frame, never blocks
    _PIPELINE_HW = (
        "udpsrc port={port} "
        "caps=\"application/x-rtp,encoding-name=H264,payload=96\" ! "
        "rtph264depay ! h264parse ! "
        "v4l2h264dec ! "
        "videoconvert ! "
        "appsink drop=true max-buffers=1 sync=false emit-signals=false"
    )

    # Fallback: software decode (libav) if v4l2h264dec is unavailable
    _PIPELINE_SW = (
        "udpsrc port={port} "
        "caps=\"application/x-rtp,encoding-name=H264,payload=96\" ! "
        "rtph264depay ! h264parse ! "
        "avdec_h264 max-threads=2 ! "
        "videoconvert ! "
        "appsink drop=true max-buffers=1 sync=false emit-signals=false"
    )

    def __init__(self, port: int = 5000, custom_pipeline: str = None, width: int = 0, height: int = 0):
        """
        Args:
            port: UDP port the camera is pushing RTP packets to (default 5000).
            custom_pipeline: Full GStreamer pipeline string. Overrides the default template.
                             Must end with 'appsink ...' for OpenCV compatibility.
            width/height: Expected resolution (0 = auto-detect from stream).
        """
        self.port = port
        self._lock = threading.Lock()
        self._frame = None
        self._grabbed = False
        self._stopped = False
        self._cap = None
        self._width = width
        self._height = height

        # Build pipeline string
        if custom_pipeline:
            self._pipeline_str = custom_pipeline
            logger.info(f"[GStreamerStream] Using custom pipeline: {custom_pipeline}")
        else:
            self._pipeline_str = self._build_pipeline(port)

        self._connect()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        time.sleep(1.0)  # Give GStreamer a moment to negotiate and buffer first frame

    def _build_pipeline(self, port: int) -> str:
        """Tries hardware decode first; falls back to software if HW not available."""
        # Quick probe: does v4l2h264dec exist on this system?
        import subprocess
        try:
            result = subprocess.run(
                ["gst-inspect-1.0", "v4l2h264dec"],
                capture_output=True, timeout=3
            )
            if result.returncode == 0:
                pipeline = self._PIPELINE_HW.format(port=port)
                logger.info("[GStreamerStream] v4l2h264dec (hardware) decoder available — using HW decode")
            else:
                raise RuntimeError("v4l2h264dec not found")
        except Exception:
            pipeline = self._PIPELINE_SW.format(port=port)
            logger.warning("[GStreamerStream] v4l2h264dec not found. Falling back to software (avdec_h264) decode.")
        return pipeline

    def _connect(self) -> bool:
        """Opens the GStreamer pipeline via OpenCV's CAP_GSTREAMER backend."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        logger.info(f"[GStreamerStream] Opening pipeline:\n  {self._pipeline_str}")
        cap = cv2.VideoCapture(self._pipeline_str, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            self._cap = cap
            logger.info(f"[GStreamerStream] Pipeline open. Receiving on UDP port {self.port}")
            return True
        logger.error(
            "[GStreamerStream] Failed to open GStreamer pipeline. "
            "Check that gstreamer1.0-plugins-good and gstreamer1.0-plugins-bad are installed.\n"
            f"  Pipeline: {self._pipeline_str}"
        )
        self._cap = cap
        return False

    def _reader_loop(self):
        """Tight read loop — always stores the newest frame, drops old ones."""
        while not self._stopped:
            grabbed, frame = self._cap.read()
            if grabbed and frame is not None:
                with self._lock:
                    self._grabbed = True
                    self._frame = frame

    def read(self):
        """Returns (grabbed, frame) — always the newest decoded frame."""
        with self._lock:
            return self._grabbed, self._frame.copy() if self._frame is not None else None

    def isOpened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get(self, propId):
        if self._cap:
            val = self._cap.get(propId)
            # GStreamer pipelines often can't report resolution via CAP_PROP — use stored values
            if propId == cv2.CAP_PROP_FRAME_WIDTH and val == 0:
                return float(self._width)
            if propId == cv2.CAP_PROP_FRAME_HEIGHT and val == 0:
                return float(self._height)
            return val
        return 0

    def release(self):
        self._stopped = True
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info("[GStreamerStream] Released.")



def load_config(config_path: str) -> Dict[str, Any]:
    """Loads variables from configuration file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}. Using default settings.")
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Error loading yaml config: {e}. Using defaults.")
        return {}

def main():
    # Enable OpenCV CPU/thread optimizations
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)

    parser = argparse.ArgumentParser(description="Mini Golf AI Stroke Detection POC")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml", 
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--mock-mqtt", 
        action="store_true", 
        help="Use a mock MQTT publisher that only prints to console"
    )
    parser.add_argument(
        "--end-frame", 
        type=int, 
        default=-1, 
        help="Frame index to end processing (exclusive)"
    )
    parser.add_argument(
        "--save-debug", 
        action="store_true", 
        help="Draw visual overlays and save all processed frames to disk"
    )
    parser.add_argument(
        "--frame-skip", 
        type=int, 
        default=1, 
        help="Process every N-th frame (1=all, 2=every second frame, etc.)"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=640, 
        help="Requested camera width for local USB/RPi cameras (e.g., 640 for 4:3)"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=480, 
        help="Requested camera height for local USB/RPi cameras (e.g., 480 for 4:3)"
    )
    parser.add_argument(
        "--video", 
        type=str, 
        default="assets/input.mp4", 
        help="Path to input video file, device index (0), or RTSP URL (rtsp://...)"
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["classic", "yolo", "yolo-hybrid", "yolo-only"],
        default="classic",
        help="Detector to use: 'classic' (OpenCV HSV), 'yolo' (Hybrid COCO Person Masking + OpenCV), 'yolo-hybrid' (Custom YOLO Lock-on + OpenCV), or 'yolo-only' (Custom YOLO Ball-Only)"
    )
    parser.add_argument(
        "--rtsp-transport",
        type=str,
        choices=["tcp", "udp"],
        default="tcp",
        help="RTSP transport protocol. 'tcp' is reliable over WiFi; 'udp' has lower latency on LAN."
    )
    parser.add_argument(
        "--rpicam",
        action="store_true",
        help="Automatically launch rpicam-vid in the background for max FOV uncropped sensor mode."
    )
    parser.add_argument(
        "--gst",
        action="store_true",
        help=(
            "Use direct GStreamer RTP/UDP pipeline instead of RTSP. "
            "Eliminates RTSP overhead and uses RPi3 hardware H264 decode (v4l2h264dec). "
            "Camera must be configured to push H264 RTP to --gst-port on this machine's IP."
        )
    )
    parser.add_argument(
        "--gst-port",
        type=int,
        default=5000,
        help="UDP port to listen on for incoming RTP/H264 stream (used with --gst). Default: 5000."
    )
    parser.add_argument(
        "--gst-width",
        type=int,
        default=1280,
        help="Expected stream width in pixels (used with --gst when auto-detect fails). Default: 1280."
    )
    parser.add_argument(
        "--gst-height",
        type=int,
        default=720,
        help="Expected stream height in pixels (used with --gst when auto-detect fails). Default: 720."
    )
    parser.add_argument(
        "--gst-pipeline",
        type=str,
        default=None,
        help=(
            "Custom GStreamer pipeline string (overrides --gst-port template). "
            "Must end with 'appsink drop=true max-buffers=1 sync=false emit-signals=false'. "
            "Example: \"udpsrc port=5000 caps=\\\"application/x-rtp,encoding-name=H264\\\" ! "
            "rtph264depay ! h264parse ! v4l2h264dec ! videoconvert ! "
            "appsink drop=true max-buffers=1 sync=false emit-signals=false\""
        )
    )
    parser.add_argument(
        "--laptop",
        action="store_true",
        help="Tune tracker search window parameters for low FPS (e.g. 10 FPS) testing on a developer laptop."
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=None,
        help="Path to custom YOLO model weights (.pt or .onnx) to override config.yaml setting."
    )
    args = parser.parse_args()

    # 1. Load config
    logger.info("Initializing Mini Golf AI Stroke Detection...")
    config = load_config(args.config)
    config["low_fps"] = args.laptop
    
    if args.yolo_model:
        if "yolo_detector" not in config:
            config["yolo_detector"] = {}
        config["yolo_detector"]["model_path"] = args.yolo_model
        logger.info(f"[Config Override] Using YOLO model: {args.yolo_model}")

    # Start Flask Web Server in a background thread
    logger.info("Starting Flask web dashboard on http://0.0.0.0:5001")
    flask_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False, threaded=True),
        daemon=True
    )
    flask_thread.start()

    # 2. Extract configuration fields
    video_input_path = args.video
    video_output_path = "outputs/output.mp4"
    
    # Ensure correct folders exist
    os.makedirs(os.path.dirname(os.path.abspath(video_output_path)), exist_ok=True)
    
    # Exporters configuration
    csv_output_path = "outputs/strokes.csv"
    json_output_path = "outputs/strokes.json"
    
    # 3. Instantiate concrete implementations based on interfaces (Dependency Injection)
    detector_type = args.detector
    if detector_type == "yolo-hybrid":
        logger.info("Configuring Option A: Custom YOLO Lock-on + OpenCV Tracking Detector...")
        detector = CustomYoloOpenCVHybridDetector(config)
    elif detector_type == "yolo-only":
        logger.info("Configuring Pure YOLO-Only ONNX Ball Detector...")
        detector = YoloOnlyBallDetector(config)
    elif detector_type == "yolo":
        logger.info("Configuring Hybrid YOLO + OpenCV Color Detector...")
        detector = HybridYoloBallDetector(config)
    else:
        logger.info("Configuring Classic CV/Contour Sizing Detector...")
        detector = OpenCVBallDetector(config)

    # Motion analyzer setup (lower smoothing window on laptop to prevent motion lag at low FPS)
    motion_analyzer = MotionAnalyzer(smoothing_window=2 if args.laptop else 5)

    # MQTT Publisher setup
    if args.mock_mqtt:
        logger.info("Configuring Mock MQTT Publisher...")
        mqtt_publisher = MockMqttPublisher(config)
    else:
        logger.info("Configuring Production MQTT Publisher...")
        mqtt_publisher = MqttPublisher(config)

    # CSV and JSON Exporters setup
    csv_exporter = CsvExporter(csv_output_path)
    json_exporter = JsonExporter(json_output_path)

    # 4. Open Video Stream
    video_input_path = args.video
    is_rtsp   = video_input_path.startswith("rtsp://") or video_input_path.startswith("rtsps://") or video_input_path.startswith("tcp://")
    is_udp    = video_input_path.startswith("udp://")
    is_rtmp   = video_input_path.startswith("rtmp://")
    is_usb    = video_input_path.isdigit()
    is_sdp    = video_input_path.endswith(".sdp") and os.path.exists(video_input_path)
    is_gst    = args.gst or args.gst_pipeline is not None or args.rpicam
    is_live_stream = is_rtsp or is_udp or is_rtmp or is_usb or is_gst or is_sdp
    
    # Auto-start Raspberry Pi camera feed remotely via SSH if --laptop and tcp:// are used
    if args.laptop and video_input_path.startswith("tcp://"):
        try:
            parts = video_input_path.replace("tcp://", "").split(":")
            pi_ip = parts[0]
            pi_port = int(parts[1]) if len(parts) > 1 else 5000
            start_pi_camera_via_ssh(pi_ip, pi_port)
        except Exception as e:
            logger.warning(f"[SSH Auto-Start] Failed to parse TCP URL {video_input_path}: {e}")

    rpicam_process = None
    if args.rpicam:
        import subprocess
        logger.info("[RPICAM] Starting hardware camera subprocess for maximum Uncropped FOV...")
        cmd = [
            "rpicam-vid", "-t", "0", 
            "--mode", "4608:2592:10", 
            "--width", "640", "--height", "360", 
            "--framerate", "30", "--inline", 
            "--codec", "h264", "--profile", "baseline",
            "--listen", "-o", f"tcp://127.0.0.1:{args.gst_port}"
        ]
        rpicam_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Force the pipeline to the raw H264 TCP pipeline to guarantee delivery (no corruption)
        # Added leaky queue after decoder to drop frames if the CPU falls behind, ensuring absolute zero latency!
        args.gst_pipeline = f"tcpclientsrc host=127.0.0.1 port={args.gst_port} ! h264parse ! v4l2h264dec ! queue max-size-buffers=1 leaky=downstream ! videoconvert ! appsink drop=true max-buffers=1 sync=false emit-signals=false"
        time.sleep(1.5)  # Give camera hardware time to warm up

    if is_gst:
        # Direct GStreamer RTP/UDP pipeline — lowest latency, hardware H264 decode on RPi3
        logger.info(
            f"[Mode: GStreamer Direct] Listening for H264 RTP on UDP port {args.gst_port}. "
            f"Configure your camera to push H264 RTP to this machine's IP:{args.gst_port}."
        )
        cap = GStreamerStream(
            port=args.gst_port,
            custom_pipeline=args.gst_pipeline,
            width=args.gst_width,
            height=args.gst_height
        )
    elif is_sdp:
        # SDP file — GStreamer can open this directly as a session descriptor
        logger.info(f"[Mode: SDP/RTP] Opening SDP session descriptor: {video_input_path}")
        cap = RtspStream(video_input_path)
    elif is_rtsp:
        logger.info(f"[Mode: RTSP/{args.rtsp_transport.upper()}] {video_input_path}")
        cap = RtspStream(video_input_path, rtsp_transport=args.rtsp_transport)
    elif is_usb:
        camera_idx = int(video_input_path)
        logger.info(f"[Mode: USB Camera] /dev/video{camera_idx} at {args.width}x{args.height}")
        cap = cv2.VideoCapture(camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif is_udp or is_rtmp:
        logger.info(f"[Mode: UDP/RTMP] {video_input_path}")
        cap = RtspStream(video_input_path, rtsp_transport="udp")
    else:
        # Local video file
        if not os.path.exists(video_input_path):
            logger.error(f"Input video file not found: {video_input_path}")
            return
        logger.info(f"[Mode: Video File] {video_input_path}")
        cap = cv2.VideoCapture(video_input_path)

    # Read stream properties
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = -1 if is_live_stream else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        if is_live_stream:
            logger.warning(f"[RtspStream] Live stream {video_input_path} is offline. Running in background reconnect mode...")
            # Fallback to default properties to allow Flask dashboard and tracker to boot up
            fps = 30.0
            total_frames = -1
            frame_width = 960
            frame_height = 540
        else:
            logger.error(f"Failed to open video source: {video_input_path}")
            return

    # If the video source is portrait, we will be rotating it to landscape,
    # so swap dimensions to ensure all resizing calculations remain correct.
    if frame_height > frame_width:
        logger.info("[Auto-Rotate] Portrait video source detected. Swapping processing dimensions.")
        frame_width, frame_height = frame_height, frame_width

    if fps <= 0 or fps > 120:
        # RTSP cameras often report 0 or garbage FPS — default to 25 (common for IP cameras)
        fps = 25.0 if is_rtsp else 30.0
        logger.warning(f"FPS could not be read from stream. Defaulting to {fps:.0f} fps. "
                       f"Adjust --config if your camera runs at a different rate.")

    logger.info(f"Stream: {frame_width}x{frame_height} @ {fps:.2f} FPS | "
                f"Live: {is_live_stream} | RTSP: {is_rtsp}")

    # Adjust pixels_per_meter based on resizing
    resize_width = config.get("processing", {}).get("resize_width", 1280)
    if resize_width > 0 and frame_width != resize_width:
        scale = resize_width / frame_width
        original_ppm = config.get("video", {}).get("pixels_per_meter", 200.0)
        scaled_ppm = original_ppm * scale
        if "video" not in config:
            config["video"] = {}
        config["video"]["pixels_per_meter"] = scaled_ppm
        logger.info(f"Scaled pixels_per_meter from {original_ppm} to {scaled_ppm} for processing width {resize_width}")

    # Tracker setup (depends on FPS)
    logger.info("Configuring Kalman Ball Tracker...")
    tracker = KalmanBallTracker(config, fps=int(fps), frame_width=frame_width, low_fps=args.laptop)

    # Cache resize parameters once (avoids re-reading config dict every frame)
    resize_width = config.get("processing", {}).get("resize_width", 640)
    if resize_width > 0 and frame_width != resize_width:
        proc_scale  = resize_width / frame_width
        proc_height = int(frame_height * proc_scale)
        needs_resize = True
    else:
        proc_scale  = 1.0
        proc_height = frame_height
        needs_resize = False

    # 5. Initialize Stroke Engine (Decoupled Coordinator)
    stroke_engine = StrokeEngine(
        config=config,
        detector=detector,
        tracker=tracker,
        motion_analyzer=motion_analyzer,
        mqtt_publisher=mqtt_publisher,
        csv_exporter=csv_exporter,
        json_exporter=json_exporter
    )

    # 6. Initialize Visualizer Overlay
    visualizer = Visualizer(motion_analyzer=motion_analyzer)

    # 7. Ensure output image directories exist and are cleared of previous data
    frames_dir = "outputs/frames"
    strokes_dir = "outputs/strokes"
    import shutil
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    if os.path.exists(strokes_dir):
        shutil.rmtree(strokes_dir)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(strokes_dir, exist_ok=True)

    # Connect MQTT
    mqtt_publisher.connect()

    frame_idx  = 0
    start_time = time.monotonic()
    _last_perf_log = start_time
    camera_was_offline = True



    try:
        while True:
            if args.end_frame != -1 and frame_idx >= args.end_frame:
                break

            ret, frame = cap.read()
            if ret and frame is not None and camera_was_offline:
                camera_was_offline = False
                global STATUS_MESSAGE
                STATUS_MESSAGE = "Live feed active."
                logger.info("[RtspStream] Live stream came online! Triggering frontend video reload.")
                push_stroke_event(0, 0, "camera_connected")

            if not ret or frame is None:
                camera_was_offline = True
                if is_live_stream:
                    # Push a visual placeholder to the Flask dashboard so it remains active
                    now = time.monotonic()
                    if not hasattr(cap, "_last_placeholder_push") or now - cap._last_placeholder_push >= 0.2:
                        import numpy as np
                        cap._last_placeholder_push = now
                        placeholder = np.zeros((540, 960, 3), dtype=np.uint8)
                        cv2.putText(placeholder, "Waiting for Camera Connection...", (180, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(placeholder, f"Status: {STATUS_MESSAGE}", (160, 270),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(placeholder, "Please start the stream on the Raspberry Pi.", (160, 330),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
                        cv2.putText(placeholder, f"Source: {video_input_path}", (160, 380),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
                        _, jpeg = cv2.imencode('.jpg', placeholder)
                        push_frame(jpeg.tobytes())
                    
                    time.sleep(0.05)
                    continue
                break  # EOF for video files

            # Rotate portrait frames to landscape format for YOLO consistency
            fh, fw = frame.shape[:2]
            if fh > fw:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Frame skipping logic: skip processing but still read the frame to drain the camera buffer
            if args.frame_skip > 1 and frame_idx % args.frame_skip != 0:
                frame_idx += 1
                continue

            # Wall-clock timestamp for live streams; frame-count timestamp for video files
            if is_live_stream:
                timestamp = time.monotonic() - start_time
            else:
                timestamp = frame_idx / fps

            # Log FPS every 5 seconds
            now = time.monotonic()
            if now - _last_perf_log >= 5.0:
                elapsed = now - start_time
                actual_fps = frame_idx / elapsed if elapsed > 0 else 0
                progress = f"({frame_idx / total_frames * 100:.1f}%)" if total_frames > 0 else "live"
                logger.info(f"[Perf] Processed {frame_idx} frames | Actual: {actual_fps:.1f} fps | {progress}")
                _last_perf_log = now

            # Resize to processing resolution
            if needs_resize:
                proc_frame = cv2.resize(frame, (resize_width, proc_height))
            else:
                proc_frame = frame
                
            # Run core processing pipeline
            # Pass the effective FPS (stream FPS divided by frame skip) to keep velocity math accurate!
            active_metrics, active_states, stroke_events = stroke_engine.process_frame(
                frame=proc_frame,
                frame_idx=frame_idx,
                fps=fps / args.frame_skip,
                timestamp=timestamp
            )

            # Broadcast events to UI via SSE
            for track_id, stroke_count, event_type in stroke_events:
                push_stroke_event(track_id, stroke_count, event_type)
                logger.info(f"[Event] {event_type.upper()} | ball={track_id} | count={stroke_count}")

            # Draw visualizer overlay
            annotated_frame = visualizer.draw(proc_frame, active_metrics)
            

            
            # Encode frame as JPEG and push to web dashboard
            _, jpeg = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            push_frame(jpeg.tobytes())

            frame_idx += 1
                
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.error(f"Exception during processing loop: {e}", exc_info=True)
    finally:
        cap.release()

            
        mqtt_publisher.disconnect()
        
        if rpicam_process:
            logger.info("[RPICAM] Stopping background camera process...")
            rpicam_process.terminate()

        elapsed_sec = time.monotonic() - start_time
        logger.info("Processing complete.")
        logger.info("========================================")
        logger.info("        Execution Summary              ")
        logger.info("========================================")
        logger.info(f"Processed frames     : {frame_idx}")
        logger.info(f"Wall-clock time      : {elapsed_sec:.2f} seconds")
        logger.info(f"Average throughput   : {frame_idx / elapsed_sec:.2f} fps" if elapsed_sec > 0 else "N/A")
        logger.info(f"CSV Logs Output      : {os.path.abspath(csv_output_path)}")
        logger.info(f"JSON Logs Output     : {os.path.abspath(json_output_path)}")
        for track_id, sm in stroke_engine.state_machines.items():
            logger.info(f"Ball ID {track_id:2d}: {sm.stroke_count} strokes | Final State: {sm.state.value}")
        logger.info("========================================")


if __name__ == "__main__":
    main()
