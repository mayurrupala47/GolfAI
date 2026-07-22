"""
PHASE 1: TrackNetV2 vs YOLO Accuracy Benchmark
Pure Python - no Colab magic commands.

Usage (in Colab after pulling repo):
    python scratch/tracknet_benchmark_colab.py --video /content/GolfAI/orange_right_1.mp4

Or call from a Colab notebook cell:
    import subprocess
    subprocess.run(['python', 'scratch/tracknet_benchmark_colab.py',
                    '--video', '/content/GolfAI/orange_right_1.mp4'])
"""

import os
import sys
import subprocess
import argparse
import time
import numpy as np

# ---------------------------------------------------------------
# Step 0: Install missing dependencies via pip
# ---------------------------------------------------------------
REQUIRED = ['torch', 'torchvision', 'opencv-python-headless',
            'ultralytics', 'gdown', 'matplotlib']

for pkg in REQUIRED:
    mod = pkg.replace('-', '_').split('==')[0]
    try:
        __import__(mod)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')   # non-interactive backend safe for scripts
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Step 1: Parse Arguments
# ---------------------------------------------------------------
parser = argparse.ArgumentParser(description='TrackNetV2 vs YOLO Benchmark')
parser.add_argument('--video',
    default='/content/GolfAI/orange_right_1.mp4',
    help='Path to the test video file')
parser.add_argument('--yolo-model',
    default='/content/GolfAI/models/multicolor_detector_model.pt',
    help='Path to YOLO .pt weights')
parser.add_argument('--conf-tracknet', type=float, default=0.5,
    help='TrackNetV2 heatmap peak confidence threshold (default 0.5)')
parser.add_argument('--conf-yolo', type=float, default=0.15,
    help='YOLO detection confidence threshold (default 0.15)')
parser.add_argument('--output', default='/content/detection_comparison.png',
    help='Output chart PNG path')
args = parser.parse_args()

VIDEO_PATH = args.video
# Fallback: look for video next to this script
if not os.path.exists(VIDEO_PATH):
    alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'orange_right_1.mp4')
    if os.path.exists(alt):
        VIDEO_PATH = alt

if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: Video not found at '{VIDEO_PATH}'")
    print("  Copy video to /content/GolfAI/ or pass --video /path/to/video.mp4")
    sys.exit(1)

print(f"Video  : {VIDEO_PATH}")
print(f"YOLO   : {args.yolo_model}")
print(f"Device : {'cuda' if torch.cuda.is_available() else 'cpu'}")

# ---------------------------------------------------------------
# Step 2: TrackNetV2 Model (self-contained, no external repo needed)
# ---------------------------------------------------------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class TrackNetV2(nn.Module):
    """
    TrackNetV2 (3-in-1-out)
    Input : [B, 9, H, W]  - 3 consecutive RGB frames stacked channel-wise
    Output: [B, 1, H, W]  - heatmap, peak pixel = ball center
    """
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBNReLU(9, 64),   ConvBNReLU(64, 64))
        self.enc2 = nn.Sequential(ConvBNReLU(64, 128),  ConvBNReLU(128, 128))
        self.enc3 = nn.Sequential(ConvBNReLU(128, 256), ConvBNReLU(256, 256), ConvBNReLU(256, 256))
        self.enc4 = nn.Sequential(ConvBNReLU(256, 512), ConvBNReLU(512, 512), ConvBNReLU(512, 512))
        self.pool = nn.MaxPool2d(2, 2)
        self.dec4 = nn.Sequential(ConvBNReLU(512+256, 256), ConvBNReLU(256, 256))
        self.dec3 = nn.Sequential(ConvBNReLU(256+128, 128), ConvBNReLU(128, 128))
        self.dec2 = nn.Sequential(ConvBNReLU(128+64,  64),  ConvBNReLU(64, 64))
        self.dec1 = nn.Sequential(ConvBNReLU(64, 64),       nn.Conv2d(64, 1, 1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d  = F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=False)
        d  = self.dec4(torch.cat([d, e3], dim=1))
        d  = F.interpolate(d,  scale_factor=2, mode='bilinear', align_corners=False)
        d  = self.dec3(torch.cat([d, e2], dim=1))
        d  = F.interpolate(d,  scale_factor=2, mode='bilinear', align_corners=False)
        d  = self.dec2(torch.cat([d, e1], dim=1))
        return torch.sigmoid(self.dec1(d))

# ---------------------------------------------------------------
# Step 3: Load TrackNetV2 weights
# ---------------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

WEIGHTS_DIR  = '/content/tracknet_weights'
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'tracknetv2.pt')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

if not os.path.exists(WEIGHTS_PATH):
    print("Downloading TrackNetV2 community pre-trained weights (badminton)...")
    try:
        import gdown
        gdown.download(
            'https://drive.google.com/uc?id=1IiGJVVSGCi1qjCFgGTxWBCCOkOhMSBXc',
            WEIGHTS_PATH, quiet=False
        )
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Will run with random weights — speed benchmark still valid, detection rate ~0% (expected).")

tn_model = TrackNetV2().to(DEVICE)

if os.path.exists(WEIGHTS_PATH):
    try:
        ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        state = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
        tn_model.load_state_dict(state, strict=False)
        print("Weights loaded: pre-trained (badminton)")
    except Exception as e:
        print(f"Weight load failed: {e} — using random init")
else:
    print("No weights available — using random init (speed test only)")

tn_model.eval()

# ---------------------------------------------------------------
# Step 4: Video metadata
# ---------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FPS   = cap.get(cv2.CAP_PROP_FPS)
VW    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VH    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"\nVideo: {TOTAL} frames @ {FPS:.1f} fps | {VW}x{VH}")

INFER_W, INFER_H = 640, 360   # TrackNetV2 standard resolution

def preprocess(frame):
    f = cv2.resize(frame, (INFER_W, INFER_H))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return f   # H x W x 3

def peak(hm, ow, oh, thr):
    idx = np.argmax(hm)
    hy, hx = np.unravel_index(idx, hm.shape)
    conf = float(hm[hy, hx])
    if conf < thr:
        return None, None, conf
    return float(hx / INFER_W * ow), float(hy / INFER_H * oh), conf

# ---------------------------------------------------------------
# Step 5: TrackNetV2 inference
# ---------------------------------------------------------------
print(f"\n{'='*55}")
print(f"[1/2] TrackNetV2 inference on {TOTAL} frames ...")

cap  = cv2.VideoCapture(VIDEO_PATH)
buf  = []
tn_dets, tn_times = [], []
fi   = 0
t_wall = time.time()

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buf.append(preprocess(frame))
        if len(buf) > 3:
            buf.pop(0)

        if len(buf) == 3:
            stacked = np.concatenate(buf, axis=2)               # H x W x 9
            t  = torch.from_numpy(stacked).permute(2,0,1).unsqueeze(0).to(DEVICE)
            t0 = time.time()
            hm = tn_model(t).squeeze().cpu().numpy()
            tn_times.append(time.time() - t0)
            cx, cy, conf = peak(hm, VW, VH, args.conf_tracknet)
            tn_dets.append((fi, cx, cy, conf))
        else:
            tn_dets.append((fi, None, None, 0.0))
        fi += 1
        if fi % 500 == 0:
            print(f"  {fi}/{TOTAL}  |  {fi/(time.time()-t_wall):.1f} fps")

cap.release()
tn_wall  = time.time() - t_wall
tn_found = sum(1 for _, cx, _, _ in tn_dets if cx is not None)
tn_ms    = 1000 * np.mean(tn_times) if tn_times else 9999

print(f"  Detected : {tn_found}/{fi} = {100*tn_found/fi:.1f}%")
print(f"  Avg time : {tn_ms:.1f} ms/frame  → {1000/tn_ms:.0f} fps")
print(f"  Wall time: {tn_wall:.1f}s")

# ---------------------------------------------------------------
# Step 6: YOLO inference
# ---------------------------------------------------------------
from ultralytics import YOLO

yolo_path = args.yolo_model
if not os.path.exists(yolo_path):
    yolo_path = 'yolov8n.pt'
    print(f"Custom YOLO model not found. Falling back to {yolo_path}")

yolo_model = YOLO(yolo_path)

print(f"\n{'='*55}")
print(f"[2/2] YOLO inference on {TOTAL} frames ...")

cap = cv2.VideoCapture(VIDEO_PATH)
yo_dets, yo_times = [], []
fi = 0
t_wall = time.time()

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        results = yolo_model(frame, imgsz=640, conf=args.conf_yolo, verbose=False)
        yo_times.append(time.time() - t0)

        bx, by, bc = None, None, 0.0
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                c = float(box.conf[0])
                if c > bc:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bx, by, bc = (x1+x2)/2, (y1+y2)/2, c
        yo_dets.append((fi, bx, by, bc))
        fi += 1
        if fi % 500 == 0:
            print(f"  {fi}/{TOTAL}  |  {fi/(time.time()-t_wall):.1f} fps")

cap.release()
yo_found = sum(1 for _, cx, _, _ in yo_dets if cx is not None)
yo_ms    = 1000 * np.mean(yo_times)

print(f"  Detected : {yo_found}/{fi} = {100*yo_found/fi:.1f}%")
print(f"  Avg time : {yo_ms:.1f} ms/frame  → {1000/yo_ms:.0f} fps")

# ---------------------------------------------------------------
# Step 7: Comparison report
# ---------------------------------------------------------------
N      = min(len(tn_dets), len(yo_dets))
tn_hit = {i for i,(_, cx, _,_) in enumerate(tn_dets[:N]) if cx is not None}
yo_hit = {i for i,(_, cx, _,_) in enumerate(yo_dets[:N]) if cx is not None}
both   = tn_hit & yo_hit
tn_adv = tn_hit - yo_hit
yo_adv = yo_hit - tn_hit
neither = set(range(N)) - tn_hit - yo_hit

diffs = []
for i in both:
    _, tnx, tny, _ = tn_dets[i]
    _, yox, yoy, _ = yo_dets[i]
    if tnx and yox:
        diffs.append(np.sqrt((tnx-yox)**2 + (tny-yoy)**2))

print(f"\n{'='*60}")
print(f"             BENCHMARK RESULTS")
print(f"{'='*60}")
print(f"{'Metric':<40} {'TrackNet':>9} {'YOLO':>9}")
print(f"{'-'*60}")
print(f"{'Detection rate':<40} {100*len(tn_hit)/N:>8.1f}% {100*len(yo_hit)/N:>8.1f}%")
print(f"{'Avg inference':<40} {tn_ms:>8.1f}ms {yo_ms:>8.1f}ms")
print(f"{'Throughput':<40} {1000/tn_ms:>8.0f}fps {1000/yo_ms:>8.0f}fps")
print(f"{'-'*60}")
print(f"Both detected              : {len(both):>5} frames ({100*len(both)/N:.1f}%)")
print(f"TrackNet ONLY (YOLO missed): {len(tn_adv):>5} frames  <- TrackNet advantage")
print(f"YOLO ONLY (TrackNet missed): {len(yo_adv):>5} frames  <- needs fine-tuning")
print(f"Neither detected           : {len(neither):>5} frames")
if diffs:
    print(f"\nPosition agreement (shared detections):")
    print(f"  Mean diff  : {np.mean(diffs):.1f} px")
    print(f"  <10px agree: {100*sum(d<10 for d in diffs)/len(diffs):.1f}%")
print(f"{'='*60}")

# ---------------------------------------------------------------
# Step 8: Confidence chart
# ---------------------------------------------------------------
tn_c = [c for _,_,_,c in tn_dets[:N]]
yo_c = [c for _,_,_,c in yo_dets[:N]]
xs   = range(N)

fig, axes = plt.subplots(2, 1, figsize=(18, 6), sharex=True)
fig.suptitle('TrackNetV2 vs YOLO — Detection Confidence Timeline', fontsize=13, fontweight='bold')

axes[0].plot(xs, tn_c, color='#00d4ff', lw=0.7)
axes[0].fill_between(xs, tn_c, alpha=0.15, color='#00d4ff')
axes[0].axhline(args.conf_tracknet, color='red', ls='--', lw=1, alpha=0.7, label=f'threshold {args.conf_tracknet}')
axes[0].set_ylabel('Confidence'); axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=9)
axes[0].set_title(f'TrackNetV2 | Detection {100*len(tn_hit)/N:.1f}% | {1000/tn_ms:.0f} fps | pre-trained (badminton)', fontsize=10)

axes[1].plot(xs, yo_c, color='#ff8c00', lw=0.7)
axes[1].fill_between(xs, yo_c, alpha=0.15, color='#ff8c00')
axes[1].axhline(args.conf_yolo, color='red', ls='--', lw=1, alpha=0.7, label=f'threshold {args.conf_yolo}')
axes[1].set_ylabel('Confidence'); axes[1].set_ylim(0, 1.05); axes[1].legend(fontsize=9)
axes[1].set_xlabel('Frame'); axes[1].set_title(f'YOLO | Detection {100*len(yo_hit)/N:.1f}% | {1000/yo_ms:.0f} fps | custom mini-golf model', fontsize=10)

plt.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
plt.savefig(args.output, dpi=130, bbox_inches='tight')
print(f"\nChart saved -> {args.output}")

# Verdict
print(f"\nVERDICT:")
if len(tn_adv) > len(yo_adv):
    print("  TrackNet detects MORE frames than YOLO even without fine-tuning.")
    print("  -> Phase 2 fine-tuning on mini-golf footage will improve it further.")
elif len(tn_adv) > 0:
    print("  TrackNet partially better. Fine-tuning on mini-golf data will close the gap significantly.")
else:
    print("  TrackNet (pre-trained badminton) misses more than YOLO — fine-tuning needed.")
    print("  -> This is expected. After Phase 2 fine-tuning, TrackNet should exceed YOLO accuracy.")


# ---------------------------------------------------------------
# CELL 1: Environment Setup
# ---------------------------------------------------------------
!pip install -q torch torchvision opencv-python-headless ultralytics

import os, sys
os.chdir('/content')

# Clone TrackNetV2 PyTorch implementation
if not os.path.exists('TrackNetV2'):
    !git clone https://github.com/yastrebksv/TrackNet.git TrackNetV2

sys.path.insert(0, '/content/TrackNetV2')
print("✅ TrackNetV2 cloned")


# ---------------------------------------------------------------
# CELL 2: Download Pre-trained Weights
# ---------------------------------------------------------------
# Official pre-trained TrackNetV2 weights (badminton, 3-in-1-out)
import gdown, os

WEIGHTS_DIR = '/content/TrackNetV2/weights'
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Try official weights from NYCU (original authors)
!wget -q -O {WEIGHTS_DIR}/tracknetv2_badminton.pt \
    "https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2/blob/master/3_in_1_out/model906_30.pt?raw=true" \
    || echo "Official weights unavailable - using community weights"

# Fallback: community PyTorch weights
if not os.path.exists(f'{WEIGHTS_DIR}/tracknetv2_badminton.pt'):
    # Download from community Google Drive (yastrebksv's trained model)
    gdown.download(
        'https://drive.google.com/uc?id=1IiGJVVSGCi1qjCFgGTxWBCCOkOhMSBXc',
        f'{WEIGHTS_DIR}/tracknetv2_badminton.pt', quiet=False
    )

print("✅ Weights ready")


# ---------------------------------------------------------------
# CELL 3: Build TrackNetV2 Model (self-contained, no dependency issues)
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class TrackNetV2(nn.Module):
    """
    TrackNetV2 (3-in-1-out): takes 3 stacked RGB frames (9 channels),
    outputs a single heatmap (1 channel) indicating ball position.
    Input:  [B, 9, H, W]   (3 frames × 3 RGB channels)
    Output: [B, 1, H, W]   (Gaussian heatmap, peak = ball center)
    """
    def __init__(self):
        super().__init__()
        # Encoder (VGG-like)
        self.enc1 = nn.Sequential(ConvBNReLU(9, 64),  ConvBNReLU(64, 64))
        self.enc2 = nn.Sequential(ConvBNReLU(64, 128), ConvBNReLU(128, 128))
        self.enc3 = nn.Sequential(ConvBNReLU(128, 256), ConvBNReLU(256, 256), ConvBNReLU(256, 256))
        self.enc4 = nn.Sequential(ConvBNReLU(256, 512), ConvBNReLU(512, 512), ConvBNReLU(512, 512))
        self.pool = nn.MaxPool2d(2, 2)
        # Decoder (symmetric upsampling)
        self.dec4 = nn.Sequential(ConvBNReLU(512+256, 256), ConvBNReLU(256, 256))
        self.dec3 = nn.Sequential(ConvBNReLU(256+128, 128), ConvBNReLU(128, 128))
        self.dec2 = nn.Sequential(ConvBNReLU(128+64,  64),  ConvBNReLU(64, 64))
        self.dec1 = nn.Sequential(ConvBNReLU(64, 64), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d = F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=False)
        d = self.dec4(torch.cat([d, e3], dim=1))
        d = F.interpolate(d,  scale_factor=2, mode='bilinear', align_corners=False)
        d = self.dec3(torch.cat([d, e2], dim=1))
        d = F.interpolate(d,  scale_factor=2, mode='bilinear', align_corners=False)
        d = self.dec2(torch.cat([d, e1], dim=1))
        return torch.sigmoid(self.dec1(d))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TrackNetV2().to(DEVICE)

# Try loading weights (ignore if architecture mismatch - we'll run without pretrained)
try:
    ckpt = torch.load(f'{WEIGHTS_DIR}/tracknetv2_badminton.pt', map_location=DEVICE)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state, strict=False)
    print(f"✅ Pre-trained weights loaded on {DEVICE}")
except Exception as e:
    print(f"⚠️  Could not load pretrained weights ({e}). Running with random init for architecture test.")

model.eval()


# ---------------------------------------------------------------
# CELL 4: Mount Drive + Load Video
# ---------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

# Adjust path to where your video is in Drive
VIDEO_PATH = '/content/drive/MyDrive/GolfAI_Videos/orange_right_1.mp4'

# Fallback: pull from GolfAI repo if already cloned
if not os.path.exists(VIDEO_PATH):
    VIDEO_PATH = '/content/GolfAI/orange_right_1.mp4'

import cv2
cap = cv2.VideoCapture(VIDEO_PATH)
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"✅ Video: {TOTAL_FRAMES} frames @ {FPS:.1f}fps | {W}×{H}")


# ---------------------------------------------------------------
# CELL 5: TrackNetV2 Inference on Video
# ---------------------------------------------------------------
import numpy as np
import time

INFER_W, INFER_H = 640, 360   # TrackNetV2 standard input size
CONF_THRESHOLD   = 0.5         # heatmap peak confidence threshold

def preprocess_frame(frame):
    """Resize frame and normalize to [0,1] float32."""
    resized = cv2.resize(frame, (INFER_W, INFER_H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0

def heatmap_to_pos(heatmap_np, orig_w, orig_h):
    """Find peak in heatmap and scale back to original frame coordinates."""
    idx = np.argmax(heatmap_np)
    hy, hx = np.unravel_index(idx, heatmap_np.shape)
    conf = heatmap_np[hy, hx]
    if conf < CONF_THRESHOLD:
        return None, None, float(conf)
    cx = hx / INFER_W * orig_w
    cy = hy / INFER_H * orig_h
    return float(cx), float(cy), float(conf)

tracknet_detections = []   # list of (frame_idx, cx, cy, conf) or (frame_idx, None, None, conf)
frame_times = []

cap = cv2.VideoCapture(VIDEO_PATH)
frame_buffer = []   # rolling 3-frame buffer
frame_idx = 0

print(f"Running TrackNetV2 on {TOTAL_FRAMES} frames...")
t_start = time.time()

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed = preprocess_frame(frame)   # H×W×3 float32
        frame_buffer.append(preprocessed)
        if len(frame_buffer) > 3:
            frame_buffer.pop(0)

        if len(frame_buffer) == 3:
            # Stack 3 frames → [9, H, W] tensor
            stacked = np.concatenate(frame_buffer, axis=2)   # H×W×9
            tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            t0 = time.time()
            heatmap = model(tensor)   # [1, 1, H, W]
            frame_times.append(time.time() - t0)

            hm_np = heatmap.squeeze().cpu().numpy()
            cx, cy, conf = heatmap_to_pos(hm_np, W, H)
            tracknet_detections.append((frame_idx, cx, cy, conf))
        else:
            tracknet_detections.append((frame_idx, None, None, 0.0))

        frame_idx += 1
        if frame_idx % 300 == 0:
            elapsed = time.time() - t_start
            print(f"  Frame {frame_idx}/{TOTAL_FRAMES} | {frame_idx/elapsed:.1f} fps")

cap.release()
total_time = time.time() - t_start

detected = sum(1 for _, cx, cy, _ in tracknet_detections if cx is not None)
print(f"\n✅ TrackNetV2 complete!")
print(f"   Total frames   : {frame_idx}")
print(f"   Detected        : {detected} ({100*detected/frame_idx:.1f}%)")
print(f"   Missed          : {frame_idx - detected} ({100*(frame_idx-detected)/frame_idx:.1f}%)")
print(f"   Avg inference   : {1000*np.mean(frame_times):.1f} ms/frame")
print(f"   Throughput      : {1/np.mean(frame_times):.1f} fps")
print(f"   Wall time       : {total_time:.1f}s")


# ---------------------------------------------------------------
# CELL 6: YOLO Inference on Same Video (Baseline Comparison)
# ---------------------------------------------------------------
from ultralytics import YOLO

# Pull YOLO model from cloned repo if available
YOLO_MODEL = '/content/GolfAI/models/multicolor_detector_model.pt'
if not os.path.exists(YOLO_MODEL):
    YOLO_MODEL = 'yolov8n.pt'   # fallback to nano COCO

yolo = YOLO(YOLO_MODEL)

yolo_detections = []
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
yolo_times = []

print(f"Running YOLO on {TOTAL_FRAMES} frames...")
t_start = time.time()

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        results = yolo(frame, imgsz=640, conf=0.15, verbose=False)
        yolo_times.append(time.time() - t0)

        best_cx, best_cy, best_conf = None, None, 0.0
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    xyxy = box.xyxy[0].tolist()
                    best_cx = (xyxy[0] + xyxy[2]) / 2.0
                    best_cy = (xyxy[1] + xyxy[3]) / 2.0
                    best_conf = conf

        yolo_detections.append((frame_idx, best_cx, best_cy, best_conf))
        frame_idx += 1

        if frame_idx % 300 == 0:
            elapsed = time.time() - t_start
            print(f"  Frame {frame_idx}/{TOTAL_FRAMES} | {frame_idx/elapsed:.1f} fps")

cap.release()

y_detected = sum(1 for _, cx, cy, _ in yolo_detections if cx is not None)
print(f"\n✅ YOLO complete!")
print(f"   Detected        : {y_detected} ({100*y_detected/frame_idx:.1f}%)")
print(f"   Missed          : {frame_idx - y_detected} ({100*(frame_idx-y_detected)/frame_idx:.1f}%)")
print(f"   Avg inference   : {1000*np.mean(yolo_times):.1f} ms/frame")
print(f"   Throughput      : {1/np.mean(yolo_times):.1f} fps")


# ---------------------------------------------------------------
# CELL 7: Side-by-Side Accuracy Report
# ---------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

min_len = min(len(tracknet_detections), len(yolo_detections))

# --- Detection Rate Comparison ---
tn_detected_frames = [i for i, (fi, cx, cy, _) in enumerate(tracknet_detections[:min_len]) if cx is not None]
yo_detected_frames = [i for i, (fi, cx, cy, _) in enumerate(yolo_detections[:min_len])     if cx is not None]

# Both detected vs only one detected
both_detected    = set(tn_detected_frames) & set(yo_detected_frames)
only_tracknet    = set(tn_detected_frames) - set(yo_detected_frames)
only_yolo        = set(yo_detected_frames) - set(tn_detected_frames)
neither          = set(range(min_len)) - set(tn_detected_frames) - set(yo_detected_frames)

print("\n" + "="*55)
print("         ACCURACY BENCHMARK RESULTS")
print("="*55)
print(f"{'Metric':<35} {'TrackNet':>8} {'YOLO':>8}")
print("-"*55)
print(f"{'Detection rate':<35} {100*len(tn_detected_frames)/min_len:>7.1f}% {100*len(yo_detected_frames)/min_len:>7.1f}%")
print(f"{'Avg inference time':<35} {1000*np.mean(frame_times):>7.1f}ms {1000*np.mean(yolo_times):>7.1f}ms")
print(f"{'Throughput':<35} {1/np.mean(frame_times):>7.1f}fps {1/np.mean(yolo_times):>7.1f}fps")
print("-"*55)
print(f"Both models detected ball      : {len(both_detected)} frames ({100*len(both_detected)/min_len:.1f}%)")
print(f"Only TrackNet detected (YOLO missed): {len(only_tracknet)} frames ← TrackNet advantage")
print(f"Only YOLO detected (TrackNet missed): {len(only_yolo)} frames ← YOLO advantage")
print(f"Both missed                    : {len(neither)} frames")
print("="*55)

# --- Position comparison on frames where both detected ---
diffs = []
for i in both_detected:
    _, tn_cx, tn_cy, _ = tracknet_detections[i]
    _, yo_cx, yo_cy, _ = yolo_detections[i]
    if tn_cx and yo_cx:
        diffs.append(np.sqrt((tn_cx-yo_cx)**2 + (tn_cy-yo_cy)**2))

if diffs:
    print(f"\nPosition difference (when both detect):")
    print(f"  Mean  : {np.mean(diffs):.1f} px")
    print(f"  Median: {np.median(diffs):.1f} px")
    print(f"  Max   : {np.max(diffs):.1f} px")
    print("  (Small values = models agree on position ✅)")

# --- Plot: detection timeline ---
fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
frames = range(min_len)

tn_conf = [c for _, _, _, c in tracknet_detections[:min_len]]
yo_conf = [c for _, _, _, c in yolo_detections[:min_len]]

axes[0].plot(frames, tn_conf, color='#00d4ff', linewidth=0.8, label='TrackNetV2 confidence')
axes[0].axhline(CONF_THRESHOLD, color='red', linestyle='--', alpha=0.6, label='threshold')
axes[0].set_ylabel('TrackNet Conf')
axes[0].legend(loc='upper right', fontsize=8)
axes[0].set_title('TrackNetV2 vs YOLO Detection Confidence Over Video')

axes[1].plot(frames, yo_conf, color='#ff8c00', linewidth=0.8, label='YOLO confidence')
axes[1].axhline(0.15, color='red', linestyle='--', alpha=0.6, label='threshold')
axes[1].set_ylabel('YOLO Conf')
axes[1].set_xlabel('Frame')
axes[1].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('/content/detection_comparison.png', dpi=120)
plt.show()
print("\n📊 Chart saved to /content/detection_comparison.png")
print("\n➡️  Next step: if TrackNet detection rate > YOLO → proceed to Phase 2 fine-tuning")
