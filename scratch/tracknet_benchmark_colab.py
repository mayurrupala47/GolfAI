# ================================================================
# PHASE 1: TrackNetV2 vs YOLO Accuracy Benchmark
# Run this in Google Colab
# ================================================================

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
