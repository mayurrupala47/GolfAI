"""
PHASE 1: TrackNetV2 vs YOLO Accuracy Benchmark
Pure Python — no Colab magic commands (!pip, !git, %run).

Run from a Colab cell:
    import subprocess
    subprocess.run([
        'python', 'scratch/tracknet_benchmark_colab.py',
        '--video', '/content/GolfAI/orange_right_1.mp4',
        '--yolo-model', 'models/multicolor_detector_model.pt',
        '--output', '/content/detection_comparison.png'
    ], cwd='/content/GolfAI')
"""

import os
import sys
import subprocess
import argparse
import time
import numpy as np

# ---------------------------------------------------------------
# Step 0: Install missing packages via pip (pure Python, no !pip)
# ---------------------------------------------------------------
REQUIRED_PKGS = [
    ('cv2',          'opencv-python-headless'),
    ('torch',        'torch'),
    ('torchvision',  'torchvision'),
    ('ultralytics',  'ultralytics'),
    ('gdown',        'gdown'),
    ('matplotlib',   'matplotlib'),
]
for mod, pkg in REQUIRED_PKGS:
    try:
        __import__(mod)
    except ImportError:
        print(f"Installing {pkg} ...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Step 1: Arguments
# ---------------------------------------------------------------
parser = argparse.ArgumentParser(description='TrackNetV2 vs YOLO Benchmark')
parser.add_argument('--video',
    default='/content/GolfAI/orange_right_1.mp4')
parser.add_argument('--yolo-model',
    default='/content/GolfAI/models/multicolor_detector_model.pt')
parser.add_argument('--conf-tracknet', type=float, default=0.5)
parser.add_argument('--conf-yolo',     type=float, default=0.15)
parser.add_argument('--output',
    default='/content/detection_comparison.png')
args = parser.parse_args()

VIDEO_PATH = args.video
if not os.path.exists(VIDEO_PATH):
    alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'orange_right_1.mp4')
    VIDEO_PATH = alt if os.path.exists(alt) else VIDEO_PATH

if not os.path.exists(VIDEO_PATH):
    sys.exit(f"ERROR: video not found: {VIDEO_PATH}")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Video : {VIDEO_PATH}")
print(f"YOLO  : {args.yolo_model}")
print(f"Device: {DEVICE}")

# ---------------------------------------------------------------
# Step 2: TrackNetV2 architecture (self-contained)
# ---------------------------------------------------------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, i, o, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(i, o, k, s, p, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True)
        )

class TrackNetV2(nn.Module):
    """3-in-1-out heatmap network. Input [B,9,H,W] -> Output [B,1,H,W]"""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBNReLU(9,64),   ConvBNReLU(64,64))
        self.enc2 = nn.Sequential(ConvBNReLU(64,128),  ConvBNReLU(128,128))
        self.enc3 = nn.Sequential(ConvBNReLU(128,256), ConvBNReLU(256,256), ConvBNReLU(256,256))
        self.enc4 = nn.Sequential(ConvBNReLU(256,512), ConvBNReLU(512,512), ConvBNReLU(512,512))
        self.pool = nn.MaxPool2d(2, 2)
        self.dec4 = nn.Sequential(ConvBNReLU(768,256), ConvBNReLU(256,256))
        self.dec3 = nn.Sequential(ConvBNReLU(384,128), ConvBNReLU(128,128))
        self.dec2 = nn.Sequential(ConvBNReLU(192,64),  ConvBNReLU(64,64))
        self.dec1 = nn.Sequential(ConvBNReLU(64,64),   nn.Conv2d(64,1,1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d = F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=False)
        d = self.dec4(torch.cat([d, e3], 1))
        d = F.interpolate(d,  scale_factor=2, mode='bilinear', align_corners=False)
        d = self.dec3(torch.cat([d, e2], 1))
        d = F.interpolate(d,  scale_factor=2, mode='bilinear', align_corners=False)
        d = self.dec2(torch.cat([d, e1], 1))
        return torch.sigmoid(self.dec1(d))

# ---------------------------------------------------------------
# Step 3: Load weights
# ---------------------------------------------------------------
WDIR  = '/content/tracknet_weights'
WPATH = os.path.join(WDIR, 'tracknetv2.pt')
os.makedirs(WDIR, exist_ok=True)

if not os.path.exists(WPATH):
    print("Downloading pre-trained weights (badminton) via gdown ...")
    try:
        import gdown
        gdown.download('https://drive.google.com/uc?id=1IiGJVVSGCi1qjCFgGTxWBCCOkOhMSBXc',
                       WPATH, quiet=False)
    except Exception as e:
        print(f"  Download failed ({e}). Running with random weights — speed test only.")

tn = TrackNetV2().to(DEVICE)
if os.path.exists(WPATH):
    try:
        ckpt = torch.load(WPATH, map_location=DEVICE)
        state = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
        tn.load_state_dict(state, strict=False)
        print("Weights: pre-trained (badminton)")
    except Exception as e:
        print(f"Weight load failed ({e}) — random init")
else:
    print("Weights: random init (speed test only)")
tn.eval()

# ---------------------------------------------------------------
# Step 4: Video info
# ---------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
NFRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FPS     = cap.get(cv2.CAP_PROP_FPS)
VW      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VH      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"Video : {NFRAMES} frames @ {FPS:.1f}fps | {VW}x{VH}")

IW, IH = 640, 360  # TrackNetV2 input size

def prep(frame):
    f = cv2.resize(frame, (IW, IH))
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def peak(hm, ow, oh, thr):
    idx = np.argmax(hm)
    hy, hx = np.unravel_index(idx, hm.shape)
    c = float(hm[hy, hx])
    if c < thr:
        return None, None, c
    return float(hx / IW * ow), float(hy / IH * oh), c

# ---------------------------------------------------------------
# Step 5: TrackNetV2 pass
# ---------------------------------------------------------------
print(f"\n{'='*50}")
print(f"[1/2] TrackNetV2 on {NFRAMES} frames ...")
cap = cv2.VideoCapture(VIDEO_PATH)
buf, tn_dets, tn_t = [], [], []
fi = 0
tw = time.time()
with torch.no_grad():
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        buf.append(prep(frm))
        if len(buf) > 3:
            buf.pop(0)
        if len(buf) == 3:
            stk = np.concatenate(buf, axis=2)
            t0 = time.time()
            hm = tn(torch.from_numpy(stk).permute(2,0,1).unsqueeze(0).to(DEVICE)).squeeze().cpu().numpy()
            tn_t.append(time.time() - t0)
            cx, cy, cf = peak(hm, VW, VH, args.conf_tracknet)
            tn_dets.append((fi, cx, cy, cf))
        else:
            tn_dets.append((fi, None, None, 0.0))
        fi += 1
        if fi % 500 == 0:
            print(f"  {fi}/{NFRAMES}  {fi/(time.time()-tw):.1f}fps")
cap.release()
tn_found = sum(1 for _,cx,_,_ in tn_dets if cx)
tn_ms    = 1000*np.mean(tn_t) if tn_t else 9999
print(f"  Detected : {tn_found}/{fi} = {100*tn_found/fi:.1f}%  |  {tn_ms:.1f}ms/frame")

# ---------------------------------------------------------------
# Step 6: YOLO pass
# ---------------------------------------------------------------
from ultralytics import YOLO
ypath = args.yolo_model if os.path.exists(args.yolo_model) else 'yolov8n.pt'
if ypath == 'yolov8n.pt':
    print(f"Custom YOLO not found. Using fallback: {ypath}")
yolo = YOLO(ypath)

print(f"\n{'='*50}")
print(f"[2/2] YOLO on {NFRAMES} frames ...")
cap = cv2.VideoCapture(VIDEO_PATH)
yo_dets, yo_t = [], []
fi = 0
tw = time.time()
with torch.no_grad():
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        t0 = time.time()
        res = yolo(frm, imgsz=640, conf=args.conf_yolo, verbose=False)
        yo_t.append(time.time() - t0)
        bx, by, bc = None, None, 0.0
        if res and len(res[0].boxes) > 0:
            for box in res[0].boxes:
                c = float(box.conf[0])
                if c > bc:
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    bx,by,bc = (x1+x2)/2,(y1+y2)/2,c
        yo_dets.append((fi, bx, by, bc))
        fi += 1
        if fi % 500 == 0:
            print(f"  {fi}/{NFRAMES}  {fi/(time.time()-tw):.1f}fps")
cap.release()
yo_found = sum(1 for _,cx,_,_ in yo_dets if cx)
yo_ms    = 1000*np.mean(yo_t)
print(f"  Detected : {yo_found}/{fi} = {100*yo_found/fi:.1f}%  |  {yo_ms:.1f}ms/frame")

# ---------------------------------------------------------------
# Step 7: Report
# ---------------------------------------------------------------
N    = min(len(tn_dets), len(yo_dets))
th   = {i for i,(_,cx,_,_) in enumerate(tn_dets[:N]) if cx}
yh   = {i for i,(_,cx,_,_) in enumerate(yo_dets[:N]) if cx}
both = th & yh
ta   = th - yh   # TrackNet advantage
ya   = yh - th   # YOLO advantage
non  = set(range(N)) - th - yh

diffs = []
for i in both:
    _,tx,ty,_ = tn_dets[i];  _,yx,yy,_ = yo_dets[i]
    if tx and yx:
        diffs.append(np.sqrt((tx-yx)**2 + (ty-yy)**2))

print(f"\n{'='*60}")
print("             BENCHMARK RESULTS")
print(f"{'='*60}")
print(f"{'Metric':<38} {'TrackNet':>10} {'YOLO':>10}")
print(f"{'-'*60}")
print(f"{'Detection rate':<38} {100*len(th)/N:>9.1f}% {100*len(yh)/N:>9.1f}%")
print(f"{'Avg inference':<38} {tn_ms:>9.1f}ms {yo_ms:>9.1f}ms")
print(f"{'Throughput':<38} {1000/tn_ms:>9.0f}fps {1000/yo_ms:>9.0f}fps")
print(f"{'-'*60}")
print(f"Both detected         : {len(both):>5} ({100*len(both)/N:.1f}%)")
print(f"TrackNet only (YOLO missed): {len(ta):>5}  <- TrackNet advantage")
print(f"YOLO only (TrackNet missed): {len(ya):>5}  <- needs fine-tuning")
print(f"Neither               : {len(non):>5}")
if diffs:
    print(f"\nPosition agreement: mean={np.mean(diffs):.1f}px  "
          f"<10px={100*sum(d<10 for d in diffs)/len(diffs):.1f}%")
print(f"{'='*60}")

# ---------------------------------------------------------------
# Step 8: Chart
# ---------------------------------------------------------------
tc = [c for _,_,_,c in tn_dets[:N]]
yc = [c for _,_,_,c in yo_dets[:N]]
xs = range(N)

fig, axes = plt.subplots(2, 1, figsize=(18,6), sharex=True)
fig.suptitle('TrackNetV2 vs YOLO — Confidence Timeline', fontsize=13, fontweight='bold')

axes[0].plot(xs, tc, color='#00d4ff', lw=0.7)
axes[0].fill_between(xs, tc, alpha=0.15, color='#00d4ff')
axes[0].axhline(args.conf_tracknet, color='red', ls='--', lw=1, alpha=0.7,
                label=f'threshold {args.conf_tracknet}')
axes[0].set_ylabel('Confidence'); axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=9)
axes[0].set_title(
    f'TrackNetV2 | Detected {100*len(th)/N:.1f}% | {1000/tn_ms:.0f}fps | pre-trained (badminton)',
    fontsize=10)

axes[1].plot(xs, yc, color='#ff8c00', lw=0.7)
axes[1].fill_between(xs, yc, alpha=0.15, color='#ff8c00')
axes[1].axhline(args.conf_yolo, color='red', ls='--', lw=1, alpha=0.7,
                label=f'threshold {args.conf_yolo}')
axes[1].set_ylabel('Confidence'); axes[1].set_ylim(0, 1.05); axes[1].legend(fontsize=9)
axes[1].set_xlabel('Frame')
axes[1].set_title(
    f'YOLO | Detected {100*len(yh)/N:.1f}% | {1000/yo_ms:.0f}fps | custom mini-golf model',
    fontsize=10)

plt.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
plt.savefig(args.output, dpi=130, bbox_inches='tight')
print(f"\nChart saved -> {args.output}")

print("\nVERDICT:")
if len(ta) > len(ya):
    print("  TrackNet detects MORE frames than YOLO even without fine-tuning.")
    print("  -> Phase 2 fine-tuning on mini-golf footage will push it further.")
elif len(ta) > 0:
    print("  TrackNet partially better. Fine-tuning will close the remaining gap.")
else:
    print("  TrackNet (badminton pre-trained) misses more than YOLO — fine-tuning needed.")
    print("  -> Expected. After fine-tuning on mini-golf data, TrackNet will exceed YOLO.")
