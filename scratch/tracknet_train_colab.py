"""
PHASE 2: TrackNetV2 Auto-labeling & Fine-Tuning
Pure Python — run directly on Colab.

Usage in Colab:
    python scratch/tracknet_train_colab.py --video /content/GolfAI/orange_right_1.mp4
"""

import os
import sys
import subprocess
import argparse
import time
import numpy as np
import cv2
cv2.setNumThreads(0)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------
# Step 0: Ensure dependencies
# ---------------------------------------------------------------
REQUIRED_PKGS = [
    ('cv2', 'opencv-python-headless'), ('torch', 'torch'),
    ('torchvision', 'torchvision'), ('ultralytics', 'ultralytics'),
    ('pandas', 'pandas')
]
for mod, pkg in REQUIRED_PKGS:
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--video', default='/content/GolfAI/orange_right_1.mp4')
parser.add_argument('--yolo-model', default='/content/GolfAI/models/multicolor_detector_model.pt')
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--output', default='/content/GolfAI/models/tracknet_minigolf.pt')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INFER_W, INFER_H = 640, 360

# ---------------------------------------------------------------
# Step 1: Auto-Labeling (YOLO + Interpolation)
# ---------------------------------------------------------------
print("\n" + "="*50)
print("[1/3] Auto-generating Dataset using YOLO...")
from ultralytics import YOLO

yolo_path = args.yolo_model if os.path.exists(args.yolo_model) else 'yolov8n.pt'
yolo = YOLO(yolo_path)

cap = cv2.VideoCapture(args.video)
frames_dir = getattr(args, 'frames_dir', './dataset_frames')
os.makedirs(frames_dir, exist_ok=True)

labels = []
fi = 0
VW = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
VH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Extract frames and run YOLO
while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Save resized frame for training
    resized = cv2.resize(frame, (INFER_W, INFER_H))
    cv2.imwrite(f"{frames_dir}/{fi:05d}.jpg", resized)
    
    # Auto-detect if using COCO or custom single-class model
    target_class = 0
    if len(yolo.names) > 32 and yolo.names[32] == "sports ball":
        target_class = 32
        
    res = yolo(frame, imgsz=640, conf=0.15, verbose=False)
    bx, by = None, None
    if res and len(res[0].boxes) > 0:
        # Find the box with highest confidence that matches our target class
        best_conf = 0
        best_box = None
        for box in res[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id == target_class and conf > best_conf:
                best_conf = conf
                best_box = box
                
        if best_box is not None:
            x1, y1, x2, y2 = best_box.xyxy[0].tolist()
            # Scale to TrackNet coordinates
            bx = ((x1+x2)/2) / VW * INFER_W
            by = ((y1+y2)/2) / VH * INFER_H
        
    labels.append({'frame': fi, 'x': bx, 'y': by})
    fi += 1
    if fi % 300 == 0: print(f"  Processed {fi} frames...")
cap.release()

# Interpolate missing frames (motion blur!)
df = pd.DataFrame(labels)
df['x'] = df['x'].interpolate(method='linear', limit=10) # Interpolate up to 10 frame gaps
df['y'] = df['y'].interpolate(method='linear', limit=10)
df['visibility'] = df['x'].notna().astype(int)

df.fillna(0, inplace=True)
df.to_csv('./labels.csv', index=False)
print(f"Generated {len(df)} labels. Visible/Interpolated frames: {df['visibility'].sum()}")

# ---------------------------------------------------------------
# Step 2: TrackNetV2 Architecture
# ---------------------------------------------------------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, i, o, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(i, o, k, s, p, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True)
        )

class TrackNetV2(nn.Module):
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
# Step 3: PyTorch Dataset
# ---------------------------------------------------------------
def generate_heatmap(x, y, w, h, sigma=2.5):
    hm = np.zeros((h, w), dtype=np.float32)
    if x <= 0 or y <= 0: return hm
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    dist = (xx - x)**2 + (yy - y)**2
    hm = np.exp(-dist / (2 * sigma**2))
    return hm

class GolfDataset(Dataset):
    def __init__(self, df, frames_dir):
        self.df = df
        self.frames_dir = frames_dir
        self.valid_indices = df[df['frame'] >= 2].index.tolist()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        row3 = self.df.iloc[self.valid_indices[idx]]
        row2 = self.df.iloc[self.valid_indices[idx]-1]
        row1 = self.df.iloc[self.valid_indices[idx]-2]

        path1 = f"{self.frames_dir}/{int(row1['frame']):05d}.jpg"
        path2 = f"{self.frames_dir}/{int(row2['frame']):05d}.jpg"
        path3 = f"{self.frames_dir}/{int(row3['frame']):05d}.jpg"
        
        f1 = cv2.imread(path1)
        f2 = cv2.imread(path2)
        f3 = cv2.imread(path3)
        
        try:
            f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            f3 = cv2.cvtColor(f3, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            stacked = np.concatenate([f1, f2, f3], axis=2)
        except Exception as e:
            print(f"\n[DATALOADER ERROR] Failed to load or concatenate frames.")
            print(f"Paths: {path1}, {path2}, {path3}")
            print(f"f1: {type(f1)} {f1.shape if isinstance(f1, np.ndarray) else ''}")
            print(f"f2: {type(f2)} {f2.shape if isinstance(f2, np.ndarray) else ''}")
            print(f"f3: {type(f3)} {f3.shape if isinstance(f3, np.ndarray) else ''}")
            raise e
        X = torch.from_numpy(stacked).permute(2,0,1).float()

        if row3['visibility'] == 1:
            hm = generate_heatmap(row3['x'], row3['y'], INFER_W, INFER_H)
        else:
            hm = np.zeros((INFER_H, INFER_W), dtype=np.float32)
            
        Y = torch.from_numpy(hm).unsqueeze(0).float()
        return X, Y

# ---------------------------------------------------------------
# Step 4: Training Loop
# ---------------------------------------------------------------
print("\n" + "="*50)
print("[2/3] Initializing TrackNetV2 Training...")

dataset = GolfDataset(df, frames_dir)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

model = TrackNetV2().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.BCELoss()

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    pred = torch.clamp(pred, 1e-6, 1-1e-6)
    pt = torch.where(target == 1, pred, 1 - pred)
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)
    loss = -alpha_t * (1 - pt)**gamma * torch.log(pt)
    return loss.mean()

print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)} samples")
print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

best_val_loss = float('inf')
for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    t0 = time.time()
    
    for i, (X, Y) in enumerate(train_loader):
        if i == 0:
            print(f"\n  [Device: {DEVICE}] Starting batch 0...")
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, Y) + focal_loss(pred, Y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print(f"  Epoch {epoch+1} - Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}", end='\r', flush=True)
        
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, Y in val_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            pred = model(X)
            val_loss += (criterion(pred, Y) + focal_loss(pred, Y)).item()
            
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    t_ep = time.time() - t0
    
    print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {t_ep:.1f}s")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        torch.save(model.state_dict(), args.output)

print("\n" + "="*50)
print(f"[3/3] Training Complete! Best model saved to: {args.output}")
print("You can now download this .pt file and use it in your local Jetson pipeline.")
