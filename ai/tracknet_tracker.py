import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# TrackNetV2 Model Definition (must match training exactly)
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

class TrackNetEngine:
    def __init__(self, weights_path='models/TrackNet_best.pt', conf_threshold=0.3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[TrackNetEngine] Initializing on {self.device}...")
        
        self.model = TrackNetV2().to(self.device)
        try:
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            state = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
            self.model.load_state_dict(state, strict=False)
            print(f"[TrackNetEngine] Successfully loaded weights from {weights_path}")
        except Exception as e:
            print(f"[TrackNetEngine] ERROR loading weights: {e}")
            
        self.model.eval()
        self.conf_threshold = conf_threshold
        
        self.infer_w = 640
        self.infer_h = 360
        self.frame_buffer = []

    def preprocess(self, frame):
        """Resize and normalize frame"""
        resized = cv2.resize(frame, (self.infer_w, self.infer_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    def extract_peak(self, heatmap, orig_w, orig_h):
        """Find the peak coordinate in the heatmap above threshold."""
        idx = np.argmax(heatmap)
        hy, hx = np.unravel_index(idx, heatmap.shape)
        conf = float(heatmap[hy, hx])
        
        if conf < self.conf_threshold:
            return None, 0.0
            
        cx = hx / self.infer_w * orig_w
        cy = hy / self.infer_h * orig_h
        return (cx, cy), conf

    def update(self, frame):
        """
        Ingest a new frame and return the detected (cx, cy) and confidence.
        Requires 3 frames to output a valid detection.
        """
        orig_h, orig_w = frame.shape[:2]
        prepped = self.preprocess(frame)
        self.frame_buffer.append(prepped)
        
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)
            
        if len(self.frame_buffer) < 3:
            return None, 0.0
            
        # Stack 3 frames (HxWx9)
        stacked = np.concatenate(self.frame_buffer, axis=2)
        
        # To PyTorch tensor (1x9xHxW)
        tensor = torch.from_numpy(stacked).permute(2,0,1).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            heatmap = self.model(tensor).squeeze().cpu().numpy()
            
        pos, conf = self.extract_peak(heatmap, orig_w, orig_h)
        return pos, conf
