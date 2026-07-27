import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class ConvBNReLU(nn.Module):
    def __init__(self, i, o, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(i, o, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(o)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_1 = ConvBNReLU(in_ch, out_ch)
        self.conv_2 = ConvBNReLU(out_ch, out_ch)
    def forward(self, x):
        return self.conv_2(self.conv_1(x))

class TripleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_1 = ConvBNReLU(in_ch, out_ch)
        self.conv_2 = ConvBNReLU(out_ch, out_ch)
        self.conv_3 = ConvBNReLU(out_ch, out_ch)
    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x)))

class TrackNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_block_1 = DoubleConv(27, 64)
        self.down_block_2 = DoubleConv(64, 128)
        self.down_block_3 = TripleConv(128, 256)
        self.bottleneck = TripleConv(256, 512)
        
        self.up_block_1 = TripleConv(768, 256)
        self.up_block_2 = DoubleConv(384, 128)
        self.up_block_3 = DoubleConv(192, 64)
        self.predictor = nn.Conv2d(64, 8, kernel_size=1, bias=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        d1 = self.down_block_1(x)
        d2 = self.down_block_2(self.pool(d1))
        d3 = self.down_block_3(self.pool(d2))
        b = self.bottleneck(self.pool(d3))
        
        u1 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.up_block_1(torch.cat([u1, d3], dim=1))
        
        u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = self.up_block_2(torch.cat([u2, d2], dim=1))
        
        u3 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        u3 = self.up_block_3(torch.cat([u3, d1], dim=1))
        
        return torch.sigmoid(self.predictor(u3))

import os

class TrackNetEngine:
    def __init__(self, weights_path='models/TrackNet_best.pt', conf_threshold=0.3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[TrackNetEngine] Initializing on {self.device}...")
        
        self.infer_w = 640
        self.infer_h = 360
        self.frame_buffer = []
        self.conf_threshold = conf_threshold
        
        # Auto-detect if ONNX alternative exists
        onnx_alternative = weights_path.replace('.pt', '.onnx')
        if os.path.exists(onnx_alternative):
            weights_path = onnx_alternative
            self.is_onnx = True
        else:
            self.is_onnx = weights_path.endswith('.onnx')
        
        if self.is_onnx:
            import onnxruntime as ort
            print(f"[TrackNetEngine] Loading ONNX model from {weights_path}...")
            # Use TensorRT and CUDA providers if GPU is active
            providers = ['TensorRTExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(weights_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        else:
            self.model = TrackNetV3().to(self.device)
            try:
                ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
                state = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
                self.model.load_state_dict(state, strict=True)
                print(f"[TrackNetEngine] Successfully loaded weights strictly from {weights_path}")
            except Exception as e:
                print(f"[TrackNetEngine] ERROR loading weights: {e}")
            self.model.eval()
            if self.device == 'cuda':
                self.model = self.model.half()
                try:
                    print("[TrackNetEngine] Compiling PyTorch model with torch.compile() for max GPU speed...")
                    self.model = torch.compile(self.model)
                    print("[TrackNetEngine] torch.compile() compilation successful!")
                except Exception as e:
                    print(f"[TrackNetEngine] torch.compile() fallback to eager mode: {e}")

    def preprocess(self, frame):
        """Resize and convert to RGB (uint8)"""
        resized = cv2.resize(frame, (self.infer_w, self.infer_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb

    def extract_peak(self, heatmap, orig_w, orig_h):
        """Find the peak coordinate in the heatmap above threshold with compactness filtering."""
        idx = np.argmax(heatmap)
        hy, hx = np.unravel_index(idx, heatmap.shape)
        conf = float(heatmap[hy, hx])
        
        if conf < self.conf_threshold:
            return None, conf
            
        # Compactness filter: golf balls have sharp peaks, shirts/shoes have large diffuse blobs
        y_min = max(0, hy - 7)
        y_max = min(self.infer_h, hy + 8)
        x_min = max(0, hx - 7)
        x_max = min(self.infer_w, hx + 8)
        
        local_window = heatmap[y_min:y_max, x_min:x_max]
        active_pixels = np.sum(local_window > (conf * 0.5))
        
        # If the activation is too large/diffuse (e.g. > 45 pixels active),
        # it is a large object (shirt/shoe/clutter) rather than a tiny golf ball.
        if active_pixels > 45:
            return None, conf
            
        cx = hx / self.infer_w * orig_w
        cy = hy / self.infer_h * orig_h
        return (cx, cy), conf

    def update(self, frame):
        """
        Ingest a new frame and return the detected (cx, cy) and confidence.
        Requires 9 frames to output a valid detection.
        """
        orig_h, orig_w = frame.shape[:2]
        prepped = self.preprocess(frame)
        self.frame_buffer.append(prepped)
        
        if len(self.frame_buffer) > 9:
            self.frame_buffer.pop(0)
            
        if len(self.frame_buffer) < 9:
            return None, 0.0
            
        # Stack 9 frames (HxWx27)
        stacked = np.concatenate(self.frame_buffer, axis=2)
        
        if self.is_onnx:
            # ONNX Runtime Inference (keep float32)
            tensor = np.transpose(stacked, (2, 0, 1))
            tensor = np.expand_dims(tensor, axis=0).astype(np.float32) / 255.0
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
            heatmaps = outputs[0][0] # shape (8, 360, 640)
        else:
            # PyTorch Inference
            # 1. Convert uint8 numpy stacked frames to torch tensor and transfer to GPU
            tensor = torch.from_numpy(stacked).permute(2,0,1).unsqueeze(0).to(self.device)
            # 2. Normalize and cast to FP16 (half precision) on GPU!
            if self.device == 'cuda':
                tensor = tensor.half() / 255.0
            else:
                tensor = tensor.float() / 255.0
                
            with torch.no_grad():
                heatmaps = self.model(tensor).squeeze(0).float().cpu().numpy()
            
        # Predict using the last predicted heatmap (corresponding to the current frame)
        pos, conf = self.extract_peak(heatmaps[-1], orig_w, orig_h)
        return pos, conf
