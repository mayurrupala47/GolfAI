import torch
import os
import sys

# Add root dir to path so we can import ai module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.tracknet_tracker import TrackNetV2

def export_onnx(weights_path='models/TrackNet_best.pt', output_path='models/TrackNet_best.onnx'):
    print(f"Loading PyTorch weights from {weights_path}...")
    model = TrackNetV2()
    
    try:
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)
        
    model.eval()

    # TrackNet input is 3 stacked frames (3 channels each) = 9 channels.
    # Training was done at 640x360.
    dummy_input = torch.randn(1, 9, 360, 640)
    
    print(f"Exporting ONNX graph to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # Dynamic batch size to support multi-camera batched inference later!
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("✅ ONNX export complete!")
    print("\nNext step on your Jetson Orin Nano:")
    print(f"trtexec --onnx={output_path} --saveEngine=models/TrackNet_best.engine --fp16")

if __name__ == '__main__':
    export_onnx()
