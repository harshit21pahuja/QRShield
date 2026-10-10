import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
import numpy as np

from src.models.image_baseline import build_mobilenetv2_model, preprocess_image, gradcam_heatmap

model_path = PROJECT_ROOT / "artifacts" / "image_baseline_gpu" / "best_model.pt"
print(f"Loading checkpoint from: {model_path}", flush=True)

checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
model = build_mobilenetv2_model(num_classes=1, pretrained=False)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    print("Loaded model_state_dict from checkpoint dict", flush=True)
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    print("Loaded state_dict from checkpoint dict", flush=True)
elif isinstance(checkpoint, dict):
    state_dict = checkpoint
    print("Loaded state_dict directly", flush=True)
else:
    raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

model.load_state_dict(state_dict)
model.eval()
print("MobileNetV2 weights loaded successfully!", flush=True)

# Test forward pass with a dummy 224x224 RGB image
dummy_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
tensor = preprocess_image(dummy_img, size=224)
with torch.no_grad():
    logits = model(tensor.unsqueeze(0))
    prob = torch.sigmoid(logits).item()
print(f"Test MobileNetV2 dummy image prob: {prob:.4f}", flush=True)

# Test Grad-CAM
heatmap = gradcam_heatmap(model, tensor)
print(f"Grad-CAM generated successfully, shape: {heatmap.shape}, min: {heatmap.min():.4f}, max: {heatmap.max():.4f}", flush=True)

print("IMAGE MODEL & GRAD-CAM VERIFIED SUCCESSFULLY!", flush=True)
