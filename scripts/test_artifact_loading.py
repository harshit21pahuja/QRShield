import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import torch
from src.models.image_baseline import build_mobilenetv2_model


artifacts_dir = Path("artifacts")

print("1. Loading URL model...", flush=True)
url_model_path = artifacts_dir / "url_baseline_phase3" / "url_random_forest.joblib"
url_rf = joblib.load(url_model_path)
print(f"   URL model loaded: {type(url_rf)}, n_estimators={getattr(url_rf, 'n_estimators', None)}", flush=True)

print("2. Loading Isotonic Calibrator...", flush=True)
calibrator_path = artifacts_dir / "fusion_phase3" / "isotonic_calibrator.joblib"
calibrator = joblib.load(calibrator_path)
print(f"   Calibrator loaded: {type(calibrator)}", flush=True)

print("3. Loading Inference Config...", flush=True)
config_path = artifacts_dir / "fusion_phase3" / "inference_config.json"
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)
print(f"   Config loaded: {config}", flush=True)

print("4. Loading MobileNetV2 checkpoint...", flush=True)
image_model_path = artifacts_dir / "image_baseline_gpu" / "best_model.pt"
checkpoint = torch.load(image_model_path, map_location="cpu", weights_only=False)
print(f"   Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'tensor/model'}", flush=True)

model = build_mobilenetv2_model(num_classes=1, pretrained=False)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
elif isinstance(checkpoint, dict):
    model.load_state_dict(checkpoint)
else:
    model = checkpoint
model.eval()
print("   MobileNetV2 model successfully initialized and set to eval!", flush=True)

print("ALL ARTIFACTS VERIFIED SUCCESSFULLY!", flush=True)
