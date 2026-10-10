import sys
from pathlib import Path
import qrcode
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import fusion_engine, qr_decoder

print("1. Creating safe synthetic QR code...", flush=True)
test_url = "https://secure-login-account-update.suspicious-domain-example.com/login?token=abc12345"
qr = qrcode.QRCode(box_size=10, border=4)
qr.add_data(test_url)
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")

import io
img_bytes = io.BytesIO()
img.save(img_bytes, format="PNG")
raw_bytes = img_bytes.getvalue()

print("2. Running complete predict_pipeline...", flush=True)
res = fusion_engine.predict_pipeline(raw_bytes, generate_gradcam=True)

print("\n--- INFERENCE RESULT ---")
print(f"Status: {res['status']}")
print(f"Decoded URL: {res['decoded_url']}")
print(f"URL Probability: {res['url_probability']}")
print(f"Image Probability: {res['image_probability']}")
print(f"Fusion Probability: {res['fusion_probability']}")
print(f"Calibrated Probability: {res['calibrated_probability']}")
print(f"Prediction: {res['prediction']}")
print(f"Risk Level: {res['risk_level']} ({res['risk_label']})")
print(f"URL Explanations: {len(res['explanation']['url'])} findings")
for exp in res['explanation']['url']:
    print(f"   [{exp.get('impact')}] {exp.get('feature')}: {exp.get('finding')}")
print(f"Image Explanations: {len(res['explanation']['image'])} findings")
print(f"Grad-CAM Available: {res['explanation']['gradcam']['available']}")
print(f"Grad-CAM Focus Region: {res['explanation']['gradcam']['focus_region']}")
print(f"Safety Notice: {res['safety_notice']}")

assert res["status"] == "ok"
assert res["decoded_url"] == test_url
assert 0.0 <= res["calibrated_probability"] <= 1.0

print("\nALL INFERENCE ENGINE CHECKS PASSED PERFECTLY!", flush=True)
