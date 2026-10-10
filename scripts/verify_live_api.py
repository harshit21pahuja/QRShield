import json
import urllib.request
import qrcode
from PIL import Image
import io

print("1. Calling GET http://127.0.0.1:8000/health ...", flush=True)
req = urllib.request.Request("http://127.0.0.1:8000/health")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    print(f"Health Response (status {resp.status}):")
    print(json.dumps(data, indent=2), flush=True)
    assert data["status"] == "healthy"
    assert data["models_ready"] is True

print("\n2. Creating a safe local QR code image ...", flush=True)
test_url = "https://bennett.edu.in/academics/programs"
qr = qrcode.QRCode(box_size=8, border=4)
qr.add_data(test_url)
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")
buf = io.BytesIO()
img.save(buf, format="PNG")
img_bytes = buf.getvalue()

print("\n3. Calling POST http://127.0.0.1:8000/predict via multipart upload ...", flush=True)
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = bytearray()
body.extend(f"--{boundary}\r\n".encode("utf-8"))
body.extend(b'Content-Disposition: form-data; name="file"; filename="test_qr.png"\r\n')
body.extend(b"Content-Type: image/png\r\n\r\n")
body.extend(img_bytes)
body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

predict_req = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=bytes(body),
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)

with urllib.request.urlopen(predict_req) as resp:
    result = json.loads(resp.read().decode())
    print(f"Predict Response (status {resp.status}):")
    print(f"  Decoded URL: {result['decoded_url']}")
    print(f"  URL Probability: {result['url_probability']}")
    print(f"  Image Probability: {result['image_probability']}")
    print(f"  Fusion Probability: {result['fusion_probability']}")
    print(f"  Calibrated Probability: {result['calibrated_probability']}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Risk Level: {result['risk_level']} ({result['risk_label']})")
    print(f"  Grad-CAM Focus: {result['explanation']['gradcam']['focus_region']}")
    print(f"  Safety Notice: {result['safety_notice']}")
    assert result["status"] == "ok"
    assert result["decoded_url"] == test_url

print("\nLIVE FASTAPI VALIDATION PASSED 100%!", flush=True)
