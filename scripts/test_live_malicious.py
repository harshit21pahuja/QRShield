import urllib.request
import json
from pathlib import Path

img_path = Path(r"DATASET\QR_EXTRACTED\malicious\QR_All_Malicious\QR_All_Malicious\qrs\benign_316254.png")
with open(img_path, "rb") as f:
    img_bytes = f.read()

boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = bytearray()
body.extend(f"--{boundary}\r\n".encode("utf-8"))
body.extend(b'Content-Disposition: form-data; name="file"; filename="test_malicious.png"\r\n')
body.extend(b"Content-Type: image/png\r\n\r\n")
body.extend(img_bytes)
body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

req = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=bytes(body),
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)

with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    print("Decoded URL:    ", data["decoded_url"])
    print("Prediction:     ", data["prediction"])
    print("Risk Level:     ", data["risk_level"], f"({data['risk_label']})")
    print("Calibrated Prob:", data["calibrated_probability"])
    print("URL Prob:       ", data["url_probability"])
    print("Image Prob:     ", data["image_probability"])
