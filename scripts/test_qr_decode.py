import sys
from pathlib import Path
import cv2
import numpy as np
import qrcode
from PIL import Image

def test_qr():
    # 1. Generate a synthetic inert QR image
    test_url = "https://example.com/safe/page?id=123"
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(test_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert PIL Image to BGR numpy array for OpenCV
    img_np = np.array(img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 2. Decode using cv2.QRCodeDetector
    detector = cv2.QRCodeDetector()
    decoded_text, points, straight_qrcode = detector.detectAndDecode(img_bgr)
    
    print(f"Original text: {test_url}", flush=True)
    print(f"Decoded text:  {decoded_text}", flush=True)
    assert decoded_text == test_url, f"Expected {test_url} but got {decoded_text}"
    print("QR Code Detector PASSED!", flush=True)

if __name__ == "__main__":
    test_qr()
