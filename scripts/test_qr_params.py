import sys
from pathlib import Path
import qrcode
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import image_engine, url_engine, fusion_engine

url = "https://en.wikipedia.org/wiki/QR_code"
url_res = url_engine.predict(url)
print(f"URL Random Forest probability for Wikipedia: {url_res['url_probability']:.4f}")

for bs in [2, 4, 6, 8, 10]:
    for border in [1, 2, 4, 6]:
        qr = qrcode.QRCode(box_size=bs, border=border)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").get_image()
        res = image_engine.predict(img, generate_gradcam=False)
        img_p = res["image_probability"]
        fusion = fusion_engine.compute_fusion(url_res["url_probability"], img_p)
        print(f"box_size={bs:2d}, border={border:2d}, size={img.size} -> Image Prob: {img_p:.4f} | Fusion Prob: {fusion['fusion_probability']:.4f} | Calibrated: {fusion['calibrated_probability']:.4f} -> {fusion['prediction']}")
