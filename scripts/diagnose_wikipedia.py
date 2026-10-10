import sys
from pathlib import Path
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import url_engine, image_engine, fusion_engine
from src.features.url_features import extract_url_features

print("=== 1. DIAGNOSING URL MODEL ON WIKIPEDIA ===")
wiki_url = "https://en.wikipedia.org/wiki/QR_code"
features = extract_url_features(wiki_url)
url_pred = url_engine.predict(wiki_url)
print(f"URL: {wiki_url}")
print(f"URL Probability: {url_pred['url_probability']:.4f}")
print("Features of interest:")
for k, v in features.items():
    if v > 0:
        print(f"   {k}: {v}")

print("\n=== 2. TESTING ACTUAL BENIGN DATASET IMAGES ON MOBILENETV2 ===")
dev_test_csv = Path("data/processed/image_dev/test.csv")
if dev_test_csv.exists():
    df_test = pd.read_csv(dev_test_csv)
    benign_samples = df_test[df_test["label"] == 0].head(5)
    malicious_samples = df_test[df_test["label"] == 1].head(5)

    print(f"Found {len(benign_samples)} test benign samples:")
    for idx, row in benign_samples.iterrows():
        img_path = Path(row["image_path"])
        if img_path.exists():
            img_pred = image_engine.predict(img_path, generate_gradcam=False)
            url_p = url_engine.predict(row["url"])
            print(f"  [BENIGN TRUE] URL: {row['url'][:40]} | URL Prob: {url_p['url_probability']:.4f} | Image Prob: {img_pred['image_probability']:.4f}")

    print(f"\nFound {len(malicious_samples)} test malicious samples:")
    for idx, row in malicious_samples.iterrows():
        img_path = Path(row["image_path"])
        if img_path.exists():
            img_pred = image_engine.predict(img_path, generate_gradcam=False)
            url_p = url_engine.predict(row["url"])
            print(f"  [MALICIOUS TRUE] URL: {row['url'][:40]} | URL Prob: {url_p['url_probability']:.4f} | Image Prob: {img_pred['image_probability']:.4f}")

print("\n=== 3. TESTING SYNTHETIC QR CODE ON MOBILENETV2 ===")
sample_safe_qr = Path("frontend/public/samples/sample_safe_qr.png")
if sample_safe_qr.exists():
    sample_pred = image_engine.predict(sample_safe_qr, generate_gradcam=False)
    print(f"Synthetic sample_safe_qr.png Image Prob: {sample_pred['image_probability']:.4f}")
