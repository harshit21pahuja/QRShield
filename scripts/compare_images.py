import sys
from pathlib import Path
from PIL import Image
import numpy as np

# 1. Dataset benign image
dataset_benign_path = Path(r"C:\Users\Harshit Pahuja\Desktop\New folder\QRShield\DATASET\QR_EXTRACTED\malicious\QR_All_Malicious\QR_All_Malicious\qrs\benign_421379.png")
if not dataset_benign_path.exists():
    # Find any image in DATASET
    import glob
    candidates = list(Path("DATASET").rglob("*.png"))[:5]
    print("Candidates:", candidates)
else:
    im_benign = Image.open(dataset_benign_path)
    print(f"Dataset Image: size={im_benign.size}, mode={im_benign.mode}, format={im_benign.format}")
    arr_b = np.array(im_benign)
    print(f"  min={arr_b.min()}, max={arr_b.max()}, mean={arr_b.mean()}")

# 2. Synthetic QR
sample_path = Path("frontend/public/samples/sample_safe_qr.png")
im_sample = Image.open(sample_path)
print(f"Synthetic Image: size={im_sample.size}, mode={im_sample.mode}, format={im_sample.format}")
arr_s = np.array(im_sample)
print(f"  min={arr_s.min()}, max={arr_s.max()}, mean={arr_s.mean()}")
