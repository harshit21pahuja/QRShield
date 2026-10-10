import sys
import time

def check(name, fn):
    t0 = time.time()
    print(f"Checking {name}...", end="", flush=True)
    try:
        fn()
        dt = time.time() - t0
        print(f" OK ({dt:.2f}s)", flush=True)
    except Exception as e:
        print(f" FAILED: {e}", flush=True)

print(f"Python: {sys.version}", flush=True)

check("numpy", lambda: __import__("numpy"))
check("PIL", lambda: __import__("PIL"))
check("cv2", lambda: __import__("cv2"))
check("torch", lambda: __import__("torch"))
check("torchvision", lambda: __import__("torchvision"))
check("joblib", lambda: __import__("joblib"))
check("sklearn", lambda: __import__("sklearn"))
check("pandas", lambda: __import__("pandas"))
check("fastapi", lambda: __import__("fastapi"))
check("uvicorn", lambda: __import__("uvicorn"))
check("qrcode", lambda: __import__("qrcode"))
print("ALL CHECKS COMPLETE", flush=True)
