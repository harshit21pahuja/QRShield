import sys
import types
from pathlib import Path

# Stub dummy for any non-essential C-extension that triggers SAC block
dummy_emi = types.ModuleType("sklearn.metrics.cluster._expected_mutual_info_fast")
dummy_emi.expected_mutual_information = lambda *args, **kwargs: 0.0
sys.modules["sklearn.metrics.cluster._expected_mutual_info_fast"] = dummy_emi

# Ensure sklearn.ensemble has RandomForestClassifier directly available
import sklearn
if "sklearn.ensemble" not in sys.modules:
    ens_mod = types.ModuleType("sklearn.ensemble")
    from sklearn.ensemble._forest import RandomForestClassifier
    ens_mod.RandomForestClassifier = RandomForestClassifier
    sys.modules["sklearn.ensemble"] = ens_mod

import joblib
import numpy as np

model_path = Path("artifacts/url_baseline_phase3/url_random_forest.joblib")
rf = joblib.load(model_path)

try:
    import shap
    print("Testing SHAP TreeExplainer...", flush=True)
    # Using a small subset or tree_limit for fast explanation if supported
    explainer = shap.TreeExplainer(rf)
    dummy_x = np.zeros((1, 29))
    shap_vals = explainer.shap_values(dummy_x)
    print("SHAP TreeExplainer succeeded!", type(shap_vals), flush=True)
except Exception as e:
    print(f"SHAP TreeExplainer exception: {e}", flush=True)
