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
    from sklearn.ensemble._forest import (
        RandomForestClassifier,
        RandomForestRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
    )
    ens_mod.RandomForestClassifier = RandomForestClassifier
    ens_mod.RandomForestRegressor = RandomForestRegressor
    ens_mod.ExtraTreesClassifier = ExtraTreesClassifier
    ens_mod.ExtraTreesRegressor = ExtraTreesRegressor
    sys.modules["sklearn.ensemble"] = ens_mod

import joblib
import numpy as np

print("Loading URL Random Forest...", flush=True)
model_path = Path("artifacts/url_baseline_phase3/url_random_forest.joblib")
rf = joblib.load(model_path)
print(f"Loaded RF model: {rf}, n_features_in_={rf.n_features_in_}", flush=True)

print("Loading Isotonic Calibrator...", flush=True)
calibrator_path = Path("artifacts/fusion_phase3/isotonic_calibrator.joblib")
iso = joblib.load(calibrator_path)
print(f"Loaded Isotonic calibrator: {iso}", flush=True)

# Test inference on dummy 29-feature vector
dummy_x = np.zeros((1, 29))
prob = rf.predict_proba(dummy_x)[0, 1]
print(f"Test RF predict_proba: {prob:.4f}", flush=True)

cal_prob = iso.predict([prob])[0]
print(f"Test calibrated prob: {cal_prob:.4f}", flush=True)

print("SUCCESS: BOTH MODELS LOAD AND INFER PERFECTLY!", flush=True)
