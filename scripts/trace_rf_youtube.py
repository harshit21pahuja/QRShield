import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.url_features import extract_url_features, FEATURE_NAMES
from src.inference import url_engine

url = "https://www.youtube.com"
feat_dict = extract_url_features(url)
x = np.array([[feat_dict[k] for k in FEATURE_NAMES]], dtype=np.float32)

rf = url_engine.model
trees = rf.estimators_

# Check how many trees vote 1 vs 0
votes_1 = 0
votes_0 = 0
for t in trees:
    p = t.predict_proba(x)[0]
    if p[1] > 0.5:
        votes_1 += 1
    else:
        votes_0 += 1

print(f"Random Forest trees voting malicious: {votes_1}/{len(trees)} ({votes_1/len(trees)*100:.1f}%)")

# Let's inspect the decision path in tree 0
t0 = trees[0]
path = t0.decision_path(x).toarray()[0]
node_indices = np.where(path == 1)[0]
print("\nDecision path in Tree 0:")
for node_id in node_indices:
    feat_id = t0.tree_.feature[node_id]
    thresh = t0.tree_.threshold[node_id]
    if feat_id >= 0:
        f_name = FEATURE_NAMES[feat_id]
        val = feat_dict[f_name]
        direction = "<=" if val <= thresh else ">"
        print(f"  Node {node_id}: {f_name} ({val}) {direction} {thresh:.4f}")
    else:
        values = t0.tree_.value[node_id][0]
        prob1 = values[1] / sum(values) if sum(values) > 0 else 0
        print(f"  Leaf {node_id}: value={values}, prob_malicious={prob1:.4f}")

# Check test dataset benign URLs
dev_test_csv = Path("data/processed/image_dev/test.csv")
if dev_test_csv.exists():
    df = pd.read_csv(dev_test_csv)
    benign_df = df[df["label"] == 0]
    print(f"\nEvaluating 10 benign test URLs from dataset:")
    for idx, r in benign_df.head(10).iterrows():
        p = url_engine.predict(r["url"])["url_probability"]
        print(f"  Label {r['label']}: {r['url'][:45]} -> URL Prob: {p:.4f}")
