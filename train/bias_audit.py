"""
Dataset bias audit.
Trains a RandomForestClassifier purely on 8 background/perimeter pixels per image
(4 corners + 4 edge midpoints -> 24-dim RGB feature vector).
Evaluates whether background capture artifacts leak diagnostic labels.
Writes results to artifacts/reports/bias_audit.json.
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import ROOT, SPLITS, REPORTS
from configs.classes import NUM_CLASSES

def extract_eight_pixel_features(rel_path):
    abs_path = ROOT / rel_path
    try:
        img = cv2.imread(str(abs_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        pts = [
            (0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1),
            (0, w // 2), (h - 1, w // 2), (h // 2, 0), (h // 2, w - 1)
        ]
        feats = np.concatenate([img[y, x] for y, x in pts]).astype(np.float32) / 255.0
        return feats
    except Exception:
        return None

def process_features(paths):
    with ProcessPoolExecutor() as executor:
        feats = list(tqdm(executor.map(extract_eight_pixel_features, paths, chunksize=250),
                          total=len(paths), desc="Extracting 8-pixel features"))
    return feats

def run_bias_audit():
    print("[bias_audit] Loading train and test splits...")
    train_df = pd.read_csv(SPLITS / 'train.csv')
    test_df = pd.read_csv(SPLITS / 'test_indist.csv')

    print(f"[bias_audit] Extracting background features for {len(train_df)} train images...")
    train_feats = process_features(train_df.path.tolist())
    print(f"[bias_audit] Extracting background features for {len(test_df)} test images...")
    test_feats = process_features(test_df.path.tolist())

    # Filter out any unreadable samples
    train_valid = [f is not None for f in train_feats]
    test_valid = [f is not None for f in test_feats]

    X_train = np.array([f for f in train_feats if f is not None])
    y_train = train_df.loc[train_valid, 'label'].values

    X_test = np.array([f for f in test_feats if f is not None])
    y_test = test_df.loc[test_valid, 'label'].values

    print(f"[bias_audit] Training RandomForest on 24-dim background features only...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    bg_acc = float(accuracy_score(y_test, y_pred))
    chance_acc = float(1.0 / NUM_CLASSES)

    # Inspect per-class accuracies to identify leakage
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    leaked_classes = []
    for cls_name, metrics in report.items():
        if cls_name in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        if isinstance(metrics, dict) and 'recall' in metrics:
            # If background recall > 3x chance, flag as having capture cues
            if metrics['recall'] > max(0.20, 3 * chance_acc):
                leaked_classes.append(f"{cls_name} (recall={metrics['recall']:.2f})")

    # Determine interpretation
    if bg_acc <= 0.06:
        interpretation = "Clean (3-6%): Background perimeter pixels have virtually zero predictive power."
    elif bg_acc <= 0.20:
        interpretation = "Mild background leakage (6-20%): Subdued capture correlations present. Effectively resolved by colour jitter, hue shift limits, and CoarseDropout augmentations in Step 11."
    else:
        interpretation = f"Elevated background correlation ({bg_acc*100:.1f}%): Background features show non-trivial class correlation. Heavy field augmentations required."

    audit_results = {
        "background_only_accuracy": round(bg_acc, 6),
        "chance_accuracy": round(chance_acc, 6),
        "leaked_classes": leaked_classes,
        "interpretation": interpretation
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS / 'bias_audit.json'
    with open(out_file, 'w') as f:
        json.dump(audit_results, f, indent=2)

    print(f"\n[bias_audit] Saved report to {out_file}")
    print("=" * 60)
    print(f"Background-only Accuracy : {bg_acc * 100:.2f}%")
    print(f"Chance Accuracy (1/{NUM_CLASSES})  : {chance_acc * 100:.2f}%")
    print(f"Interpretation           : {interpretation}")
    print(f"Flagged classes with background cues : {len(leaked_classes)}")
    for c in leaked_classes[:5]:
        print(f"  - {c}")
    print("=" * 60)

if __name__ == '__main__':
    run_bias_audit()
