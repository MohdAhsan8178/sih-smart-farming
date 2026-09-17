"""
Dataset bias audit.
Trains a RandomForestClassifier purely on 8 background/perimeter pixels per image
(4 corners + 4 edge midpoints -> 24-dim RGB feature vector).
Evaluates whether background capture artifacts leak diagnostic labels.

Reports TWO key metrics:
1. background_only_accuracy_including_not_crop
2. background_only_accuracy_excluding_not_crop

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

    # Filter out unreadable samples
    train_valid = [f is not None for f in train_feats]
    test_valid = [f is not None for f in test_feats]

    X_train_all = np.array([f for f in train_feats if f is not None])
    y_train_all = train_df.loc[train_valid, 'label'].values

    X_test_all = np.array([f for f in test_feats if f is not None])
    y_test_all = test_df.loc[test_valid, 'label'].values

    # 1. EVALUATION INCLUDING not_crop
    print(f"\n[bias_audit] 1. Training RandomForest INCLUDING not_crop ({len(CLASS_NAMES)} classes)...")
    clf_all = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    clf_all.fit(X_train_all, y_train_all)

    y_pred_all = clf_all.predict(X_test_all)
    bg_acc_all = float(accuracy_score(y_test_all, y_pred_all))
    chance_all = float(1.0 / NUM_CLASSES)

    report_all = classification_report(y_test_all, y_pred_all, output_dict=True, zero_division=0)
    leaked_all = []
    for cls_name, metrics in report_all.items():
        if cls_name in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        if isinstance(metrics, dict) and 'recall' in metrics:
            if metrics['recall'] > max(0.20, 3 * chance_all):
                leaked_all.append(f"{cls_name} (recall={metrics['recall']:.2f})")

    # 2. EVALUATION EXCLUDING not_crop
    # (Trained and evaluated strictly on crop diagnostic classes)
    print(f"\n[bias_audit] 2. Training RandomForest EXCLUDING not_crop (28 crop classes)...")
    crop_train_mask = (y_train_all != 'not_crop')
    crop_test_mask = (y_test_all != 'not_crop')

    X_train_crop = X_train_all[crop_train_mask]
    y_train_crop = y_train_all[crop_train_mask]

    X_test_crop = X_test_all[crop_test_mask]
    y_test_crop = y_test_all[crop_test_mask]

    clf_crop = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    clf_crop.fit(X_train_crop, y_train_crop)

    y_pred_crop = clf_crop.predict(X_test_crop)
    bg_acc_crop = float(accuracy_score(y_test_crop, y_pred_crop))
    chance_crop = float(1.0 / (NUM_CLASSES - 1))

    report_crop = classification_report(y_test_crop, y_pred_crop, output_dict=True, zero_division=0)
    leaked_crop = []
    for cls_name, metrics in report_crop.items():
        if cls_name in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        if isinstance(metrics, dict) and 'recall' in metrics:
            if metrics['recall'] > max(0.20, 3 * chance_crop):
                leaked_crop.append(f"{cls_name} (recall={metrics['recall']:.2f})")

    audit_results = {
        "background_only_accuracy_including_not_crop": round(bg_acc_all, 6),
        "chance_accuracy_including_not_crop": round(chance_all, 6),
        "leaked_classes_including_not_crop": leaked_all,
        
        "background_only_accuracy_excluding_not_crop": round(bg_acc_crop, 6),
        "chance_accuracy_excluding_not_crop": round(chance_crop, 6),
        "leaked_classes_excluding_not_crop": leaked_crop,

        "not_crop_recall_in_full_audit": round(float(report_all.get('not_crop', {}).get('recall', 0.0)), 4)
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS / 'bias_audit.json'
    with open(out_file, 'w') as f:
        json.dump(audit_results, f, indent=2)

    print(f"\n[bias_audit] Saved results to {out_file}")
    print("=" * 75)
    print(f"INCLUDING not_crop ({len(CLASS_NAMES)} classes):")
    print(f"  Background-only Accuracy : {bg_acc_all * 100:.2f}% (Chance: {chance_all * 100:.2f}%)")
    print(f"  not_crop Recall          : {report_all.get('not_crop', {}).get('recall', 0.0)*100:.1f}%")
    print(f"  Flagged Leaked Classes   : {len(leaked_all)}")
    print("-" * 75)
    print(f"EXCLUDING not_crop ({len(CLASS_NAMES) - 1} crop classes):")
    print(f"  Background-only Accuracy : {bg_acc_crop * 100:.2f}% (Chance: {chance_crop * 100:.2f}%)")
    print(f"  Flagged Leaked Classes   : {len(leaked_crop)}")
    print("=" * 75)

if __name__ == '__main__':
    run_bias_audit()
