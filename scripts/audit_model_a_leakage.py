#!/usr/bin/env python3
"""
scripts/audit_model_a_leakage.py

Rigorous Model A Audit (Parts E2, E3, E4, E5).
E2: 4-orientation pHash dedup across train, val, and test_indist.
    Identifies cross-split leakage, outputs CSV and 30-pair contact sheet.
E3: Re-evaluates Model A ONNX on clean test_indist (excluding train duplicates).
E4: Re-evaluates Model A ONNX on clean val (excluding train duplicates),
    refits T_cal and tau_energy on clean val.
E5: Generates per-class recall table for test_sourceheldout.
"""
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import imagehash
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.classes import CLASS_NAMES, NUM_CLASSES
from core.rejection import stable_logsumexp

AUDIT_DIR = ROOT / "artifacts" / "audit" / "2026-09-16"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

LEAKAGE_CSV = AUDIT_DIR / "model_a_cross_split_leakage.csv"
CONTACT_SHEET_PNG = AUDIT_DIR / "model_a_leakage_pairs.png"
ONNX_MODEL_A = ROOT / "artifacts" / "onnx" / "model_a_fused.onnx"

def hash_worker(rel_path):
    abs_p = ROOT / rel_path
    if not abs_p.exists():
        return None
    try:
        with Image.open(abs_p) as raw_img:
            img = raw_img.convert("RGB")
            # 4 orientations: 0, 90, 180, 270
            h0 = int(str(imagehash.phash(img, hash_size=8)), 16)
            h90 = int(str(imagehash.phash(img.rotate(90), hash_size=8)), 16)
            h180 = int(str(imagehash.phash(img.rotate(180), hash_size=8)), 16)
            h270 = int(str(imagehash.phash(img.rotate(270), hash_size=8)), 16)
            return (h0, h90, h180, h270)
    except Exception:
        return None

def compute_hashes_for_df(df, name):
    print(f"Computing 4-orientation pHash for {name} ({len(df)} images)...")
    t0 = time.time()
    paths = df["path"].tolist()
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(hash_worker, paths, chunksize=250))
    dt = time.time() - t0
    print(f"Done in {dt:.1f}s ({dt/len(df)*1000:.1f}ms per image)")
    return results

def find_leakage(eval_df, eval_hashes, split_name, train_df, train_uint64_arr, train_meta, threshold=5):
    print(f"\nSearching for train duplicates in {split_name} (threshold <= {threshold})...")
    leaked_records = []
    chunk_size = 500
    n_eval = len(eval_df)
    
    for start_idx in range(0, n_eval, chunk_size):
        end_idx = min(start_idx + chunk_size, n_eval)
        chunk_queries = []
        for i in range(start_idx, end_idx):
            h_tuple = eval_hashes[i]
            if h_tuple is not None:
                for h_val in h_tuple:
                    chunk_queries.append((i, h_val))
        
        if not chunk_queries:
            continue
            
        q_indices, q_hashes = zip(*chunk_queries)
        q_arr = np.array(q_hashes, dtype=np.uint64)[:, None]
        
        t_slice_size = 25000
        for t_start in range(0, len(train_uint64_arr), t_slice_size):
            t_end = min(t_start + t_slice_size, len(train_uint64_arr))
            sub_train = train_uint64_arr[None, t_start:t_end]
            diff_bits = np.bitwise_count(q_arr ^ sub_train)
            match_mask = (diff_bits <= threshold)
            
            if np.any(match_mask):
                q_hit_rows, t_hit_cols = np.where(match_mask)
                for q_hit, t_hit in zip(q_hit_rows, t_hit_cols):
                    eval_idx = q_indices[q_hit]
                    train_meta_idx = t_start + t_hit
                    train_idx, train_orient = train_meta[train_meta_idx]
                    dist = int(diff_bits[q_hit, t_hit])
                    
                    e_row = eval_df.iloc[eval_idx]
                    tr_row = train_df.iloc[train_idx]
                    
                    leaked_records.append({
                        "eval_split": split_name,
                        "eval_idx": eval_idx,
                        "eval_path": e_row["path"],
                        "eval_label": e_row["label"],
                        "eval_group_id": e_row["group_id"],
                        "train_idx": train_idx,
                        "train_path": tr_row["path"],
                        "train_label": tr_row["label"],
                        "train_group_id": tr_row["group_id"],
                        "hamming_dist": dist,
                        "same_label": (e_row["label"] == tr_row["label"]),
                        "same_group_id": (e_row["group_id"] == tr_row["group_id"]),
                    })

    if leaked_records:
        df_leak = pd.DataFrame(leaked_records)
        df_leak = df_leak.sort_values("hamming_dist").drop_duplicates(subset=["eval_path", "train_path"]).reset_index(drop=True)
        return df_leak
    else:
        return pd.DataFrame()

def evaluate_model_a(eval_df, session, input_name, class_to_idx, desc):
    preds, targets = [], []
    logits_list = []
    
    for i, row in eval_df.iterrows():
        p = ROOT / row["path"]
        if not p.exists():
            continue
        try:
            with Image.open(p) as raw_img:
                img = raw_img.convert("RGB")
                w, h = img.size
                min_dim = min(w, h)
                left = (w - min_dim) // 2
                top = (h - min_dim) // 2
                cropped = img.crop((left, top, left + min_dim, top + min_dim)).resize((224, 224), Image.BILINEAR)
                
                arr = np.array(cropped, dtype=np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                chw = ((arr - mean) / std).transpose(2, 0, 1)[None, :]
                
                out = session.run(None, {input_name: chw})[0][0]
                logits_list.append(out)
                preds.append(int(np.argmax(out)))
                targets.append(class_to_idx[row["label"]])
        except Exception:
            continue
            
    preds = np.array(preds)
    targets = np.array(targets)
    logits_arr = np.array(logits_list)
    
    top1 = float((preds == targets).mean())
    unique_classes = np.unique(targets)
    f1_list = []
    per_class_recalls = {}
    for c in unique_classes:
        tp = np.sum((preds == c) & (targets == c))
        fp = np.sum((preds == c) & (targets != c))
        fn = np.sum((preds != c) & (targets == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        per_class_recalls[CLASS_NAMES[c]] = (rec, int(np.sum(targets == c)))
        
    macro_f1 = float(np.mean(f1_list))
    return top1, macro_f1, len(targets), logits_arr, targets, per_class_recalls

def main():
    print("=================================================================")
    print("PART E: MODEL A AUDIT - LEAKAGE QUANTIFICATION & EVALUATION")
    print("=================================================================")

    train_df = pd.read_csv(ROOT / "splits_v3" / "train.csv")
    val_df = pd.read_csv(ROOT / "splits_v3" / "val.csv")
    test_indist_df = pd.read_csv(ROOT / "splits_v3" / "test_indist.csv")
    test_heldout_df = pd.read_csv(ROOT / "splits_v3" / "test_sourceheldout.csv")

    print(f"Loaded splits_v3:")
    print(f"  train             : {len(train_df)} rows")
    print(f"  val               : {len(val_df)} rows")
    print(f"  test_indist       : {len(test_indist_df)} rows")
    print(f"  test_sourceheldout: {len(test_heldout_df)} rows")

    train_hashes = compute_hashes_for_df(train_df, "train")
    val_hashes = compute_hashes_for_df(val_df, "val")
    test_hashes = compute_hashes_for_df(test_indist_df, "test_indist")

    train_uint64_list = []
    train_meta = []
    for idx, h_tuple in enumerate(train_hashes):
        if h_tuple is not None:
            for orient, h_val in enumerate(h_tuple):
                train_uint64_list.append(h_val)
                train_meta.append((idx, orient))

    train_uint64_arr = np.array(train_uint64_list, dtype=np.uint64)
    print(f"Total valid train hash keys (4x orientations): {len(train_uint64_arr):,}")

    leak_test_df = find_leakage(test_indist_df, test_hashes, "test_indist", train_df, train_uint64_arr, train_meta, threshold=5)
    leak_val_df = find_leakage(val_df, val_hashes, "val", train_df, train_uint64_arr, train_meta, threshold=5)

    all_leak_df = pd.concat([leak_test_df, leak_val_df], ignore_index=True)
    all_leak_df.to_csv(LEAKAGE_CSV, index=False)
    print(f"\nSaved full leakage report to {LEAKAGE_CSV}")

    # E2 Summary Table
    print("\n--- E2. LEAKAGE QUANTIFICATION SUMMARY (Hamming <= 5, 4 orientations) ---")
    print(f"test_indist Total Rows       : {len(test_indist_df)}")
    test_leaked_unique = leak_test_df["eval_path"].nunique() if not leak_test_df.empty else 0
    print(f"test_indist Leaked Images    : {test_leaked_unique} / {len(test_indist_df)} ({test_leaked_unique/len(test_indist_df)*100:.2f}%)")
    if not leak_test_df.empty:
        same_lbl = (leak_test_df["eval_label"] == leak_test_df["train_label"]).sum()
        diff_lbl = (leak_test_df["eval_label"] != leak_test_df["train_label"]).sum()
        print(f"  - Leaked Pairs with SAME label     : {same_lbl}")
        print(f"  - Leaked Pairs with DIFFERENT label: {diff_lbl}")
        print(f"  - Distance distribution:")
        for d, count in leak_test_df["hamming_dist"].value_counts().sort_index().items():
            print(f"      dist = {d}: {count} pairs")

    print(f"\nval Total Rows               : {len(val_df)}")
    val_leaked_unique = leak_val_df["eval_path"].nunique() if not leak_val_df.empty else 0
    print(f"val Leaked Images            : {val_leaked_unique} / {len(val_df)} ({val_leaked_unique/len(val_df)*100:.2f}%)")
    if not leak_val_df.empty:
        same_lbl_v = (leak_val_df["eval_label"] == leak_val_df["train_label"]).sum()
        diff_lbl_v = (leak_val_df["eval_label"] != leak_val_df["train_label"]).sum()
        print(f"  - Leaked Pairs with SAME label     : {same_lbl_v}")
        print(f"  - Leaked Pairs with DIFFERENT label: {diff_lbl_v}")
        print(f"  - Distance distribution:")
        for d, count in leak_val_df["hamming_dist"].value_counts().sort_index().items():
            print(f"      dist = {d}: {count} pairs")

    # Contact sheet of 30 pairs
    print("\nGenerating 30-pair leakage contact sheet...")
    if not all_leak_df.empty:
        sample_pairs = all_leak_df.sample(n=min(30, len(all_leak_df)), random_state=42).reset_index(drop=True)
        pair_w, pair_h = 450, 150
        cols = 2
        rows = (len(sample_pairs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * pair_w, rows * pair_h), color=(30, 30, 30))
        draw = ImageDraw.Draw(sheet)
        
        for idx, row in sample_pairs.iterrows():
            c_idx = idx % cols
            r_idx = idx // cols
            x_off = c_idx * pair_w
            y_off = r_idx * pair_h
            
            try:
                im_eval = Image.open(ROOT / row["eval_path"]).convert("RGB").resize((120, 120))
                im_train = Image.open(ROOT / row["train_path"]).convert("RGB").resize((120, 120))
                
                sheet.paste(im_eval, (x_off + 10, y_off + 15))
                sheet.paste(im_train, (x_off + 140, y_off + 15))
                
                lbl_e = row['eval_label'].split('__')[-1][:14]
                lbl_t = row['train_label'].split('__')[-1][:14]
                dist = row['hamming_dist']
                split = row['eval_split'][:4]
                
                text1 = f"[{split}] {lbl_e} (d={dist})"
                text2 = f"[train] {lbl_t}"
                same_tag = "MATCH" if row["same_label"] else "CONFLICT"
                color = (100, 255, 100) if row["same_label"] else (255, 100, 100)
                
                draw.text((x_off + 270, y_off + 30), text1, fill=(220, 220, 220))
                draw.text((x_off + 270, y_off + 55), text2, fill=(220, 220, 220))
                draw.text((x_off + 270, y_off + 80), f"Status: {same_tag}", fill=color)
            except Exception as e:
                draw.text((x_off + 20, y_off + 50), f"Error: {e}", fill=(255, 50, 50))
                
        sheet.save(str(CONTACT_SHEET_PNG))
        print(f"Saved contact sheet to {CONTACT_SHEET_PNG}")

    # E3: Re-evaluate Model A ONNX on Cleaned test_indist
    print("\n=================================================================")
    print("PART E3: RE-EVALUATE MODEL A ON CLEAN test_indist")
    print("=================================================================")
    leaked_test_paths = set(leak_test_df["eval_path"].unique()) if not leak_test_df.empty else set()
    clean_test_df = test_indist_df[~test_indist_df["path"].isin(leaked_test_paths)].reset_index(drop=True)

    session = ort.InferenceSession(str(ONNX_MODEL_A), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    top1_orig, f1_orig, n_orig, _, _, _ = evaluate_model_a(test_indist_df, session, input_name, class_to_idx, "Original test_indist")
    top1_clean, f1_clean, n_clean, _, _, _ = evaluate_model_a(clean_test_df, session, input_name, class_to_idx, "Cleaned test_indist")

    print(f"Model A Evaluation on test_indist:")
    print(f"  - Original test_indist (N={n_orig}): Top-1 = {top1_orig*100:.2f}%, Macro-F1 = {f1_orig*100:.2f}%")
    print(f"  - Cleaned test_indist  (N={n_clean}): Top-1 = {top1_clean*100:.2f}%, Macro-F1 = {f1_clean*100:.2f}%")
    print(f"  - Delta (|Clean - Orig|)         : Top-1 Delta = {(top1_clean-top1_orig)*100:+.2f} pts, Macro-F1 Delta = {(f1_clean-f1_orig)*100:+.2f} pts")

    # E4: Check Val Leakage, Refit T_cal and tau_energy on Clean Val
    print("\n=================================================================")
    print("PART E4: CHECK VAL SET LEAKAGE & REFIT T_CAL / TAU_ENERGY")
    print("=================================================================")
    leaked_val_paths = set(leak_val_df["eval_path"].unique()) if not leak_val_df.empty else set()
    clean_val_df = val_df[~val_df["path"].isin(leaked_val_paths)].reset_index(drop=True)

    top1_v_orig, f1_v_orig, n_v_orig, l_v_orig, y_v_orig, _ = evaluate_model_a(val_df, session, input_name, class_to_idx, "Original val")
    top1_v_clean, f1_v_clean, n_v_clean, l_v_clean, y_v_clean, _ = evaluate_model_a(clean_val_df, session, input_name, class_to_idx, "Cleaned val")

    print(f"Model A Performance on Validation:")
    print(f"  - Original val (N={n_v_orig}): Top-1 = {top1_v_orig*100:.2f}%, Macro-F1 = {f1_v_orig*100:.2f}%")
    print(f"  - Cleaned val  (N={n_v_clean}): Top-1 = {top1_v_clean*100:.2f}%, Macro-F1 = {f1_v_clean*100:.2f}%")

    logits_clean_t = torch.tensor(l_v_clean, dtype=torch.float32)
    targets_clean_t = torch.tensor(y_v_clean, dtype=torch.long)

    temp = nn.Parameter(torch.ones(1) * 0.597)
    opt = torch.optim.LBFGS([temp], lr=0.01, max_iter=50)
    crit = nn.CrossEntropyLoss()

    def eval_t():
        opt.zero_grad()
        loss = crit(logits_clean_t / temp, targets_clean_t)
        loss.backward()
        return loss

    opt.step(eval_t)
    refit_t_clean = float(temp.item())
    nll_before = float(crit(logits_clean_t / 0.597, targets_clean_t).item())
    nll_after = float(crit(logits_clean_t / refit_t_clean, targets_clean_t).item())

    crop_cols = [i for i in range(NUM_CLASSES) if CLASS_NAMES[i] != "not_crop"]
    z_clean = l_v_clean[:, crop_cols] / 1.0
    m_clean = z_clean.max(axis=1, keepdims=True)
    e_clean = -1.0 * (m_clean.squeeze(1) + np.log(np.exp(z_clean - m_clean).sum(axis=1)))
    refit_tau_energy = float(np.percentile(e_clean, 95.0))

    print(f"\nRefitted Calibration on Cleaned Validation Set:")
    print(f"  - Original T_CAL          : 0.5970 (NLL on clean val: {nll_before:.4f})")
    print(f"  - Refitted T_CAL          : {refit_t_clean:.4f} (NLL on clean val: {nll_after:.4f})")
    print(f"  - Original TAU_ENERGY     : -2.8529")
    print(f"  - Refitted TAU_ENERGY     : {refit_tau_energy:.4f}")

    # E5: Per-Class Recall Table for test_sourceheldout
    print("\n=================================================================")
    print("PART E5: PER-CLASS RECALL TABLE FOR test_sourceheldout")
    print("=================================================================")
    top1_sh, f1_sh, n_sh, _, _, recalls_sh = evaluate_model_a(test_heldout_df, session, input_name, class_to_idx, "test_sourceheldout")

    print(f"test_sourceheldout Overall Performance:")
    print(f"  Total Samples Evaluated : {n_sh}")
    print(f"  Top-1 Accuracy          : {top1_sh*100:.2f}%")
    print(f"  Macro-F1                : {f1_sh*100:.2f}%")

    print(f"\n{'Class Name':<35} | {'Support':<8} | {'Recall (%)':<12} | {'Status'}")
    print("-" * 70)
    for cls_name, (rec, supp) in sorted(recalls_sh.items(), key=lambda x: x[0]):
        status = "COLLAPSED (<10%)" if rec < 0.10 else ("DEGRADED (<50%)" if rec < 0.50 else "ROBUST")
        print(f"{cls_name:<35} | {supp:<8} | {rec*100:<12.2f} | {status}")

if __name__ == "__main__":
    main()
