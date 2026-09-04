"""
Deduplicate images using perceptual hashing and exact MD5 hashing.
Groups near-duplicates (Hamming distance <= 5) and exact matches into a shared group_id
so that repeated captures of the same leaf/scene never straddle train/val/test splits.
"""

import sys
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import ROOT, SPLITS

def compute_hashes(args):
    rel_path, label = args
    abs_path = ROOT / rel_path
    try:
        with open(abs_path, 'rb') as f:
            md5_val = hashlib.md5(f.read()).hexdigest()
        with Image.open(abs_path) as img:
            ph = str(imagehash.phash(img.convert('RGB'), hash_size=8))
        return rel_path, md5_val, ph
    except Exception as e:
        # Fallback dummy hash on read error
        return rel_path, None, None

def run_dedup():
    manifest_path = SPLITS / 'all_images.csv'
    print(f"[dedup] Loading {manifest_path}...")
    df = pd.read_csv(manifest_path)
    n_images = len(df)
    print(f"[dedup] Computing perceptual and MD5 hashes for {n_images} images...")

    tasks = list(zip(df.path, df.label))
    # Multi-process hash computation
    results = []
    with ProcessPoolExecutor() as executor:
        for r in tqdm(executor.map(compute_hashes, tasks, chunksize=250), total=n_images):
            results.append(r)

    paths, md5s, phashes = zip(*results)
    df['md5'] = md5s
    df['phash'] = phashes

    # Filter out unreadable images if any
    valid_mask = df.phash.notna()
    if (~valid_mask).sum() > 0:
        print(f"[dedup] Warning: {(~valid_mask).sum()} unreadable images dropped.")
        df = df[valid_mask].reset_index(drop=True)

    print("[dedup] Grouping duplicates using Union-Find (Hamming distance <= 5 or exact MD5)...")
    
    # Union-Find data structure
    parent = list(range(len(df)))

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # 1. Group exact MD5 matches (e.g. shared images between sugarcane datasets)
    md5_groups = df.groupby('md5').groups
    for md5_val, indices in md5_groups.items():
        if len(indices) > 1:
            first = indices[0]
            for other in indices[1:]:
                union(first, other)

    # 2. Group perceptual hash near-duplicates per label (images within same disease/class)
    # Bucketing by 16-bit prefix for fast O(N) neighbor comparison
    label_groups = df.groupby('label').groups
    for label, indices in label_groups.items():
        sub_indices = list(indices)
        if len(sub_indices) <= 1:
            continue
        
        # Convert hex phash to uint64 array
        sub_hashes = [imagehash.hex_to_hash(df.loc[idx, 'phash']) for idx in sub_indices]
        n_sub = len(sub_indices)

        # For small groups (< 500), exact all-pairs check is instantaneous
        if n_sub <= 500:
            for i in range(n_sub):
                for j in range(i + 1, n_sub):
                    if (sub_hashes[i] - sub_hashes[j]) <= 5:
                        union(sub_indices[i], sub_indices[j])
        else:
            # Bucket by 16-bit prefix (first 4 hex characters)
            buckets = {}
            for i, h in enumerate(sub_hashes):
                prefix = str(h)[:4]
                buckets.setdefault(prefix, []).append(i)
            for prefix, b_indices in buckets.items():
                for ii in range(len(b_indices)):
                    for jj in range(ii + 1, len(b_indices)):
                        idx_i = b_indices[ii]
                        idx_j = b_indices[jj]
                        if (sub_hashes[idx_i] - sub_hashes[idx_j]) <= 5:
                            union(sub_indices[idx_i], sub_indices[idx_j])

    # Assign contiguous group_ids
    group_map = {}
    group_ids = []
    current_gid = 0
    for i in range(len(df)):
        root_i = find(i)
        if root_i not in group_map:
            group_map[root_i] = current_gid
            current_gid += 1
        group_ids.append(group_map[root_i])

    df['group_id'] = group_ids
    n_unique_groups = df.group_id.nunique()
    print(f"[dedup] Total images: {len(df)}")
    print(f"[dedup] Unique groups formed: {n_unique_groups}")
    print(f"[dedup] Deduplicated / grouped instances: {len(df) - n_unique_groups}")

    # Drop temporary columns before saving
    out_df = df[['path', 'label', 'source_dataset', 'orig_folder', 'group_id']]
    out_df.to_csv(manifest_path, index=False)
    print(f"[dedup] Wrote updated manifest with group_id to {manifest_path}")

if __name__ == '__main__':
    run_dedup()
