#!/usr/bin/env python3
"""
scripts/package_splits_v2.py

Packages splits_v2 into:
1. data/packaged_min/splits_v2/
2. splits_v2_packaged/

Rewrites paths from data/raw/ to data/packaged_min/ using rename_map.csv.
Enforces zero non-compliant characters and verifies all paths exist.
"""

import os
import re
import shutil
import pandas as pd

ROOT_SPLITS_V2 = 'splits_v2'
DST_PACKAGED_SPLITS_V2 = 'data/packaged_min/splits_v2'
DST_LOCAL_PACKAGED = 'splits_v2_packaged'
RENAME_MAP_PATH = 'data/packaged_min/splits_packaged/rename_map.csv'

os.makedirs(DST_PACKAGED_SPLITS_V2, exist_ok=True)
os.makedirs(DST_LOCAL_PACKAGED, exist_ok=True)

rename_df = pd.read_csv(RENAME_MAP_PATH)
file_map = dict(zip(rename_df['old_relative_path'], rename_df['new_relative_path']))
print(f'Loaded rename map: {len(file_map)} entries')

csv_files = [
    'all_images.csv',
    'train.csv',
    'val.csv',
    'test_indist.csv',
    'test_crossdomain.csv',
    'test_external_riceblast.csv'
]

def map_path(p):
    prefix_raw = 'data/raw/'
    if p.startswith(prefix_raw):
        rel = p[len(prefix_raw):]
        sanitized_rel = file_map.get(rel, rel)
        return 'data/packaged_min/' + sanitized_rel
    return p

total_paths = 0
missing_paths = 0
non_compliant_chars = set()

for f in csv_files:
    src_csv = os.path.join(ROOT_SPLITS_V2, f)
    df = pd.read_csv(src_csv)
    df['path'] = df['path'].apply(map_path)
    total_paths += len(df)
    
    for p in df['path']:
        if not os.path.exists(p):
            missing_paths += 1
        for c in p:
            if not re.match(r'[A-Za-z0-9._/-]', c):
                non_compliant_chars.add(c)
                
    df.to_csv(os.path.join(DST_PACKAGED_SPLITS_V2, f), index=False)
    df.to_csv(os.path.join(DST_LOCAL_PACKAGED, f), index=False)
    print(f'Wrote {f}: {len(df)} rows')

# Copy metadata files
meta_files = ['class_weights.json', 'class_mapping.csv', 'openset_categories.csv']
for mf in meta_files:
    src = os.path.join(ROOT_SPLITS_V2, mf)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(DST_PACKAGED_SPLITS_V2, mf))
        shutil.copy2(src, os.path.join(DST_LOCAL_PACKAGED, mf))

# Also copy rename_map.csv into splits_v2
shutil.copy2(RENAME_MAP_PATH, os.path.join(DST_PACKAGED_SPLITS_V2, 'rename_map.csv'))
shutil.copy2(RENAME_MAP_PATH, os.path.join(DST_LOCAL_PACKAGED, 'rename_map.csv'))

print(f"\nVerification:")
print(f'  Total paths verified across all 6 CSVs: {total_paths}')
print(f'  Missing paths on disk: {missing_paths}')
print(f'  Non-compliant characters: {non_compliant_chars}')

assert missing_paths == 0, f'Missing paths: {missing_paths}'
assert len(non_compliant_chars) == 0, f'Non-compliant chars: {non_compliant_chars}'

# Count images and total files in data/packaged_min
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
actual_imgs = 0
actual_total = 0
for dp, dns, fns in os.walk('data/packaged_min'):
    for f in fns:
        actual_total += 1
        ext = os.path.splitext(f)[1].lower()
        if ext in IMG_EXTS:
            actual_imgs += 1

print(f"\ndata/packaged_min Audit:")
print(f'  Actual Image Count: {actual_imgs} (Expected: 35551)')
print(f'  Actual Total Files Count: {actual_total} (Expected: 35573)')

assert actual_imgs == 35551, f'Image count mismatch: {actual_imgs} != 35551'
assert actual_total == 35573, f'Total files mismatch: {actual_total} != 35573'

print("\nAll assertions PASSED!")
