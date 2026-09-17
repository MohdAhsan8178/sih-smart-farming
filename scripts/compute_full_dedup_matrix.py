import os, sys, time, json
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

def dhash(img, hash_size=8):
    img = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    diff = [pixels[row * (hash_size + 1) + col] > pixels[row * (hash_size + 1) + col + 1]
            for row in range(hash_size) for col in range(hash_size)]
    val = 0
    for b in diff:
        val = (val << 1) | b
    return val

def hash_file(p):
    try:
        with Image.open(p) as im:
            return str(p), dhash(im)
    except Exception:
        return str(p), None

def hash_directory(file_list, name, max_workers=16):
    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(hash_file, f): f for f in file_list}
        for fut in as_completed(futs):
            p, h = fut.result()
            if h is not None:
                results[p] = h
    print(f"Hashed {len(results)}/{len(file_list)} images for {name} in {time.time() - t0:.2f}s")
    return results

print("=== 1. COLLECTING FILES ===")

# Existing sources
existing_dirs = {
    'paddy_doctor': Path('data/raw/paddy_doctor'),
    'rice_sethy': Path('data/raw/rice_sethy'),
    'rice_hasan': Path('data/raw/rice_hasan'),
    'rice_rifat': Path('data/raw/rice_pest_rifat'),
    'plantwild': Path('data/raw/plantwild'),
    'wheat_mendeley': Path('data/raw/wheat_mendeley'),
    'wheat_small': Path('data/raw/wheat_small'),
    'sugarcane_thite': Path('data/raw/sugarcane_thite'),
    'sugarcane_daphal': Path('data/raw/sugarcane_daphal'),
    'not_crop': Path('data/raw/not_crop'),
    'openset': Path('data/raw/openset')
}

# New sources
new_dirs = {
    'mendeley_rice': Path('data/downloads_v3/source1_mendeley_hx6f852hw4/extracted/Original Images'),
    'dhan_shomadhan': Path('data/downloads_v3/source2_mendeley_znsxdctwtt/images'),
    'banglariceleaf': Path('data/downloads_v3/source4_banglariceleaf/extracted'),
    'philippines': Path('data/downloads_v3/source3_kaggle_philippines/resized_raw_images/resized_raw_images'),
    'kushagra_wheat': Path('data/downloads_v3/source7_kushagra_wheat/data'),
    'sugarcane_ld_bd': Path('data/downloads_v3/source8_sugarcane_ld_bd')
}

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.jfif', '.webp'}

def get_images(d):
    return [p for p in d.glob('**/*.*') if p.suffix.lower() in VALID_EXTS]

existing_files = {name: get_images(d) for name, d in existing_dirs.items()}
new_files = {name: get_images(d) for name, d in new_dirs.items()}

# Cache file
cache_file = Path('/tmp/sih_dedup_hashes_v3.json')
all_hashes = {}
if cache_file.exists():
    try:
        with open(cache_file) as f:
            all_hashes = json.load(f)
        print(f"Loaded {len(all_hashes)} hashes from cache.")
    except Exception:
        all_hashes = {}

# Compute missing hashes
to_hash_existing = []
for name, flist in existing_files.items():
    missing = [f for f in flist if str(f) not in all_hashes]
    if missing:
        to_hash_existing.extend(missing)

to_hash_new = []
for name, flist in new_files.items():
    missing = [f for f in flist if str(f) not in all_hashes]
    if missing:
        to_hash_new.extend(missing)

if to_hash_existing:
    print(f"Hashing {len(to_hash_existing)} existing files...")
    h = hash_directory(to_hash_existing, "existing_missing")
    all_hashes.update(h)

if to_hash_new:
    print(f"Hashing {len(to_hash_new)} new files...")
    h = hash_directory(to_hash_new, "new_missing")
    all_hashes.update(h)

with open(cache_file, 'w') as f:
    json.dump(all_hashes, f)
print(f"Total cached hashes: {len(all_hashes)}")

# Organize hashes by dataset
existing_hashes = {}
for name, flist in existing_files.items():
    existing_hashes[name] = np.array([all_hashes[str(f)] for f in flist if str(f) in all_hashes], dtype=np.uint64)
    print(f"Existing {name:<18}: {len(existing_hashes[name])} hashed images")

new_hashes = {}
for name, flist in new_files.items():
    new_hashes[name] = np.array([all_hashes[str(f)] for f in flist if str(f) in all_hashes], dtype=np.uint64)
    print(f"New      {name:<18}: {len(new_hashes[name])} hashed images")

def count_duplicates(hashes_a, hashes_b, threshold=4):
    """Counts how many unique images in A have at least one match in B with Hamming dist <= threshold."""
    if len(hashes_a) == 0 or len(hashes_b) == 0:
        return 0
    # Vectorized chunked comparison
    a = hashes_a
    b = hashes_b
    dups = 0
    # chunk A by 500 to keep memory low
    chunk_size = 500
    for i in range(0, len(a), chunk_size):
        chunk_a = a[i:i+chunk_size, None] # (C, 1)
        # xor with b (1, M)
        xor_res = np.bitwise_xor(chunk_a, b[None, :])
        # bit count
        # In numpy, unpackbits or bit_count on uint64:
        # np.bitwise_count is available in numpy >= 2.0, or gmpy2, or python int bit_count
        # Let's do fast uint64 bit count using standard bit operations:
        # v = v - ((v >> 1) & 0x5555555555555555)
        # v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333)
        # v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0F
        # v = (v * 0x0101010101010101) >> 56
        v = xor_res
        v = v - ((v >> np.uint64(1)) & np.uint64(0x5555555555555555))
        v = (v & np.uint64(0x3333333333333333)) + ((v >> np.uint64(2)) & np.uint64(0x3333333333333333))
        v = (v + (v >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        dist = (v * np.uint64(0x0101010101010101)) >> np.uint64(56)
        # minimum distance for each item in chunk_a across all B
        min_dist = np.min(dist, axis=1)
        dups += np.sum(min_dist <= threshold)
    return int(dups)

print("\n=== 2. FULL DEDUP MATRIX: NEW SOURCES vs EXISTING SOURCES (Hamming <= 4) ===")
# Header
print(f"{'New Source':<18} | {'Total Imgs':<10} | " + " | ".join([f"{k[:7]:<7}" for k in existing_hashes.keys()]))
print("-" * 125)

matrix_new_vs_exist = {}
for n_name, n_h in new_hashes.items():
    row = [f"{n_name:<18}", f"{len(n_h):<10}"]
    matrix_new_vs_exist[n_name] = {}
    for e_name, e_h in existing_hashes.items():
        cnt = count_duplicates(n_h, e_h, threshold=4)
        pct = (cnt / len(n_h) * 100) if len(n_h) > 0 else 0
        matrix_new_vs_exist[n_name][e_name] = (cnt, pct)
        cell = f"{cnt}({pct:.0f}%)" if cnt > 0 else "0"
        row.append(f"{cell:<7}")
    print(" | ".join(row))

print("\n=== 3. FULL DEDUP MATRIX: NEW SOURCES vs NEW SOURCES (Hamming <= 4) ===")
print(f"{'New Source':<18} | {'Total Imgs':<10} | " + " | ".join([f"{k[:8]:<8}" for k in new_hashes.keys()]))
print("-" * 90)

matrix_new_vs_new = {}
for n1_name, n1_h in new_hashes.items():
    row = [f"{n1_name:<18}", f"{len(n1_h):<10}"]
    matrix_new_vs_new[n1_name] = {}
    for n2_name, n2_h in new_hashes.items():
        if n1_name == n2_name:
            row.append(f"{'-':<8}")
            matrix_new_vs_new[n1_name][n2_name] = (0, 0.0)
        else:
            cnt = count_duplicates(n1_h, n2_h, threshold=4)
            pct = (cnt / len(n1_h) * 100) if len(n1_h) > 0 else 0
            matrix_new_vs_new[n1_name][n2_name] = (cnt, pct)
            cell = f"{cnt}({pct:.0f}%)" if cnt > 0 else "0"
            row.append(f"{cell:<8}")
    print(" | ".join(row))

print("\n=== 4. PHILIPPINES OPENSET CANDIDATES vs ALL EXISTING TRAINING DATA ===")
# Existing train files from splits_v2/train.csv
import pandas as pd
train_df = pd.read_csv('splits_v2/train.csv')
train_paths = set(train_df['path'].tolist())
train_hashes = []
for p in train_paths:
    full_p = str(Path(p))
    if full_p in all_hashes:
        train_hashes.append(all_hashes[full_p])
train_hashes = np.array(train_hashes, dtype=np.uint64)
print(f"Loaded {len(train_hashes)} existing training image hashes from splits_v2/train.csv")

phil_openset_dirs = [
    'stem_rot', 'sheath_rot', 'bakanae', 'rice_false_smut',
    'grassy_stunt_virus', 'ragged_stunt_virus', 'narrow_brown_spot', 'sheath_blight'
]

phil_base = Path('data/downloads_v3/source3_kaggle_philippines/resized_raw_images/resized_raw_images')
for cat in phil_openset_dirs:
    cat_d = phil_base / cat
    if not cat_d.exists():
        print(f"Philippines openset class {cat}: DIR NOT FOUND")
        continue
    c_files = get_images(cat_d)
    c_hashes = np.array([all_hashes[str(f)] for f in c_files if str(f) in all_hashes], dtype=np.uint64)
    cnt = count_duplicates(c_hashes, train_hashes, threshold=4)
    pct = cnt / len(c_hashes) * 100 if len(c_hashes) > 0 else 0
    status = "REJECT FROM OPENSET (TRAIN LEAKAGE!)" if cnt > 0 else "CLEAN (OK FOR OPENSET)"
    print(f"Philippines openset {cat:<22} ({len(c_hashes)} imgs): {cnt} hits in train ({pct:.1f}%) -> {status}")
