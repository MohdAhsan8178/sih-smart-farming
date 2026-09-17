import os, sys, shutil, json, re, random
from pathlib import Path
from PIL import Image
import pandas as pd

def sanitize_filename(name):
    clean = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    clean = re.sub(r'_+', '_', clean)
    return clean

print("=== STARTING PHASE 2 INGESTION & SANITIZATION ===")

# Load cached hashes for dedup enforcement
with open('/tmp/sih_dedup_hashes_v3.json') as f:
    all_hashes = json.load(f)

existing_hashes = {p: h for p, h in all_hashes.items() if not 'data/downloads_v3/' in p}
print(f"Loaded {len(existing_hashes)} existing image hashes for dedup enforcement.")

rename_map = {}
copied_summary = {}

def copy_and_sanitize(src_path, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    assert not re.search(r'[\s\(\)\[\]]', dst_path.name), f"Illegal char in {dst_path.name}"
    assert re.match(r'^[a-zA-Z0-9._-]+$', dst_path.name), f"Non-compliant name {dst_path.name}"
    rename_map[str(src_path)] = str(dst_path)

# ==========================================
# 1. BANGLARICELEAF (Source 4)
# ==========================================
print("\n--- 1. Processing BanglaRiceLeaf ---")
br_src = Path('data/downloads_v3/source4_banglariceleaf/extracted')
br_dst = Path('data/raw/banglariceleaf')

br_classes = {
    'Bacterial Leaf Streak (BLS)': ('bacterial_leaf_streak', 600, False),
    'Bacterial Leaf Blight (BLB)':  ('bacterial_leaf_blight', 600, False),
    'Leaf Blast':                  ('leaf_blast', 600, False),
    'Healthy Leaf':                ('normal', 500, False),
    'Sheath Blight':               ('sheath_blight', 418, True) # Openset
}

random.seed(42)
for in_cls, (out_cls, cap, is_openset) in br_classes.items():
    src_dir = br_src / in_cls
    imgs = sorted(list(src_dir.glob('*.jpg')))
    clean_imgs = []
    for img in imgs:
        h = all_hashes.get(str(img))
        if h is not None:
            dup = any((h ^ eh).bit_count() <= 4 for ep, eh in existing_hashes.items())
            if not dup:
                clean_imgs.append(img)
        else:
            clean_imgs.append(img)
    
    if len(clean_imgs) > cap:
        selected = sorted(random.sample(clean_imgs, cap))
    else:
        selected = clean_imgs
    
    if is_openset:
        target_dir = Path('data/raw/openset') / f'banglariceleaf_{out_cls}'
    else:
        target_dir = br_dst / out_cls
    
    for f in selected:
        s_name = sanitize_filename(f.name)
        copy_and_sanitize(f, target_dir / s_name)
    
    copied_summary[f"banglariceleaf/{out_cls}"] = len(selected)
    print(f"BanglaRiceLeaf {in_cls:<28} -> {str(target_dir):<45}: kept {len(selected)} (from {len(imgs)})")

# ==========================================
# 2. DHAN-SHOMADHAN (Source 2)
# ==========================================
print("\n--- 2. Processing Dhan-Shomadhan ---")
ds_csv = Path('data/downloads_v3/source2_mendeley_znsxdctwtt/Dhan-Shomadhan_picture_Information.csv')
ds_img_dir = Path('data/downloads_v3/source2_mendeley_znsxdctwtt/images')
ds_df = pd.read_csv(ds_csv)
ds_dst = Path('data/raw/dhan_shomadhan')

# Mapping including exact spellings in author CSV
ds_mapping = {
    'Rice Tungro(Feild Background)':  ('tungro', False),
    'Rice Turgro(Feild Background)':  ('tungro', False), # author typo
    'Rice Tungro(white Background)':  ('tungro', False),
    'Browon Spot(Feild Background)':  ('brown_spot', False),
    'Brown Spot(white Background)':   ('brown_spot', False),
    'Rice Blast(Feild Background)':   ('blast', False),
    'Rice Blast(white Background)':   ('blast', False),
    'Leaf Scaled(Feild Background)':  ('leaf_scald', True),
    'Leaf Scaled(white Background)':  ('leaf_scald', True),
    'Sheath Blight(Feild Background)':('sheath_blight', True),
    'Shath Blight(white Background)': ('sheath_blight', True)
}

ds_counts = {}
for idx, row in ds_df.iterrows():
    pic = row['pictureName']
    dis = row['Diseases']
    if dis not in ds_mapping:
        continue
    out_cls, is_openset = ds_mapping[dis]
    src_f = ds_img_dir / pic
    if not src_f.exists():
        continue
    
    # Dedup
    h = all_hashes.get(str(src_f))
    if h is not None:
        if any((h ^ eh).bit_count() <= 4 for ep, eh in existing_hashes.items()):
            continue
    
    if is_openset:
        target_dir = Path('data/raw/openset') / f'dhan_shomadhan_{out_cls}'
    else:
        target_dir = ds_dst / out_cls
    
    s_name = sanitize_filename(f"ds_{pic}")
    copy_and_sanitize(src_f, target_dir / s_name)
    ds_counts[out_cls] = ds_counts.get(out_cls, 0) + 1

for k, v in ds_counts.items():
    copied_summary[f"dhan_shomadhan/{k}"] = v
    print(f"Dhan-Shomadhan {k:<25}: kept {v}")

# ==========================================
# 3. RICE MENDELEY (Source 1)
# ==========================================
print("\n--- 3. Processing Rice Mendeley (hx6f852hw4) ---")
m_src = Path('data/downloads_v3/source1_mendeley_hx6f852hw4/extracted/Original Images')
m_dst = Path('data/raw/rice_mendeley')

m_classes = {
    'Bacterial Leaf Blight': ('bacterial_leaf_blight', 600, False),
    'Brown Spot':            ('brown_spot', 600, False),
    'Leaf Blast':            ('leaf_blast', 600, False)
}

for in_cls, (out_cls, cap, is_openset) in m_classes.items():
    src_dir = m_src / in_cls
    imgs = sorted(list(src_dir.glob('*.jpg')))
    clean_imgs = []
    for img in imgs:
        h = all_hashes.get(str(img))
        if h is not None:
            dup = any((h ^ eh).bit_count() <= 4 for ep, eh in existing_hashes.items())
            if not dup:
                clean_imgs.append(img)
        else:
            clean_imgs.append(img)
    
    target_dir = m_dst / out_cls
    for f in clean_imgs:
        s_name = sanitize_filename(f"rm_{f.name}")
        copy_and_sanitize(f, target_dir / s_name)
    copied_summary[f"rice_mendeley/{out_cls}"] = len(clean_imgs)
    print(f"Rice Mendeley {in_cls:<25} -> {out_cls:<25}: kept {len(clean_imgs)} (from {len(imgs)})")

# ==========================================
# 4. KUSHAGRA WHEAT (Source 7) - Mildew ONLY
# ==========================================
print("\n--- 4. Processing Kushagra Wheat Mildew ---")
kw_src_train = Path('data/downloads_v3/source7_kushagra_wheat/data/train/Mildew')
kw_src_test = Path('data/downloads_v3/source7_kushagra_wheat/data/test/mildew_test')
kw_dst = Path('data/raw/kushagra_wheat/powdery_mildew')

kw_imgs = sorted(list(kw_src_train.glob('*.*')) + list(kw_src_test.glob('*.*')))
existing_wheat = {p: h for p, h in existing_hashes.items() if any(w in p for w in ['plantwild', 'wheat_small', 'wheat_mendeley'])}
clean_kw = []
for img in kw_imgs:
    h = all_hashes.get(str(img))
    if h is not None:
        dup = any((h ^ wh).bit_count() <= 4 for wp, wh in existing_wheat.items())
        if not dup:
            clean_kw.append(img)
    else:
        clean_kw.append(img)

print(f"Kushagra Mildew: total {len(kw_imgs)}, non-duplicate clean {len(clean_kw)}")
random.seed(42)
selected_kw = sorted(random.sample(clean_kw, 600))

for f in selected_kw:
    s_name = sanitize_filename(f"kw_{f.name}")
    copy_and_sanitize(f, kw_dst / s_name)

copied_summary["kushagra_wheat/powdery_mildew"] = len(selected_kw)
print(f"Kushagra Wheat Mildew: kept exactly {len(selected_kw)} (capped at 600, seed 42)")

# ==========================================
# 5. SUGARCANE LD-BD (Source 8)
# ==========================================
print("\n--- 5. Processing SugarcaneLD-BD ---")
sc_src = Path('data/downloads_v3/source8_sugarcane_ld_bd/SugarcaneLD-BD')
sc_dst = Path('data/raw/sugarcane_ld_bd')

sc_classes = {
    'RedRot':      ('red_rot', False),
    'Healthy':     ('healthy', False),
    'RingSpot':    ('ring_spot', True), # openset
    'EyeSpot':     ('eye_spot', True),  # openset
    'RedLeafSpot': ('red_leaf_spot', True) # openset
}

for in_cls, (out_cls, is_openset) in sc_classes.items():
    src_dir = sc_src / in_cls
    imgs = sorted(list(src_dir.glob('*.jpg')))
    clean_imgs = []
    for img in imgs:
        h = all_hashes.get(str(img))
        if h is not None:
            dup = any((h ^ eh).bit_count() <= 4 for ep, eh in existing_hashes.items())
            if not dup:
                clean_imgs.append(img)
        else:
            clean_imgs.append(img)
    
    if is_openset:
        target_dir = Path('data/raw/openset') / f'sugarcane_{out_cls}'
    else:
        target_dir = sc_dst / out_cls
    
    for f in clean_imgs:
        s_name = sanitize_filename(f"scld_{f.name}")
        copy_and_sanitize(f, target_dir / s_name)
    
    copied_summary[f"sugarcane_ld_bd/{out_cls}"] = len(clean_imgs)
    print(f"SugarcaneLD-BD {in_cls:<15} -> {out_cls:<15}: kept {len(clean_imgs)} (from {len(imgs)})")

# ==========================================
# 6. PHILIPPINES (Source 3) - 3 CLEAN OPENSET CLASSES ONLY
# ==========================================
print("\n--- 6. Processing Philippines Clean Openset Classes ---")
phil_src = Path('data/downloads_v3/source3_kaggle_philippines/resized_raw_images/resized_raw_images')
clean_phil_openset = ['bakanae', 'rice_false_smut', 'ragged_stunt_virus']

for cat in clean_phil_openset:
    src_dir = phil_src / cat
    imgs = sorted(list(src_dir.glob('*.*')))
    target_dir = Path('data/raw/openset') / f'philippines_{cat}'
    for f in imgs:
        s_name = sanitize_filename(f"phil_{f.name}")
        copy_and_sanitize(f, target_dir / s_name)
    copied_summary[f"philippines_openset/{cat}"] = len(imgs)
    print(f"Philippines {cat:<20} -> openset: kept {len(imgs)}")

# Save rename map
rename_map_path = Path('scripts/rename_map.json')
with open(rename_map_path, 'w') as f:
    json.dump(rename_map, f, indent=2)

rename_map_csv = Path('scripts/rename_map.csv')
with open(rename_map_csv, 'w') as f:
    f.write("old_relative_path,new_relative_path\n")
    for k, v in rename_map.items():
        f.write(f"{k},{v}\n")

print(f"\nTotal entries in rename_map: {len(rename_map)}")

print("\n=== SUMMARY OF COPIED IMAGES BY SOURCE/CLASS ===")
total_taxonomy = 0
total_openset = 0
for k, v in sorted(copied_summary.items()):
    is_open = 'openset' in k or any(x in k for x in ['leaf_scald', 'sheath_blight', 'ring_spot', 'eye_spot', 'red_leaf_spot'])
    print(f"  {k:<40}: {v:>5} {'(openset)' if is_open else '(taxonomy)'}")
    if is_open:
        total_openset += v
    else:
        total_taxonomy += v

print(f"\nTotal newly ingested taxonomy images: {total_taxonomy}")
print(f"Total newly ingested openset images : {total_openset}")
print(f"Total newly ingested images overall  : {total_taxonomy + total_openset}")

# Final assertion: check all files in new raw directories for compliant names
# Final assertion: check all ingested files in rename_map for compliant names
print("\n=== SANITIZATION AUDIT ON INGESTED FILES ===")
illegal_count = 0
for src_p, dst_p in rename_map.items():
    f = Path(dst_p)
    if not f.is_file():
        print(f"MISSING: {f}")
        illegal_count += 1
        continue
    if re.search(r'[\s\(\)\[\]]', f.name) or not re.match(r'^[a-zA-Z0-9._-]+$', f.name):
        print(f"ILLEGAL: {f}")
        illegal_count += 1
    # Check directory path segments as well
    for part in f.parts:
        if re.search(r'[\s\(\)\[\]]', part):
            print(f"ILLEGAL DIR PART in {f}: {part}")
            illegal_count += 1

assert illegal_count == 0, f"Found {illegal_count} illegal filenames/paths!"
print(f"Sanitization assertion PASSED: All {len(rename_map)} ingested files strictly conform to ^[a-zA-Z0-9._-]+$ with zero spaces, parentheses, or non-ASCII characters.")

