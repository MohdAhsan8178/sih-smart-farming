"""
Measure whether dedup.py (which uses PIL Image.open without exif_transpose)
misses any near-duplicate pairs compared to evaluating images with exif_transpose.
"""

import os
import random
from PIL import Image, ImageOps
import imagehash

def measure_f2_impact(sample_size=50, seed=42):
    not_crop_base = "/Users/mohdahsan/Downloads/SIH/sih-smart-farming/data/raw/not_crop"
    folders = ["feet_shoes", "pavement", "walls", "soil_field", "green_noncrop"]

    tag6_images = []
    for fld in folders:
        p = os.path.join(not_crop_base, fld)
        if not os.path.isdir(p):
            continue
        for fname in sorted(os.listdir(p)):
            if fname.startswith("."):
                continue
            fpath = os.path.join(p, fname)
            try:
                with Image.open(fpath) as img:
                    exif = img.getexif()
                    if exif and exif.get(0x0112) == 6:
                        tag6_images.append(fpath)
            except Exception:
                pass

    print(f"[measure_f2] Total images with orientation tag 6 found: {len(tag6_images)}")
    rng = random.Random(seed)
    sample = rng.sample(tag6_images, min(sample_size, len(tag6_images)))
    print(f"[measure_f2] Sampled {len(sample)} images for rotation hash evaluation (seed={seed}).")

    raw_hashes = []
    trans_hashes = []

    for p in sample:
        with Image.open(p) as img:
            h_raw = imagehash.phash(img.convert("RGB"), hash_size=8)
            h_trans = imagehash.phash(ImageOps.exif_transpose(img).convert("RGB"), hash_size=8)
            raw_hashes.append(h_raw)
            trans_hashes.append(h_trans)

    self_dists = [r - t for r, t in zip(raw_hashes, trans_hashes)]
    print(f"[measure_f2] Self-hash Hamming distance (raw vs transposed): min={min(self_dists)}, max={max(self_dists)}, mean={sum(self_dists)/len(self_dists):.2f}")

    n = len(sample)
    raw_dups = 0
    trans_dups = 0
    missed_by_raw = 0
    false_by_raw = 0
    both_dups = 0

    for i in range(n):
        for j in range(i + 1, n):
            d_raw = raw_hashes[i] - raw_hashes[j]
            d_trans = trans_hashes[i] - trans_hashes[j]

            if d_raw <= 5:
                raw_dups += 1
            if d_trans <= 5:
                trans_dups += 1

            if d_raw <= 5 and d_trans <= 5:
                both_dups += 1
            elif d_raw > 5 and d_trans <= 5:
                missed_by_raw += 1
            elif d_raw <= 5 and d_trans > 5:
                false_by_raw += 1

    print(f"[measure_f2] Total pairs evaluated: {n*(n-1)//2}")
    print(f"[measure_f2] Duplicates in raw orientation (Hamming <= 5): {raw_dups}")
    print(f"[measure_f2] Duplicates in transposed orientation (Hamming <= 5): {trans_dups}")
    print(f"[measure_f2] Pairs grouped in BOTH orientations: {both_dups}")
    print(f"[measure_f2] Near-duplicates MISSED because of missing EXIF transpose (raw > 5, trans <= 5): {missed_by_raw}")
    print(f"[measure_f2] Spurious duplicates grouped in raw orientation but not transposed (raw <= 5, trans > 5): {false_by_raw}")

if __name__ == "__main__":
    measure_f2_impact()
