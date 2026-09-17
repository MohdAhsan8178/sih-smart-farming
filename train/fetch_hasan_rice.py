import os
import sys
import zipfile
import urllib.request
from pathlib import Path

DEST_DIR = Path('/Users/mohdahsan/Downloads/SIH/sih-smart-farming/data/raw/rice_hasan')
OPENSET_DIR = Path('/Users/mohdahsan/Downloads/SIH/sih-smart-farming/data/raw/openset/hasan_openset')
DEST_DIR.mkdir(parents=True, exist_ok=True)
OPENSET_DIR.mkdir(parents=True, exist_ok=True)

ZIP_PATH = DEST_DIR / 'rice_hasan.zip'
URL = 'https://data.mendeley.com/public-files/datasets/hx6f852hw4/files/482181c7-57d6-4710-9f5c-be059ed382ca/file_downloaded'

print(f"Downloading Hasan rice dataset from {URL} to {ZIP_PATH}...", flush=True)
req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=180) as resp, open(ZIP_PATH, 'wb') as out_f:
    total_size = int(resp.headers.get('content-length', 0))
    downloaded = 0
    chunk_size = 1024 * 1024 * 4
    while True:
        chunk = resp.read(chunk_size)
        if not chunk:
            break
        out_f.write(chunk)
        downloaded += len(chunk)
        if total_size > 0 and (downloaded // chunk_size) % 15 == 0:
            print(f"Hasan: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({downloaded/total_size*100:.1f}%)", flush=True)

print("Hasan download complete. Inspecting archive...", flush=True)
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    all_names = z.namelist()
    print(f"Total entries in Hasan zip: {len(all_names)}", flush=True)
    
    # Let's inspect directory structure
    dirs = {n.split('/')[0] for n in all_names if '/' in n}
    print("Top-level entries:", dirs, flush=True)
    
    # We want original images:
    # 'Rice Hispa' -> DEST_DIR / 'rice__hispa'
    # 'Healthy Rice Leaf' -> DEST_DIR / 'rice__normal'
    # 'Sheath Blight', 'Leaf scald', 'Narrow Brown Leaf Spot' -> OPENSET_DIR / <subfolder>
    
    # Separate original vs augmented
    # In Mendeley description: "There are 1,701 original images in the entire collection... The file also includes 5188 extra augmented photos"
    # Look for paths containing 'original' or not containing 'augmented'
    for name in all_names:
        if name.endswith('/'):
            continue
        lower = name.lower()
        if 'aug' in lower:
            continue # Skip augmented per user instructions
            
        fname = Path(name).name
        if 'hispa' in lower:
            target = DEST_DIR / 'hispa'
            target.mkdir(exist_ok=True)
            with z.open(name) as src, open(target / fname, 'wb') as dst:
                dst.write(src.read())
        elif 'healthy' in lower:
            target = DEST_DIR / 'healthy'
            target.mkdir(exist_ok=True)
            with z.open(name) as src, open(target / fname, 'wb') as dst:
                dst.write(src.read())
        elif 'sheath' in lower:
            target = OPENSET_DIR / 'sheath_blight'
            target.mkdir(exist_ok=True)
            with z.open(name) as src, open(target / fname, 'wb') as dst:
                dst.write(src.read())
        elif 'scald' in lower:
            target = OPENSET_DIR / 'leaf_scald'
            target.mkdir(exist_ok=True)
            with z.open(name) as src, open(target / fname, 'wb') as dst:
                dst.write(src.read())
        elif 'narrow' in lower:
            target = OPENSET_DIR / 'narrow_brown_spot'
            target.mkdir(exist_ok=True)
            with z.open(name) as src, open(target / fname, 'wb') as dst:
                dst.write(src.read())

print("Hasan extraction complete. Removing zip archive...", flush=True)
ZIP_PATH.unlink()
print("Done with Hasan dataset!", flush=True)
