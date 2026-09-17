import os
import sys
import zipfile
import urllib.request
from pathlib import Path

DEST_DIR = Path('/Users/mohdahsan/Downloads/SIH/sih-smart-farming/data/raw/rice_pest_rifat')
DEST_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = DEST_DIR / 'rice_dataset.zip'
URL = 'https://data.mendeley.com/public-files/datasets/vwv3nry3wr/files/870d94bc-4acb-4526-92a7-21a6f855b461/file_downloaded'

print(f"Downloading Rifat rice pest dataset from {URL} to {ZIP_PATH}...")
req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=120) as resp, open(ZIP_PATH, 'wb') as out_f:
    total_size = int(resp.headers.get('content-length', 0))
    downloaded = 0
    chunk_size = 1024 * 1024 * 4 # 4MB chunks
    while True:
        chunk = resp.read(chunk_size)
        if not chunk:
            break
        out_f.write(chunk)
        downloaded += len(chunk)
        if total_size > 0 and downloaded % (chunk_size * 20) == 0:
            print(f"Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({downloaded/total_size*100:.1f}%)")

print("Download complete. Inspecting archive contents...")
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    all_names = z.namelist()
    print(f"Total entries in zip: {len(all_names)}")
    leaffolder_files = [n for n in all_names if 'leaffolder' in n.lower() and not n.endswith('/')]
    print(f"Total leaffolder files found: {len(leaffolder_files)}")
    # Look for original files
    original_leaffolder = [n for n in leaffolder_files if 'aug' not in n.lower()]
    print(f"Non-augmented leaffolder files: {len(original_leaffolder)}")
    if len(original_leaffolder) > 300:
        sample_lf = original_leaffolder[:10]
        print("Sample paths:", sample_lf)
        original_leaffolder = [n for n in original_leaffolder if 'original' in n.lower() or 'unaug' in n.lower()]
        print(f"Filtered to original: {len(original_leaffolder)}")
    
    target_extract = DEST_DIR / 'original_leaffolder'
    target_extract.mkdir(exist_ok=True)
    for member in original_leaffolder:
        fname = Path(member).name
        with z.open(member) as src, open(target_extract / fname, 'wb') as dst:
            dst.write(src.read())

print(f"Extracted {len(list(target_extract.glob('*')))} files to {target_extract}")
print("Removing zip archive...")
ZIP_PATH.unlink()
print("Done!")
