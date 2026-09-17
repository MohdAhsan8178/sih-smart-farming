"""
Download and verify raw datasets for the smart farming assistant.

Supports:
  --all               : Download all datasets (Paddy, PlantDoc, Sugarcane, PlantWild)
  --sugarcane         : Download Sugarcane Thite & Daphal datasets
  --plantwild         : Download PlantWild dataset via HuggingFace
  --verify            : Scan data/raw/ and write artifacts/reports/data_inventory.txt
"""

import sys
import os
import argparse
import subprocess
import zipfile
import shutil
import json
import urllib.request
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import (
    ROOT, RAW, PADDY, SUGAR_THITE, SUGAR_DAPHAL,
    PLANTDOC, PLANTWILD, REPORTS
)

def log(msg):
    print(f"[download_data] {msg}", flush=True)

def download_file(url, dest_path):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, 'wb') as out_file:
        total = int(resp.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = downloaded * 100 / total
                print(f"\r  Downloading {dest_path.name}: {downloaded/(1024*1024):.1f}/{total/(1024*1024):.1f} MB ({percent:.1f}%)", end="", flush=True)
            else:
                print(f"\r  Downloading {dest_path.name}: {downloaded/(1024*1024):.1f} MB", end="", flush=True)
        print("", flush=True)

def download_paddy():
    log("Checking Paddy Doctor dataset...")
    train_dir = PADDY / 'train_images'
    if train_dir.exists() and any(train_dir.iterdir()):
        log("Paddy Doctor already present.")
        return True

    PADDY.mkdir(parents=True, exist_ok=True)
    log("Downloading Paddy Doctor dataset via Kaggle...")
    res = subprocess.run([sys.executable, "-m", "kaggle", "competitions", "download",
                         "-c", "paddy-disease-classification", "-p", str(PADDY)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    zips = list(PADDY.glob("*.zip"))
    if not zips:
        log("Trying mirror dataset imbikramsaha/paddy-doctor...")
        res = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download",
                             "-d", "imbikramsaha/paddy-doctor", "-p", str(PADDY)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        zips = list(PADDY.glob("*.zip"))

    for zpath in zips:
        log(f"Extracting {zpath.name}...")
        with zipfile.ZipFile(zpath, 'r') as z:
            z.extractall(PADDY)
        zpath.unlink()

    nested = list(PADDY.glob("paddy-*"))
    for n in nested:
        if n.is_dir():
            for item in n.iterdir():
                shutil.move(str(item), str(PADDY))
            shutil.rmtree(n, ignore_errors=True)

    log("Paddy Doctor ready.")
    return True

def download_plantdoc():
    log("Checking PlantDoc dataset...")
    if PLANTDOC.exists() and any(PLANTDOC.iterdir()):
        log("PlantDoc already present.")
        return True

    log("Cloning PlantDoc repository from GitHub...")
    res = subprocess.run(["git", "clone", "--depth", "1",
                          "https://github.com/pratikkayal/PlantDoc-Dataset", str(PLANTDOC)])
    return res.returncode == 0

def download_sugarcane_thite():
    log("Checking Sugarcane (Thite - 355y629ynj)...")
    if SUGAR_THITE.exists():
        subdirs = [p for p in SUGAR_THITE.iterdir() if p.is_dir()]
        if len(subdirs) >= 9:
            log(f"Sugarcane Thite already present with {len(subdirs)} folders.")
            return True

    SUGAR_THITE.mkdir(parents=True, exist_ok=True)
    log("Querying Mendeley public API for dataset 355y629ynj files...")
    api_url = "https://data.mendeley.com/public-api/datasets/355y629ynj/files?folder_id=root&version=1"
    try:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            files_meta = json.loads(resp.read().decode())
    except Exception as e:
        log(f"Mendeley API query failed: {e}")
        print_manual_instructions("sugarcane_thite")
        return False

    log(f"Found {len(files_meta)} files in Sugarcane Thite.")
    for f in files_meta:
        fname = f['filename']
        dl_url = f.get('content_details', {}).get('download_url')
        if not dl_url:
            continue
        dest_zip = SUGAR_THITE / fname
        folder_name = dest_zip.stem
        out_dir = SUGAR_THITE / folder_name
        if out_dir.exists() and any(out_dir.iterdir()):
            log(f"  {folder_name} already extracted, skipping.")
            continue
        log(f"Fetching {fname}...")
        try:
            download_file(dl_url, dest_zip)
            with zipfile.ZipFile(dest_zip, 'r') as z:
                z.extractall(out_dir)
            dest_zip.unlink()
            log(f"Extracted {fname} -> {len(list(out_dir.glob('*')))} files")
        except Exception as e:
            log(f"Error processing {fname}: {e}")

    return True

def download_sugarcane_daphal():
    log("Checking Sugarcane (Daphal - 9424skmnrk)...")
    if SUGAR_DAPHAL.exists():
        subdirs = [p for p in SUGAR_DAPHAL.iterdir() if p.is_dir()]
        if len(subdirs) >= 4:
            log(f"Sugarcane Daphal already present with {len(subdirs)} folders.")
            return True

    SUGAR_DAPHAL.mkdir(parents=True, exist_ok=True)
    log("Querying Mendeley public API for dataset 9424skmnrk files...")
    api_url = "https://data.mendeley.com/public-api/datasets/9424skmnrk/files?folder_id=root&version=1"
    try:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            files_meta = json.loads(resp.read().decode())
    except Exception as e:
        log(f"Mendeley API query failed: {e}")
        print_manual_instructions("sugarcane_daphal")
        return False

    for f in files_meta:
        fname = f['filename']
        dl_url = f.get('content_details', {}).get('download_url')
        if not dl_url:
            continue
        dest_file = SUGAR_DAPHAL / fname
        log(f"Fetching {fname}...")
        try:
            download_file(dl_url, dest_file)
            log(f"Extracting {fname} via unar / bsdtar...")
            unar_bin = shutil.which("unar") or "/opt/homebrew/bin/unar"
            extracted = False
            if os.path.exists(unar_bin):
                res = subprocess.run([unar_bin, "-q", "-o", str(SUGAR_DAPHAL), str(dest_file)])
                extracted = (res.returncode == 0)
            if not extracted:
                bsdtar_bin = shutil.which("bsdtar") or "/opt/anaconda3/bin/bsdtar"
                if os.path.exists(bsdtar_bin):
                    res = subprocess.run([bsdtar_bin, "-xf", str(dest_file), "-C", str(SUGAR_DAPHAL)])
                    extracted = (res.returncode == 0)
            if extracted:
                log(f"Extracted {fname} successfully.")
                dest_file.unlink()
            else:
                log(f"Failed extracting {fname}.")
        except Exception as e:
            log(f"Error downloading {fname}: {e}")

    return True

def download_plantwild():
    log("Checking PlantWild dataset...")
    if PLANTWILD.exists() and any(PLANTWILD.iterdir()):
        log("PlantWild already present.")
        return True

    PLANTWILD.mkdir(parents=True, exist_ok=True)
    log("Attempting download of PlantWild from HuggingFace (uqtwei2/PlantWild)...")
    try:
        from huggingface_hub import hf_hub_download
        zip_path = hf_hub_download(
            repo_id="uqtwei2/PlantWild",
            filename="plantwild_v2.zip",
            repo_type="dataset",
            local_dir=str(PLANTWILD)
        )
        log(f"Downloaded PlantWild zip to {zip_path}. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(PLANTWILD)
        Path(zip_path).unlink(missing_ok=True)
        # flatten if inside plantwild_v2
        nested = PLANTWILD / "plantwild_v2"
        if nested.exists():
            for item in nested.iterdir():
                shutil.move(str(item), str(PLANTWILD))
            nested.rmdir()
        log("PlantWild extracted successfully.")
        return True
    except Exception as e:
        log(f"HuggingFace download failed or unavailable: {e}")
        print_manual_instructions("plantwild")
        return False

def print_manual_instructions(name=None):
    instructions = {
        "paddy": (
            "PADDY DOCTOR MANUAL DOWNLOAD:\n"
            "  1. Visit: https://www.kaggle.com/competitions/paddy-disease-classification\n"
            "  2. Click 'Download All' (or download train.csv and train_images.zip)\n"
            f"  3. Extract contents into: {PADDY}\n"
        ),
        "sugarcane_thite": (
            "SUGARCANE (THITE) MANUAL DOWNLOAD:\n"
            "  1. Visit: https://data.mendeley.com/datasets/355y629ynj/1\n"
            "  2. Download the dataset zip (or individual class zips)\n"
            f"  3. Extract folders into: {SUGAR_THITE}\n"
        ),
        "sugarcane_daphal": (
            "SUGARCANE (DAPHAL) MANUAL DOWNLOAD:\n"
            "  1. Visit: https://data.mendeley.com/datasets/9424skmnrk/1\n"
            "  2. Download 'Sugarcane Leaf Disease Dataset.rar'\n"
            f"  3. Extract folders into: {SUGAR_DAPHAL}\n"
        ),
        "plantdoc": (
            "PLANTDOC MANUAL DOWNLOAD:\n"
            "  1. Clone: git clone https://github.com/pratikkayal/PlantDoc-Dataset\n"
            f"  2. Place inside: {PLANTDOC}\n"
        ),
        "plantwild": (
            "PLANTWILD MANUAL DOWNLOAD:\n"
            "  1. Visit MVPDR repo: https://github.com/tqwei05/MVPDR\n"
            "  2. Sources:\n"
            "     - Hugging Face: https://huggingface.co/datasets/uqtwei2/PlantWild\n"
            "     - UQRDM: https://cloud.rdm.uq.edu.au/index.php/s/5iTzoyby9Xobmq2 (password: plantwildv1)\n"
            "     - Google Drive: https://drive.google.com/file/d/1s7FOoztTHvO03yVfw75pQY_kzZqvAckD/view?usp=drive_link\n"
            f"  3. Extract contents into: {PLANTWILD}\n"
        )
    }
    print("=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    if name and name in instructions:
        print(instructions[name])
    else:
        for k, v in instructions.items():
            print(v)
    print("=" * 70)

def count_images_and_classes(dataset_path):
    if not dataset_path.exists():
        return 0, 0
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    total_imgs = 0
    
    # Identify root for classes
    scan_dir = dataset_path
    if (dataset_path / 'train_images').exists():
        class_scan_dir = dataset_path / 'train_images'
    elif (dataset_path / 'train').exists():
        class_scan_dir = dataset_path / 'train'
    else:
        class_scan_dir = dataset_path

    class_folders = set()
    for d in class_scan_dir.iterdir():
        if d.is_dir() and not d.name.startswith('.'):
            class_folders.add(d.name)

    for root, dirs, files in os.walk(dataset_path):
        if '.git' in root or '.cache' in root:
            continue
        imgs = [f for f in files if Path(f).suffix.lower() in img_exts]
        total_imgs += len(imgs)

    return total_imgs, len(class_folders)

def verify_all():
    log("Verifying raw datasets under data/raw/...")
    datasets = [
        ("Paddy Doctor", PADDY),
        ("Sugarcane (Thite)", SUGAR_THITE),
        ("Sugarcane (Daphal)", SUGAR_DAPHAL),
        ("PlantDoc", PLANTDOC),
        ("PlantWild", PLANTWILD),
    ]

    header = f"{'Dataset':<22} | {'Found Images':<12} | {'Found Classes':<13} | {'Status':<10}"
    sep = "-" * len(header)
    lines = [header, sep]
    
    all_nonzero = True
    any_missing = []

    for name, p in datasets:
        imgs, classes = count_images_and_classes(p)
        status = "OK" if imgs > 0 else "EMPTY"
        if imgs == 0:
            all_nonzero = False
            any_missing.append(name)
        lines.append(f"{name:<22} | {imgs:<12} | {classes:<13} | {status:<10}")

    table_str = "\n".join(lines)
    print("\n" + table_str + "\n")

    REPORTS.mkdir(parents=True, exist_ok=True)
    inv_file = REPORTS / "data_inventory.txt"
    inv_file.write_text(table_str + "\n")
    log(f"Wrote inventory report to {inv_file}")

    if not all_nonzero:
        log(f"WARNING: The following datasets have 0 images: {', '.join(any_missing)}")
        print_manual_instructions()
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Download and verify smart farming datasets.")
    parser.add_argument("--all", action="store_true", help="Download all datasets.")
    parser.add_argument("--sugarcane", action="store_true", help="Download Sugarcane Thite & Daphal datasets.")
    parser.add_argument("--plantwild", action="store_true", help="Download PlantWild dataset.")
    parser.add_argument("--verify", action="store_true", help="Verify dataset presence and write report.")
    args = parser.parse_args()

    if not args.all and not args.sugarcane and not args.plantwild and not args.verify:
        parser.print_help()
        sys.exit(1)

    if args.sugarcane or args.all:
        download_sugarcane_thite()
        download_sugarcane_daphal()

    if args.plantwild or args.all:
        download_plantwild()

    if args.all:
        download_paddy()
        download_plantdoc()

    if args.verify:
        success = verify_all()
        if not success:
            log("Verification incomplete: some datasets have 0 files.")
            sys.exit(1)
        else:
            log("Verification SUCCESS: All datasets contain files.")

if __name__ == "__main__":
    main()
