"""
Download script for Model B raw datasets:
1. Ong & Høye (Figshare 23617383) — DSLR, Webcam, Smartphone, Other objects
2. PST Whitefly Dataset (Zenodo 7801239)
3. GinJinn2 Yellow Sticky Traps (BGBM 0036)
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import DATA

RAW_MODEL_B = DATA / "model_b_sources"
RAW_MODEL_B.mkdir(parents=True, exist_ok=True)


def download_with_curl(url: str, dest_path: Path) -> bool:
    """Download a file using curl with resume and follow-redirects."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"File already exists: {dest_path.name} ({dest_path.stat().st_size} bytes)")
        return True
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    cmd = [
        "curl", "-f", "-L", "-C", "-",
        "--retry", "3",
        "--retry-delay", "2",
        "-o", str(temp_path),
        url
    ]
    print(f"Downloading {dest_path.name} from {url}...")
    res = subprocess.run(cmd)
    if res.returncode == 0 and temp_path.exists():
        temp_path.rename(dest_path)
        print(f"Successfully downloaded {dest_path.name} ({dest_path.stat().st_size} bytes)")
        return True
    else:
        print(f"Failed to download {dest_path.name}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Safely extract zip archive."""
    if not zip_path.exists():
        return False
    print(f"Extracting {zip_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        print(f"Extracted {zip_path.name}")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path.name}: {e}")
        return False


def download_ong_hoye():
    """Download Ong & Høye (Figshare 23617383)."""
    target_dir = RAW_MODEL_B / "ong_hoye"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    files = {
        "Webcam.zip": "https://ndownloader.figshare.com/files/41437941",
        "Smart_phone.zip": "https://ndownloader.figshare.com/files/41437938",
        "DSLR.zip": "https://ndownloader.figshare.com/files/41437935",
        "Other_objects-samples.zip": "https://ndownloader.figshare.com/files/45563223",
    }
    
    for fname, url in files.items():
        zpath = target_dir / fname
        if download_with_curl(url, zpath):
            extract_zip(zpath, target_dir / fname.replace(".zip", ""))


def download_pst():
    """Download PST Whitefly Dataset (Zenodo 7801239)."""
    target_dir = RAW_MODEL_B / "pst"
    target_dir.mkdir(parents=True, exist_ok=True)
    zpath = target_dir / "pest-sticky-traps.zip"
    url = "https://zenodo.org/records/7801239/files/pest-sticky-traps.zip"
    if download_with_curl(url, zpath):
        extract_zip(zpath, target_dir / "extracted")


def download_ginjinn2():
    """Download GinJinn2 Yellow Sticky Traps Dataset (BGBM 0036)."""
    target_dir = RAW_MODEL_B / "ginjinn2"
    target_dir.mkdir(parents=True, exist_ok=True)
    zpath = target_dir / "gfbio0036.zip"
    url = "https://data.bgbm.org/dataset/gfbio/0036/gfbio0036.zip"
    if download_with_curl(url, zpath):
        extract_zip(zpath, target_dir / "extracted")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ong_hoye", action="store_true", help="Download Ong & Høye dataset")
    parser.add_argument("--pst", action="store_true", help="Download PST dataset")
    parser.add_argument("--ginjinn2", action="store_true", help="Download GinJinn2 dataset")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    args = parser.parse_args()

    if args.all or args.ong_hoye:
        download_ong_hoye()
    if args.all or args.pst:
        download_pst()
    if args.all or args.ginjinn2:
        download_ginjinn2()
