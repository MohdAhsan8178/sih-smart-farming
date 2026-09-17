import urllib.request, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

out_dir = Path("data/downloads_v3/source4_banglariceleaf/rar")
out_dir.mkdir(parents=True, exist_ok=True)
extract_dir = Path("data/downloads_v3/source4_banglariceleaf/extracted")
extract_dir.mkdir(parents=True, exist_ok=True)

files = [
    (13461685, "Bacterial Leaf Blight (BLB).rar"),
    (13461686, "Bacterial Leaf Streak (BLS).rar"),
    (13461687, "Healthy Leaf.rar"),
    (13461688, "Leaf Blast.rar"),
    (13461689, "Sheath Blight.rar")
]

def download_rar(fid, fname):
    target = out_dir / fname
    if target.exists() and target.stat().st_size > 10_000_000:
        print(f"Already exists: {fname}")
        return fname, True
    url = f"https://dataverse.harvard.edu/api/access/datafile/{fid}"
    print(f"Starting download: {fname} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(target, "wb") as f:
        f.write(resp.read())
    print(f"Finished download: {fname} ({target.stat().st_size / 1e6:.2f} MB)")
    return fname, True

print("Downloading Harvard Dataverse RAR files...")
with ThreadPoolExecutor(max_workers=5) as ex:
    futs = [ex.submit(download_rar, fid, fname) for fid, fname in files]
    for fut in as_completed(futs):
        fut.result()

print("Extracting RAR files with unar...")
for fid, fname in files:
    rar_path = out_dir / fname
    cmd = ["/opt/homebrew/bin/unar", "-f", "-o", str(extract_dir), str(rar_path)]
    subprocess.run(cmd, check=True)

print("Source 4 Dataverse extraction complete!")
