import urllib.request, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

out_dir = Path("data/downloads_v3/source2_mendeley_znsxdctwtt/images")
out_dir.mkdir(parents=True, exist_ok=True)

api_url = "https://data.mendeley.com/public-api/datasets/znsxdctwtt"
req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    data = json.load(resp)

files = [f for f in data.get("files", []) if f["filename"].endswith(".jpg")]
print(f"Total jpg files in dataset: {len(files)}")

def download_file(f_info):
    fname = f_info["filename"]
    target = out_dir / fname
    if target.exists() and target.stat().st_size == f_info["size"]:
        return fname, True
    dl_url = f_info["content_details"]["download_url"]
    for attempt in range(3):
        try:
            r = urllib.request.Request(dl_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(r, timeout=30) as u, open(target, "wb") as out:
                out.write(u.read())
            return fname, True
        except Exception as e:
            time.sleep(1)
    return fname, False

start = time.time()
completed = 0
failed = []
with ThreadPoolExecutor(max_workers=16) as ex:
    futures = {ex.submit(download_file, f): f["filename"] for f in files}
    for fut in as_completed(futures):
        fname, success = fut.result()
        if success:
            completed += 1
        else:
            failed.append(fname)
        if completed % 100 == 0 or completed == len(files):
            print(f"Downloaded {completed}/{len(files)} files ({time.time() - start:.1f}s)...")

print(f"Source 2 Download complete. Success: {completed}, Failed: {len(failed)}")
if failed:
    print("Failed files:", failed)
