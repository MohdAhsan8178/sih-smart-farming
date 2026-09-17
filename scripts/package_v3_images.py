"""
Package the 6,021 new V3 images into data/packaged_min/.
Rules:
- Resize to 1024px longest edge (LANCZOS, aspect ratio preserved).
- Images with max dimension <= 1024 are NOT upscaled.
- EXIF orientation baked in via ImageOps.exif_transpose and all EXIF tags stripped.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps


def process_one_image(rel_path_str: str) -> bool:
    raw_path = Path("data/raw") / rel_path_str
    pkg_path = Path("data/packaged_min") / rel_path_str

    pkg_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(raw_path) as img:
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        if max(w, h) > 1024:
            scale = 1024.0 / max(w, h)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Save as JPEG without EXIF metadata
        img.save(pkg_path, format="JPEG", quality=95)

    return True


def main():
    rename_map_path = Path("data/packaged_min/splits_packaged/rename_map.csv")
    df = pd.read_csv(rename_map_path)
    new_entries = df.iloc[-6021:]
    print(f"Packaging {len(new_entries)} images into data/packaged_min/...")

    paths_to_process = new_entries["new_relative_path"].tolist()

    workers = min(os.cpu_count() or 4, 8)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_one_image, paths_to_process, chunksize=50))

    assert len(results) == 6021
    assert all(results)
    print(f"Successfully packaged all {len(results)} images into data/packaged_min/.")


if __name__ == "__main__":
    main()
