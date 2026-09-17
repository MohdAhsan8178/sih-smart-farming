import csv
import json
import shutil
from pathlib import Path
import pandas as pd

def main():
    rename_df = pd.read_csv("data/packaged_min/splits_packaged/rename_map.csv")
    raw_to_pkg = dict(zip(rename_df["old_relative_path"], rename_df["new_relative_path"]))
    for p in rename_df["new_relative_path"]:
        if p not in raw_to_pkg:
            raw_to_pkg[p] = p

    src_dir = Path("splits_v3")
    target_dirs = [
        Path("data/packaged_min/splits_v3"),
        Path("splits_v3_packaged")
    ]
    for d in target_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Process all split files
    for src_file in sorted(src_dir.iterdir()):
        if src_file.suffix == ".csv":
            df = pd.read_csv(src_file)
            if "path" in df.columns:
                new_paths = []
                missing_count = 0
                for p in df["path"]:
                    rel = p.replace("data/raw/", "")
                    new_rel = raw_to_pkg.get(rel, rel)
                    packaged_path = f"data/packaged_min/{new_rel}"
                    if not Path(packaged_path).is_file():
                        missing_count += 1
                        print(f"MISSING: {packaged_path}")
                    new_paths.append(packaged_path)
                assert missing_count == 0, f"Found {missing_count} missing files for {src_file.name}!"
                df["path"] = new_paths
                for d in target_dirs:
                    df.to_csv(d / src_file.name, index=False)
                print(f"Wrote {src_file.name}: {len(df)} rows, 0 missing files.")
            else:
                for d in target_dirs:
                    shutil.copy2(src_file, d / src_file.name)
                print(f"Copied non-path CSV: {src_file.name}")
        elif src_file.suffix == ".json":
            for d in target_dirs:
                shutil.copy2(src_file, d / src_file.name)
            print(f"Copied JSON: {src_file.name}")

    print("splits_v3 packaging complete and verified!")

if __name__ == "__main__":
    main()
