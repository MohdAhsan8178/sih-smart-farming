"""
Package Model B v4 dataset for Kaggle publication:
1. Zips model_b_patches_v4 directory containing all 30,555 patches.
2. Writes Kaggle-compatible manifest with root-relative paths.
3. Generates dataset-metadata.json and comprehensive README.md (Dataset Card).
"""

import json
import shutil
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = ROOT / "kaggle" / "model_b_package"
PATCHES_DIR = ROOT / "data" / "processed" / "model_b_patches_v4"
MANIFEST_SRC = ROOT / "splits" / "model_b_manifest_v4.csv"

PKG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Zip model_b_patches_v4
zip_path = PKG_DIR / "model_b_patches_v4.zip"
print(f"[package] Creating {zip_path} from {PATCHES_DIR}...")

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for file_path in PATCHES_DIR.rglob("*.png"):
        arcname = file_path.relative_to(PATCHES_DIR.parent)
        zf.write(file_path, arcname=str(arcname))

print(f"[package] Created {zip_path} (size: {zip_path.stat().st_size / (1024*1024):.2f} MB)")

# 2. Kaggle-compatible manifest
df = pd.read_csv(MANIFEST_SRC)
# Normalize rel_path to be relative to Kaggle dataset root (model_b_patches_v4/...)
df["rel_path"] = df["class_name"].apply(lambda c: f"model_b_patches_v4/{c}") + "/" + df["crop_id"] + ".png"
manifest_out = PKG_DIR / "model_b_manifest_v4.csv"
df.to_csv(manifest_out, index=False)
print(f"[package] Wrote {manifest_out} ({len(df)} rows)")

# 3. Metadata
metadata = {
    "title": "SIH Model B Dataset",
    "id": "mohdahsan8178/sih-model-b-dataset",
    "licenses": [
        {"name": "CC-BY-4.0"}
    ]
}
with open(PKG_DIR / "dataset-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# 4. Dataset Card (README.md)
readme_content = """# SIH Model B: Sticky-Trap Pest Patch Classification Dataset (v4)

## 1. Overview & Problem Definition
This dataset provides **30,555 group-isolated 64×64 image patches** for training and evaluating patch-classification models (Model B) deployed on smart agricultural sticky-trap monitoring stations (ESP32-CAM / Jetson Nano).

The primary operational count is performed deterministically by watershed blob segmentation on yellow chromotropic sticky cards. Model B performs secondary morphological classification on extracted insect patches.

---

## 2. Morphological 3-Class Taxonomy

| Class ID | Class Name | Morphology & Target Species | Sources |
|---|---|---|---|
| **0** | `small_pale_winged` | Whitefly adults (*Bemisia tabaci*, *Trialeurodes vaporariorum*) on yellow adhesive. | PST (Zenodo 7801239), Wageningen 4TU |
| **1** | `larger_insect` | Non-target insects: green predatory mirids (*Macrolophus pygmaeus*, *Nesidiocoris tenuis*) and stored-product beetles on yellow substrates. | Wageningen 4TU, Ong & Høye (Yellow stages) |
| **2** | `debris` | Pure yellow sticky card adhesive, dust specks, glue glare, and non-insect substrate imperfections. | Wageningen 4TU, PST (Zenodo 7801239) |
| **-1** | `openset_hard` | Hard Out-of-Distribution (OOD): thrips on glue, incidental dark insects, printed card grid markers, punch-holes, agricultural wax. | Wageningen 4TU, PST, BanglaRiceLeaf |

---

## 3. Split Distribution & Group Leakage Governance

Partitions enforce **Stratified Group-Isolation**; no sticky card or specimen group spans train, validation, or test sets.

| Class Name | Train (In-Dist) | Val (In-Dist) | Test In-Dist | Test Cross-Card (84 4TU Cards) | Open-Set Hard (OOD) | Total Unique Crops |
|---|---|---|---|---|---|---|
| **small_pale_winged** | **15,300** | 3,050 | 2,004 | 948 | 0 | **21,302** |
| **larger_insect** | **2,622** | 413 | 418 | 556 | 0 | **4,009** |
| **debris** | **3,154** | 339 | 569 | 603 | 0 | **4,665** |
| **openset_hard** | 0 | 0 | 0 | 0 | **579** | **579** |
| **Total** | **21,076** | **3,802** | **2,991** | **2,107** | **579** | **30,555** |

* Note: The 84 Wageningen 4TU test cards (`test_cross_card`) are held out completely from the training distribution to benchmark cross-card generalization.

---

## 4. Per-Source Attribution & Licences

1. **Wageningen 4TU Sticky Trap Dataset**:
   - Citation: Nieuwenhuizen et al., Wageningen University & Research.
   - DOI: [10.4121/uuid:8b8ba63a-1010-4de7-a7fb-6f9e3baf128e](https://doi.org/10.4121/uuid:8b8ba63a-1010-4de7-a7fb-6f9e3baf128e)
   - Licence: **CC BY 4.0**
2. **PST (Pest Sticky Trap) Dataset**:
   - Citation: Zenodo Repository 7801239.
   - DOI: [10.5281/zenodo.7801239](https://doi.org/10.5281/zenodo.7801239)
   - Licence: **CC BY 4.0**
3. **Ong & Høye Invertebrates Dataset**:
   - Citation: Ong, X.L. & Høye, T.T., Figshare.
   - DOI: [10.6084/m9.figshare.23617383.v2](https://doi.org/10.6084/m9.figshare.23617383.v2)
   - Licence: **CC BY 4.0** (Yellow substrate stages `pYellow`, `RPYellow`, `LYellow` ingested)
4. **GinJinn2 Benchmark**:
   - Citation: BGBM Dahlem.
   - DOI: [10.34656/41pk-rn18.1](https://doi.org/10.34656/41pk-rn18.1)
   - Licence: **CC0 1.0** (Verified duplicate subset of Wageningen 4TU; absorbed via 4TU).

---

## 5. Three Documented Unfixable Gaps (Read Before Training)

1. **Sugarcane Woolly Aphid (*Ceratovacuna lanigera*) Wax Gap**:
   - Woolly aphids produce white flocculent wax clumps rather than discrete insect bodies. Zero public sticky-trap datasets capture this morphology.
   - Segmented as dark/pale blobs by watershed; classified as debris or whitefly by the CNN. Requires in-field Indian trap harvesting.
2. **Soft-Bodied Aphids & Thrips (*Rhopalosiphum padi*, *Sitobion avenae*, *Anaphothrips obscurus*) Gap**:
   - Public European datasets contain zero labelled aphids and only 7 labelled thrips.
   - Detected and counted in total blob density by watershed, but intentionally unclassified into a dedicated pest taxon by Model B.
3. **Weathered Glue & High-Variance Field Debris Gap**:
   - Real field cards carry dust films, pollen crust, water spots, dried leaf fibers, and fungal spores.
   - To eliminate label contamination from unannotated insects, the v4 debris class enforced strict flatness (mean $S \\ge 95, B < 75, \\sigma < 6.0$, zero blobs $\\ge 6$ px).
   - A trivial decision tree trained strictly on grayscale standard deviation ($\\sigma$) achieves 0.5594 / 0.5869 Macro-F1 by isolating debris at $\\sigma \\approx 2.87$.
   - **Operational consequence**: Because debris is clean and flat, high-variance weathered field textures risk being misclassified as insects, inflating pest counts. In-field deployment must rely on total watershed blob density as the primary metric.

---

## 6. Honest Baseline Benchmark

* **Discipline Notice**: Both 76.79% (leaked) and 97.11% (class-imbalanced) raw accuracy figures are **STRUCK**. Primary metrics are Macro-F1 and Balanced Accuracy.
* **Trivial $\\sigma$-Only Baseline (Depth-2 Decision Tree on Gray Std)**:
  - In-Dist Macro-F1: **0.5594** (Balanced Acc: 58.51%)
  - Cross-Card Macro-F1: **0.5869** (Balanced Acc: 66.06%)
* **TrapPestCNN Baseline (Scratch ConvNet)**:
  - In-Dist Macro-F1 (HEADLINE): **0.9911** (Balanced Acc: 99.23%)
  - Cross-Card Macro-F1 (HEADLINE): **0.9735** (Balanced Acc: 97.59%)
  - Clean ETL Leakage ($WF \\to Larger$): **0.30%** (In-Dist) / **3.16%** (Cross-Card) [PASS $\\le 5.0\\%$]
"""

with open(PKG_DIR / "README.md", "w") as f:
    f.write(readme_content)

print("[package] Wrote README.md (Dataset Card)")
print("[package] Package directory prepared successfully.")
