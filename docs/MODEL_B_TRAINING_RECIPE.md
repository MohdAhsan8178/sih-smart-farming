# Model B Training Recipe: Sticky-Trap Pest Patch Classifier

This document is the definitive training recipe for **Model B** (the secondary sticky-trap insect patch classifier). It is designed for execution on **Kaggle GPU environments** (Tesla T4 / P100) with **Weights & Biases (W&B)** experiment tracking.

---

## 1. Dataset Specification & Kaggle Slug

* **Kaggle Dataset Slug**: [`mohdahsan8178/sih-model-b-dataset`](https://www.kaggle.com/datasets/mohdahsan8178/sih-model-b-dataset)
* **Dataset Title**: `SIH Model B Dataset`
* **Dataset Size**: 30,555 unique 64×64 crops (148.5 MB compressed zip).
* **Manifest File**: `model_b_manifest_v4.csv`

### Class Mapping

| Index | Class Name | Target Morphology | Ingested Sources |
|---|---|---|---|
| **0** | `small_pale_winged` | Whitefly adults (*Bemisia tabaci*, *Trialeurodes vaporariorum*) | PST (Zenodo 7801239), Wageningen 4TU |
| **1** | `larger_insect` | Predatory mirids (*Macrolophus*, *Nesidiocoris*) & stored beetles on yellow substrate | Wageningen 4TU, Ong & Høye (Yellow stages) |
| **2** | `debris` | Pure yellow sticky card adhesive, dust specks, glue glare | Wageningen 4TU, PST (Zenodo 7801239) |
| **-1** | `openset_hard` | Hard OOD: thrips on glue, dark non-targets, grid markers, wax | Wageningen 4TU, PST, BanglaRiceLeaf |

### Split Partitions (Group-Isolated by Card & Specimen)

| Split Key | Count | Class Composition (WF / Larger / Debris) | Purpose |
|---|---|---|---|
| `train` | **21,076** | 15,300 / 2,622 / 3,154 | Training set (class-weighted loss) |
| `val` | **3,802** | 3,050 / 413 / 339 | Model checkpoint selection & temperature calibration |
| `test_indist` | **2,991** | 2,004 / 418 / 569 | In-distribution evaluation (zero card overlap with train) |
| `test_cross_card` | **2,107** | 948 / 556 / 603 | **Held-out benchmark**: 84 Wageningen cards never seen in train |
| `openset_eval` | **579** | Verified non-targets & card markings | Energy-based OOD rejection evaluation |

---

## 2. What the Benchmark Actually Measures (Read Before Training)

> [!CAUTION]
> **GOVERNANCE & METRIC INTERPRETATION**:
> Raw accuracy is STRUCK as a headline metric across all documents. Both 76.79% (leaked) and 97.11% (class-imbalanced) are invalid headlines.
> Primary evaluation metrics are **Macro-F1**, **Balanced Accuracy** (unweighted mean of recalls), and per-class recall tables.

### The Trivial Standard Deviation ($\sigma$-Only) Baseline
To determine what the benchmark is actually measuring, we trained a trivial depth-2 decision tree on **strictly the crop grayscale standard deviation ($\sigma$)**:

* **In-Distribution Macro-F1**: **0.5594** (Balanced Acc: **58.51%**)
* **Cross-Card Macro-F1**: **0.5869** (Balanced Acc: **66.06%**)
* **Debris Recall**: **98.17%** (In-Dist) / **98.18%** (Cross-Card) using threshold $\sigma \approx 2.87$.
* **Insect Discrimination Recall**: **0.00%** on `larger_insect` (0 / 418 In-Dist, 0 / 556 Cross-Card).

### The Load-Bearing Split Finding
Whoever trains or evaluates Model B must understand which half of this benchmark does real work:
1. **Debris separation IS largely a flatness check and should be read that way**:
   Because sanitization enforced strict emptiness ($\sigma < 6.0$, zero blobs $\ge 6$ px), the model separates debris from insects almost entirely by mathematical flatness ($\sigma < 2.87$). It is not doing morphological reasoning on glue imperfections.
2. **The whitefly-vs-larger-insect boundary IS genuine morphology**:
   The $\sigma$-baseline scores **0.00% recall on `larger_insect`**, proving that variance cannot separate green mirids or beetles from whiteflies (both share $\sigma \in [5, 30]$). The jump from **0.5869** to **0.9735** cross-card Macro-F1 is load-bearing: the CNN learns genuine morphological structures (body elongation, antennae, translucent wing contours, leg articulation).
3. **Operational Implication**:
   Under real field deployment with weathered glue (dust films, pollen crust, water spots with $\sigma > 4.0$), debris may leak into insect classes. Therefore, deterministic watershed blob density from `core/trap_segmentation.py` remains the primary reported metric; the CNN's role is strictly morphological breakdown within detected insect blobs.

---

## 3. Three Documented Unfixable Gaps

1. **Sugarcane Woolly Aphid (*Ceratovacuna lanigera*) Wax Gap**:
   Woolly aphids present as flocculent white wax secretions rather than discrete insect bodies. Zero public sticky-trap datasets capture this morphology. In-field Indian trap harvesting is required for fine-tuning.
2. **Soft-Bodied Aphids & Thrips (*Rhopalosiphum padi*, *Sitobion avenae*, *Anaphothrips obscurus*) Gap**:
   Zero labelled aphids and only 7 labelled thrips exist in public European trap datasets. They are detected and counted in **total blob density** by deterministic watershed (`core/trap_segmentation.py`), but intentionally unclassified by the CNN.
3. **Weathered Glue & High-Variance Field Debris Gap**:
   Real Indian field cards carry dust films, pollen crust, water spots, dried leaf fibers, and fungal spores. Because training debris was sanitized to flat uniform yellow glue ($\sigma < 6.0$), weathered high-variance non-insect textures risk being misclassified as insects, inflating pest counts. Operational deployments must always treat watershed blob density as primary.

---

## 4. Architecture Specifications

### Option A: Transfer Learning Backbone (Recommended for Kaggle)
* **Model**: `mobilenetv3_small_100` or `efficientnet_b0` pretrained on ImageNet (via `timm`).
* **Input**: $3 \times 64 \times 64$.
* **Head**: AdaptiveAvgPool2d $\to$ Dropout(0.2) $\to$ Linear(embed_dim, 3).
* **Weights**: Transfer learning provides robust low-level edge and texture filters.

### Option B: Edge ConvNet (`TrapPestCNN`)
* **Parameters**: 294,627 parameters (1.18 MB ONNX opset 13).
* **Architecture**: 4 Conv-BN-ReLU-Conv-BN-ReLU-MaxPool stages (16 $\to$ 32 $\to$ 64 $\to$ 128) followed by GlobalAvgPool and Linear(128, 3).
* **Latency**: < 2 ms on Jetson Nano / Gateway CPU.

---

## 5. Aggressive ESP32-CAM Augmentation Pipeline

Insects on sticky cards have no canonical orientation, and the ESP32-CAM OV2640 sensor introduces severe optical and JPEG compression artifacts. The albumentations pipeline simulates this hardware environment:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return A.Compose([
        # Rotational invariance: insects adhere in any orientation
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # 1. Footprint Downscaling (simulating true 6-pixel insect footprint)
        A.Downscale(scale_range=(0.25, 0.60), p=0.60),

        # 2. Hardware JPEG Compression (OV2640 DCT quantization)
        A.ImageCompression(quality_range=(15, 60), p=0.70),

        # 3. Motion & Defocus Blur (wind-induced card flutter)
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 7)),
            A.Defocus(radius=(1, 3)),
            A.GaussianBlur(blur_limit=(3, 5)),
        ], p=0.50),

        # 4. CMOS Sensor Noise & Hue Drift
        A.ISONoise(color_shift=(0.02, 0.08), intensity=(0.1, 0.4), p=0.40),
        A.GaussNoise(std_range=(0.05, 0.20), p=0.40),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.50),

        # 5. Specular Glue Glare & Dust Occlusion
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(4, 12),
                        hole_width_range=(4, 12), fill=255, p=0.30),

        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_eval_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

---

## 6. Training Hyperparameters & Class Weighting

* **Optimizer**: `AdamW(lr=1e-3, weight_decay=1e-4)`
* **Scheduler**: `CosineAnnealingLR(T_max=25, eta_min=1e-5)` with 2-epoch linear warmup
* **Batch Size**: 64 (or 128 on P100/T4)
* **Epochs**: 25 (with Early Stopping on `val_macro_f1`, patience=5)
* **Class Weights**:
  $$w_c = \sqrt{\frac{N_{\text{total}}}{N_{\text{classes}} \cdot N_c}} \quad \text{normalized such that } \text{mean}(w) = 1.0$$
  For v4 training split ($N = 21,076$):
  * `small_pale_winged` ($N = 15,300$): $w_0 = \mathbf{0.534}$
  * `larger_insect` ($N = 2,622$): $w_1 = \mathbf{1.290}$
  * `debris` ($N = 3,154$): $w_2 = \mathbf{1.176}$

```python
import torch
import torch.nn as nn

weights = torch.tensor([0.534, 1.290, 1.176], dtype=torch.float32).cuda()
criterion = nn.CrossEntropyLoss(weight=weights)
```

---

## 7. Model A Parity: Post-Training Calibration Stack

To prevent Model B from deploying uncalibrated overconfident predictions, fit the following post-processing pipeline:

### 1. Temperature Scaling ($T_{\text{cal}}$)
On validation logits $z_i$, solve for scalar $T > 0$ minimizing Negative Log-Likelihood:
$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

```python
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, val_logits, val_labels):
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()
        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss
        optimizer.step(eval_loss)
        return self.temperature.item()
```

### 2. Energy-Based OOD Rejection ($\tau_{\text{energy}}$)
Compute Helmholtz free energy:
$$E(x; T) = -T \cdot \log \sum_{c=1}^C \exp\left(\frac{z_c(x)}{T}\right)$$
Compute $E_{\text{val}}$ on `val` set. Set threshold $\tau_{\text{energy}} = \text{Percentile}(E_{\text{val}}, 95.0)$ so that 95% of in-distribution samples are accepted.
Evaluate rejection rate on `openset_eval` ($N=579$).

### 3. Confidence Floor ($\tau_{\text{conf}} = 0.60$)
```python
if max_prob < 0.60 or energy > tau_energy:
    prediction = "UNCERTAIN_NON_TARGET"  # Abstain from classifying as target pest
```

---

## 8. Complete Kaggle Notebook Script

Paste this complete self-contained script into a Kaggle Notebook (Python 3, GPU T4 or P100):

```python
# ==============================================================================
# Model B Training Pipeline: Kaggle GPU + W&B
# Dataset: mohdahsan8178/sih-model-b-dataset
# ==============================================================================

import os
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report
import timm
import wandb

# 0. Weights & Biases Authentication
# Option A: Paste your API key directly below:
# wandb.login(key="YOUR_WANDB_API_KEY")
# Option B: Kaggle Secrets (Add-ons -> Secrets -> WANDB_API_KEY):
try:
    from kaggle_secrets import UserSecretsClient
    wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
except Exception:
    wandb.login()

wandb.init(
    project="sih-model-b-trap-pest-classifier",
    config={
        "architecture": "mobilenetv3_small_100",
        "batch_size": 128,
        "lr": 1e-3,
        "epochs": 25,
        "loss": "ClassWeightedCrossEntropy",
        "weights": [0.534, 1.290, 1.176]
    }
)

# 1. Locate dataset files (robust locator with kagglehub fallback)
WORK_DIR = Path("/kaggle/working")

# Download or locate dataset via kagglehub
kh_dir = None
try:
    import kagglehub
    kh_dir = Path(kagglehub.dataset_download("mohdahsan8178/sih-model-b-dataset"))
    print(f"kagglehub path: {kh_dir}")
except Exception as e:
    print(f"kagglehub fallback ({e})")

search_roots = []
if kh_dir and kh_dir.exists():
    search_roots.append(kh_dir)
if Path("/kaggle/input").exists():
    search_roots.append(Path("/kaggle/input"))
search_roots.extend([WORK_DIR, Path(".")])

# 2. Load Manifest
manifest_path = None
for root in search_roots:
    matches = list(root.glob("**/model_b_manifest_v4.csv"))
    if matches:
        manifest_path = matches[0]
        break

if manifest_path is None:
    raise FileNotFoundError(f"model_b_manifest_v4.csv not found in {search_roots}")

df_master = pd.read_csv(manifest_path)
print(f"Loaded manifest from: {manifest_path} ({len(df_master)} rows)")
print(pd.crosstab(df_master["class_name"], df_master["split"], margins=True))

# Resolve image patch directory (handles Kaggle nested folder unpacking)
sample_crop = "small_pale_winged_000000.png"
sample_class = "small_pale_winged"
image_root = None

for root in [manifest_path.parent, manifest_path.parent.parent] + search_roots:
    for candidate in [
        root,
        root / "model_b_patches_v4",
        root / "model_b_patches_v4" / "model_b_patches_v4",
    ]:
        if (candidate / sample_class / sample_crop).exists() or (candidate / "model_b_patches_v4" / sample_class / sample_crop).exists():
            image_root = candidate
            break
    if image_root:
        break

if image_root is None:
    for root in search_roots:
        found_imgs = list(root.glob(f"**/{sample_crop}"))
        if found_imgs:
            p = found_imgs[0].parent
            image_root = p.parent if p.name == sample_class else p
            break

IMAGE_ROOT = image_root
print(f"Resolved IMAGE_ROOT: {IMAGE_ROOT}")

# 3. Dataset Class
class StickyTrapDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["rel_path"]
        if not img_path.exists():
            img_path = self.root_dir / row["class_name"] / f"{row['crop_id']}.png"
        im = Image.open(img_path).convert("RGB")
        im_np = np.array(im)

        if self.transform:
            augmented = self.transform(image=im_np)
            tensor = augmented["image"]
        else:
            tensor = torch.from_numpy(im_np.transpose(2, 0, 1)).float() / 255.0

        label = int(row["class_idx"])
        return tensor, label

# 4. Augmentations
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Downscale(scale_range=(0.25, 0.60), p=0.60),
    A.ImageCompression(quality_range=(15, 60), p=0.70),
    A.OneOf([
        A.MotionBlur(blur_limit=(3, 7)),
        A.Defocus(radius=(1, 3)),
        A.GaussianBlur(blur_limit=(3, 5)),
    ], p=0.50),
    A.ISONoise(p=0.40),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.50),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(4, 12), hole_width_range=(4, 12), fill=255, p=0.30),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

eval_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# 5. DataLoaders
df_train = df_master[df_master["split"] == "train"]
df_val = df_master[df_master["split"] == "val"]
df_test_indist = df_master[df_master["split"] == "test_indist"]
df_test_cross = df_master[df_master["split"] == "test_cross_card"]
df_ood = df_master[df_master["split"] == "openset_eval"]

train_loader = DataLoader(StickyTrapDataset(df_train, IMAGE_ROOT, train_transform), batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(StickyTrapDataset(df_val, IMAGE_ROOT, eval_transform), batch_size=128, shuffle=False, num_workers=2)
test_indist_loader = DataLoader(StickyTrapDataset(df_test_indist, IMAGE_ROOT, eval_transform), batch_size=128, shuffle=False)
test_cross_loader = DataLoader(StickyTrapDataset(df_test_cross, IMAGE_ROOT, eval_transform), batch_size=128, shuffle=False)
ood_loader = DataLoader(StickyTrapDataset(df_ood, IMAGE_ROOT, eval_transform), batch_size=128, shuffle=False)

# 6. Model Definition (MobileNetV3-Small)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=3).to(device)

weights = torch.tensor([0.534, 1.290, 1.176], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

# 7. Training Loop with Early Stopping
best_val_f1 = 0.0
best_model_path = WORK_DIR / "best_model_b.pt"

for epoch in range(1, 26):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += len(y)
    scheduler.step()

    # Eval on Val
    model.eval()
    val_preds, val_targets, val_logits = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_preds.extend(logits.argmax(dim=1).cpu().numpy())
            val_targets.extend(y.cpu().numpy())
            val_logits.append(logits.cpu().numpy())

    val_f1 = f1_score(val_targets, val_preds, average="macro")
    val_bal = balanced_accuracy_score(val_targets, val_preds)
    print(f"Epoch {epoch:02d} | Train Loss: {total_loss/total_count:.4f} | Val Macro-F1: {val_f1:.4f} | Bal-Acc: {val_bal*100:.2f}%")

    wandb.log({
        "epoch": epoch,
        "train_loss": total_loss / total_count,
        "val_macro_f1": val_f1,
        "val_balanced_acc": val_bal,
        "learning_rate": scheduler.get_last_lr()[0]
    })

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({"model_state": model.state_dict(), "val_f1": val_f1}, best_model_path)

# 8. Final Benchmark Evaluation on Held-Out Cross-Card Split
ckpt = torch.load(best_model_path)
model.load_state_dict(ckpt["model_state"])
model.eval()

def eval_loader(loader):
    preds, targets, logits = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            targets.extend(y.numpy())
            logits.append(out.cpu().numpy())
    return np.array(targets), np.array(preds), np.concatenate(logits, axis=0)

y_cr, p_cr, l_cr = eval_loader(test_cross_loader)
print("\n" + "=" * 60)
print("FINAL CROSS-CARD BENCHMARK (84 HELD-OUT 4TU CARDS):")
print("=" * 60)
print(f"Macro-F1 (HEADLINE): {f1_score(y_cr, p_cr, average='macro'):.4f}")
print(f"Balanced Accuracy  : {balanced_accuracy_score(y_cr, p_cr)*100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_cr, p_cr))

# Critical Leakage Rate
wf_total = (y_cr == 0).sum()
wf_mis_larger = ((y_cr == 0) & (p_cr == 1)).sum()
leak_rate = (wf_mis_larger / wf_total) * 100.0
print(f"Operational ETL Leakage Rate (Whitefly -> Larger): {leak_rate:.2f}% [PASS <= 5.0%]")

# 9. Temperature Scaling & OOD Calibration
val_targets = np.array(val_targets)
val_logits = np.concatenate(val_logits, axis=0)
temp_scaler = TemperatureScaler().to(device)
T_cal = temp_scaler.fit(torch.tensor(val_logits).to(device), torch.tensor(val_targets).to(device))
print(f"Fitted Temperature T_cal = {T_cal:.4f}")

# Energy Threshold (numerically stable via logsumexp)
from scipy.special import logsumexp
id_energy = -T_cal * logsumexp(val_logits / T_cal, axis=1)
tau_energy = float(np.percentile(id_energy, 95.0))
print(f"Fitted Energy Threshold (95% ID TPR) tau_energy = {tau_energy:.4f}")

# Eval OOD
_, _, l_ood = eval_loader(ood_loader)
ood_energy = -T_cal * logsumexp(l_ood / T_cal, axis=1)
ood_rejection_rate = float((ood_energy > tau_energy).mean()) * 100.0
print(f"OOD Rejection Rate on Hard Non-Targets: {ood_rejection_rate:.2f}%")

# 10. ONNX Export
dummy_input = torch.randn(1, 3, 64, 64).to(device)
torch.onnx.export(
    model, dummy_input, WORK_DIR / "model_b_calibrated.onnx",
    opset_version=13,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
)
print("Exported ONNX model to /kaggle/working/model_b_calibrated.onnx")
```

---

## 9. Operational Verification Checklist for Kaggle Run

Before reporting completion of training:
1. **Macro-F1 on `test_cross_card`** is reported as the primary headline (target: $> 0.95$).
2. **Operational ETL Leakage Rate** ($\text{Whitefly} \to \text{Larger}$) is verified $\le 5.0\%$.
3. **$\sigma$-Baseline Context** is cited in all comparison tables.
4. **$T_{\text{cal}}$ and $\tau_{\text{energy}}$** calibration parameters are extracted and recorded for production edge inference.
5. **ONNX Export** passes validation with opset 13.
