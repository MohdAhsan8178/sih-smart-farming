"""
Training script for Model A (Stage 1 Baseline & Stage 2 Full Model).
Reference: ULTIMATE_IMPLEMENTATION_PLAN_1.md STEP 12.
"""

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import warnings

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None  # type: ignore
    HAS_WANDB = False

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from timm.utils import ModelEmaV2

import configs.train_config as train_cfg
from configs.classes import CLASS_NAMES
from configs.paths import CKPT, REPORTS, ROOT, SPLITS
from configs.train_config import (
    BATCH_SIZE,
    EMA_DECAY,
    EPOCHS,
    GRAD_CLIP,
    IMAGE_SIZE,
    LABEL_SMOOTH,
    LR_BACKBONE,
    LR_HEAD,
    SEED,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from train.dataset import PlantDataset, build_loaders, seed_worker
from train.model import build_model, get_parameter_groups
from train.transforms import eval_transform


def set_seed(seed: int = SEED, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_git_commit_hash() -> Optional[str]:
    """Attempts to retrieve the current git commit hash, returning None on failure."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return None


def generate_wandb_id() -> str:
    """Generates an alphanumeric run ID for W&B tracking."""
    try:
        if HAS_WANDB and wandb is not None and hasattr(wandb, "util") and hasattr(wandb.util, "generate_id"):
            return str(wandb.util.generate_id())
    except Exception:
        pass
    return uuid.uuid4().hex[:8]


def get_checkpoint_wandb_id(checkpoint_path: Union[str, Path]) -> Optional[str]:
    """Retrieves wandb_run_id from checkpoint state dict if present."""
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu")
        return ckpt.get("wandb_run_id")
    except Exception:
        return None


def init_wandb(
    args: argparse.Namespace,
    config: Dict[str, Any],
    run_id: Optional[str] = None,
) -> Optional[str]:
    """
    Initializes W&B with graceful degradation.
    Returns active wandb run ID if successfully initialized, or None if disabled/failed.
    """
    if getattr(args, "no_wandb", False):
        print("[train_model_a] W&B disabled via --no_wandb.")
        return None

    if not HAS_WANDB or wandb is None:
        print("[train_model_a] wandb package not installed; proceeding with W&B disabled.")
        return None

    try:
        init_kwargs: Dict[str, Any] = {
            "entity": getattr(args, "wandb_entity", "prism-team"),
            "project": getattr(args, "wandb_project", "SIH"),
            "config": config,
            "resume": "allow",
        }
        if run_id is not None:
            init_kwargs["id"] = run_id

        run = wandb.init(**init_kwargs)
        active_id = getattr(run, "id", None) if run is not None else run_id
        return active_id or run_id
    except Exception as e:
        print(f"[train_model_a] W&B init failed ({e}); proceeding with W&B disabled.")
        return None


def safe_wandb_log(data: Dict[str, Any], step: Optional[int] = None) -> None:
    """Safely logs data to W&B without interrupting training on failure."""
    try:
        if HAS_WANDB and wandb is not None and getattr(wandb, "run", None) is not None:
            if step is not None:
                wandb.log(data, step=step)
            else:
                wandb.log(data)
    except Exception as e:
        print(f"[train_model_a] W&B log error ignored: {e}")


def safe_wandb_summary(data: Dict[str, Any]) -> None:
    """Safely updates W&B summary metrics without interrupting training on failure."""
    try:
        if HAS_WANDB and wandb is not None and getattr(wandb, "run", None) is not None:
            for k, v in data.items():
                wandb.run.summary[k] = v
    except Exception as e:
        print(f"[train_model_a] W&B summary update error ignored: {e}")


def safe_wandb_finish() -> None:
    """Safely finalizes W&B run."""
    try:
        if HAS_WANDB and wandb is not None and getattr(wandb, "run", None) is not None:
            wandb.finish()
    except Exception as e:
        print(f"[train_model_a] W&B finish error ignored: {e}")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    ema: Optional[ModelEmaV2],
    device: torch.device,
    use_amp: bool = False,
    max_steps: Optional[int] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    global_step_start: int = 0,
    base_decay: float = EMA_DECAY,
    use_ema_warmup: bool = True,
) -> Tuple[float, List[float]]:
    model.train()
    total_loss = 0.0
    step_losses: List[float] = []
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for step, (images, targets) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        if GRAD_CLIP > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            if use_ema_warmup:
                current_step = global_step_start + step
                # EMA warmup: smoothly ramp decay up so initial random weights are rapidly flushed
                ema.decay = min(base_decay, (1.0 + current_step) / (10.0 + current_step))
            else:
                ema.decay = base_decay
            ema.update(model)

        loss_val = float(loss.item())
        total_loss += loss_val
        step_losses.append(loss_val)

        safe_wandb_log({
            "train/step_loss": loss_val,
            "train/lr": float(optimizer.param_groups[0]["lr"]),
        })

    n_steps = max(1, len(step_losses))
    return total_loss / n_steps, step_losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    max_steps: Optional[int] = None,
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    for step, (images, targets) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()

        all_preds.extend(preds)
        all_targets.extend(targets.numpy().tolist())

    if not all_targets:
        return 0.0, 0.0, {c: 0.0 for c in class_names}

    top1 = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average='macro', zero_division=0))

    # Per-class recall mapped dynamically
    num_classes = len(class_names)
    recalls = recall_score(
        all_targets,
        all_preds,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )
    per_class_recall = {
        class_names[i]: float(recalls[i]) for i in range(num_classes)
    }

    return macro_f1, top1, per_class_recall


def get_train_config_dict(args: Optional[argparse.Namespace] = None) -> Dict[str, object]:
    """
    Extracts all hyperparameters from configs/train_config.py, updated with any
    active CLI overrides passed via args.
    """
    cfg_dict = {
        k: getattr(train_cfg, k)
        for k in dir(train_cfg)
        if k.isupper() and not k.startswith("__")
    }
    if args is not None:
        cli_keys = {
            "stage": args.stage,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr_head": args.lr_head,
            "lr_backbone": args.lr_backbone,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "data_dir": args.data_dir,
            "pretrained": args.pretrained,
            "deterministic": getattr(args, "deterministic", False),
        }
        cfg_dict.update({k.upper(): v for k, v in cli_keys.items()})
    return cfg_dict


def save_checkpoint_atomic(state: dict, filepath: Union[str, Path]) -> None:
    """
    Atomically saves a checkpoint dictionary.
    Writes to a unique temporary file in the same directory, flushes and syncs to disk,
    then atomically replaces the destination path. If interrupted or an exception is raised,
    the target destination file is untouched.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.parent / f".tmp_{filepath.name}_{os.getpid()}_{random.randint(100000, 999999)}"
    try:
        with open(tmp_path, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def resume_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    ema: Optional[ModelEmaV2] = None,
    expected_total_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, float]:
    """
    Resumes training state from a checkpoint.
    Validates total_steps if expected_total_steps is provided.
    Restores model, optimizer, scheduler, scaler, and ema state dicts.

    Returns:
        Tuple of (start_epoch, best_macro_f1).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint for resume not found at: {path}")

    map_loc = device if device is not None else torch.device("cpu")
    ckpt = torch.load(path, map_location=map_loc)

    ckpt_total_steps = ckpt.get("total_steps")
    if expected_total_steps is not None and ckpt_total_steps is not None:
        if ckpt_total_steps != expected_total_steps:
            raise ValueError(
                f"Mismatched total_steps between checkpoint ({ckpt_total_steps}) and current run ({expected_total_steps}). "
                f"Cannot resume OneCycleLR schedule with mismatched total steps."
            )

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    if ema is not None and ckpt.get("ema_state_dict") is not None:
        ema.load_state_dict(ckpt["ema_state_dict"])

    start_epoch = int(ckpt["epoch"]) + 1
    best_macro_f1 = float(ckpt.get("best_macro_f1", ckpt.get("val_macro_f1", -1.0)))

    return start_epoch, best_macro_f1


def load_class_weights(
    weights_path: Union[str, Path] = SPLITS / "class_weights.json",
    class_names: List[str] = CLASS_NAMES,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Loads raw inverse-frequency class weights from splits/class_weights.json,
    validates that the file contains exactly the requested class_names (raising
    descriptive ValueError if any keys are missing or unexpected), validates that
    no weights are zero, negative, infinite or NaN, reorders the weights to match
    class_names exactly (never relying on file order), applies the square-root
    transform:
        w_c = sqrt(raw_inverse_weight_c)
    and normalises the resulting vector so that mean(w) == 1.0.
    Returns the weight tensor on the specified device.
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Class weights file not found at: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Extract class-to-weight dictionary (strictly expecting split.py output schema)
    if "raw_inverse_weights" not in data:
        raise KeyError(
            f"Malformed class weights file at {path}: expected 'raw_inverse_weights' top-level key."
        )
    raw_map = data["raw_inverse_weights"]

    # Strict key set validation: missing keys and extra keys both fail loudly
    expected_set = set(class_names)
    actual_set = set(raw_map.keys())
    missing_keys = expected_set - actual_set
    extra_keys = actual_set - expected_set

    if missing_keys or extra_keys:
        err_msg = []
        if missing_keys:
            err_msg.append(f"Missing classes: {sorted(missing_keys)}")
        if extra_keys:
            err_msg.append(f"Extra unexpected classes: {sorted(extra_keys)}")
        raise ValueError(f"Class weights key mismatch in {path}! {'; '.join(err_msg)}")

    # Extract values strictly in class_names order and validate numerical sanity
    raw_vals = []
    for c in class_names:
        v = raw_map[c]
        if isinstance(v, dict):
            v = v.get("raw_inverse_weight", v.get("weight"))
        if v is None or not np.isfinite(v) or v <= 0:
            raise ValueError(f"Class '{c}' has invalid or zero/infinite raw weight: {v}")
        raw_vals.append(float(v))

    raw_arr = np.array(raw_vals, dtype=np.float32)

    # Apply policy: sqrt transform and mean normalisation
    sqrt_arr = np.sqrt(raw_arr)
    mean_sqrt = np.mean(sqrt_arr)
    norm_arr = sqrt_arr / mean_sqrt

    weight_tensor = torch.tensor(norm_arr, dtype=torch.float32, device=device)
    return weight_tensor


def format_lowest_recall_classes(per_class_recall: Dict[str, float], n: int = 5) -> str:
    """
    Formats the n lowest-recall classes and their values into a compact single line.
    """
    sorted_items = sorted(per_class_recall.items(), key=lambda x: (x[1], x[0]))
    worst_n = sorted_items[:n]
    return " | ".join(f"{cls}={rec:.4f}" for cls, rec in worst_n)


def compute_split_metrics(
    all_targets: List[int],
    all_preds: List[int],
    class_names: List[str],
) -> Tuple[float, float, float, Dict[str, Dict[str, Optional[float]]], np.ndarray]:
    """
    Computes top-1 accuracy, macro-F1 (across classes with support > 0),
    micro-F1, per-class metrics (precision, recall, f1, support) for all classes,
    and the 29x29 confusion matrix.

    Classes with zero support in a split report null (None in Python) for
    precision, recall, and f1, never 0.0.
    """
    num_classes = len(class_names)
    targets_arr = np.array(all_targets, dtype=np.int64)
    preds_arr = np.array(all_preds, dtype=np.int64)

    if len(targets_arr) == 0:
        per_class_empty = {
            c: {"precision": None, "recall": None, "f1": None, "support": 0}
            for c in class_names
        }
        cm_empty = np.zeros((num_classes, num_classes), dtype=np.int64)
        return 0.0, 0.0, 0.0, per_class_empty, cm_empty

    top1 = float(accuracy_score(targets_arr, preds_arr))
    micro_f1 = float(f1_score(targets_arr, preds_arr, average='micro', zero_division=0))

    cm = confusion_matrix(targets_arr, preds_arr, labels=list(range(num_classes)))

    p, r, f, s = precision_recall_fscore_support(
        targets_arr,
        preds_arr,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    per_class_metrics: Dict[str, Dict[str, Optional[float]]] = {}
    present_f1s: List[float] = []

    for i, c in enumerate(class_names):
        supp = int(s[i])
        if supp == 0:
            per_class_metrics[c] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "support": 0,
            }
        else:
            prec_val = float(p[i])
            rec_val = float(r[i])
            f1_val = float(f[i])
            per_class_metrics[c] = {
                "precision": prec_val,
                "recall": rec_val,
                "f1": f1_val,
                "support": supp,
            }
            present_f1s.append(f1_val)

    macro_f1 = float(np.mean(present_f1s)) if present_f1s else 0.0

    return top1, macro_f1, micro_f1, per_class_metrics, cm


def build_test_loader(
    csv_path: Union[str, Path],
    class_to_idx: Dict[str, int],
    batch_size: int = BATCH_SIZE,
    root_dir: Optional[Union[str, Path]] = None,
    path_prefix_strip: str = "",
    num_workers: int = 0,
    seed: int = SEED,
) -> DataLoader:
    """
    Builds DataLoader for evaluation on test splits using eval_transform identical to validation.
    """
    tf = eval_transform(size=IMAGE_SIZE)
    dataset = PlantDataset(
        data_source=csv_path,
        transform=tf,
        class_to_idx=class_to_idx,
        is_train=False,
        slicing_factor=0,
        enforce_manifest_agreement=False,
        root_dir=root_dir,
        path_prefix_strip=path_prefix_strip,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def load_checkpoint_for_eval(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    eval_weights: str = "auto",
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, str]:
    """
    Loads weights from checkpoint for evaluation.
    eval_weights: 'auto', 'ema', or 'model'.
    Returns (model, weights_used).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    map_loc = device if device is not None else torch.device("cpu")
    ckpt = torch.load(path, map_location=map_loc)

    has_ema = ckpt.get("ema_state_dict") is not None
    if eval_weights == "ema" and not has_ema:
        raise ValueError(f"Requested eval_weights='ema' but checkpoint {path} does not contain 'ema_state_dict'.")

    use_ema = (eval_weights == "ema") or (eval_weights == "auto" and has_ema)

    if use_ema:
        ema_sd = ckpt["ema_state_dict"]
        cleaned_sd = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in ema_sd.items()
        }
        model.load_state_dict(cleaned_sd)
        weights_used = "ema"
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        weights_used = "model_state_dict"

    model.eval()
    return model, weights_used


@torch.no_grad()
def evaluate_test_splits(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    class_names: List[str],
    splits_dict: Optional[Dict[str, Union[str, Path]]] = None,
    splits_dir: Optional[Union[str, Path]] = None,
    output_report_path: Union[str, Path] = REPORTS / "stage1_test_metrics.json",
    reports_dir: Union[str, Path] = REPORTS,
    logits_dir: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    path_prefix_strip: str = "",
    batch_size: int = BATCH_SIZE,
    device: Optional[torch.device] = None,
    weights_used: str = "unknown",
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluates each test split separately (never averaged or concatenated).
    Computes top-1, macro-F1, micro-F1, per-class precision/recall/F1/support for all classes.
    Saves 29x29 confusion matrices to disk.
    Saves raw float32 pre-softmax logits and parallel paths JSON to artifacts/logits/.
    Writes full evaluation summary to stage1_test_metrics.json.
    """
    splits_dir_supplied = (splits_dir is not None)
    base_splits = Path(splits_dir) if splits_dir_supplied else SPLITS

    if splits_dict is None:
        splits_dict = {
            "val": base_splits / "val.csv",
            "test_indist": base_splits / "test_indist.csv",
        }
        if (base_splits / "test_sourceheldout.csv").exists():
            splits_dict["test_sourceheldout"] = base_splits / "test_sourceheldout.csv"
        if (base_splits / "test_crossdomain.csv").exists():
            splits_dict["test_crossdomain"] = base_splits / "test_crossdomain.csv"
        if "test_sourceheldout" not in splits_dict and "test_crossdomain" not in splits_dict:
            if "v3" in str(base_splits).lower():
                splits_dict["test_sourceheldout"] = base_splits / "test_sourceheldout.csv"
            else:
                splits_dict["test_crossdomain"] = base_splits / "test_crossdomain.csv"
        splits_dict["test_external_riceblast"] = base_splits / "test_external_riceblast.csv"

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(output_report_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if logits_dir is None:
        resolved_logits_dir = reports_dir.parent / "logits"
    else:
        resolved_logits_dir = Path(logits_dir)
    resolved_logits_dir.mkdir(parents=True, exist_ok=True)

    dev = device if device is not None else torch.device("cpu")
    model.eval()

    results: Dict[str, Any] = {}

    for split_key, split_path in splits_dict.items():
        path = Path(split_path)
        if not path.exists():
            if splits_dir_supplied:
                raise FileNotFoundError(
                    f"[evaluate_test_splits] Explicit splits_dir was provided ('{splits_dir}'), "
                    f"but named split file for '{split_key}' does not exist at: {path}"
                )
            print(f"[evaluate_test_splits] Skipping {split_key}: file not found at {path}")
            continue

        loader = build_test_loader(
            csv_path=path,
            class_to_idx=class_to_idx,
            batch_size=batch_size,
            root_dir=data_dir,
            path_prefix_strip=path_prefix_strip,
        )

        all_logits: List[np.ndarray] = []
        all_preds: List[int] = []
        all_targets: List[int] = []

        for step, (images, targets) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            images = images.to(dev, non_blocking=True)
            outputs = model(images)  # Raw float32 pre-softmax logits
            logits_batch = outputs.detach().cpu().to(torch.float32).numpy()
            preds = np.argmax(logits_batch, axis=1).tolist()

            all_logits.append(logits_batch)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy().tolist())

        top1, macro_f1, micro_f1, per_class, cm = compute_split_metrics(
            all_targets=all_targets,
            all_preds=all_preds,
            class_names=class_names,
        )

        cm_path = reports_dir / f"confusion_matrix_{split_key}.npy"
        np.save(cm_path, cm)

        num_classes = len(class_names)
        if all_logits:
            split_logits = np.concatenate(all_logits, axis=0).astype(np.float32)
        else:
            split_logits = np.zeros((0, num_classes), dtype=np.float32)

        # 5a: Save raw float32 pre-softmax logits and parallel paths
        logits_file = resolved_logits_dir / f"{split_key}_logits.npy"
        paths_file = resolved_logits_dir / f"{split_key}_paths.json"

        np.save(logits_file, split_logits)

        row_count = len(loader.dataset)
        rows_evaluated = len(all_targets)
        evaluated_paths = loader.dataset.image_paths[:rows_evaluated]

        paths_payload = [
            {
                "row": i,
                "path": evaluated_paths[i],
                "label": int(all_targets[i]),
                "class_name": class_names[all_targets[i]] if (0 <= all_targets[i] < num_classes) else str(all_targets[i]),
            }
            for i in range(rows_evaluated)
        ]
        with open(paths_file, "w") as f:
            json.dump(paths_payload, f, indent=2)

        assert split_logits.shape[0] == len(paths_payload) == rows_evaluated
        assert split_logits.shape[1] == num_classes

        split_record = {
            "split_name": split_key,
            "split_file": str(path),
            "row_count": row_count,
            "rows_evaluated": rows_evaluated,
            "checkpoint_path": str(checkpoint_path),
            "weights_used": weights_used,
            "top1_accuracy": top1,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "confusion_matrix_path": str(cm_path),
            "logits_path": str(logits_file),
            "paths_path": str(paths_file),
            "per_class": per_class,
        }

        # C8: If max_steps truncated evaluation, flag partial_evaluation
        if rows_evaluated != row_count:
            split_record["partial_evaluation"] = True

        results[split_key] = split_record

        partial_str = " [PARTIAL]" if rows_evaluated != row_count else ""
        print(
            f"[{split_key}]{partial_str} Evaluated: {rows_evaluated}/{row_count} | "
            f"Top-1: {top1 * 100:.2f}% | "
            f"Macro-F1: {macro_f1:.4f} | "
            f"Micro-F1: {micro_f1:.4f}"
        )

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[evaluate_test_splits] Saved test metrics to {out_json}")
    return results


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train Model A (Plant Disease Classifier)")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2], help="Training stage (1: baseline, 2: full)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr_head", type=float, default=LR_HEAD, help="Classifier head learning rate")
    parser.add_argument("--lr_backbone", type=float, default=LR_BACKBONE, help="Backbone learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--out", type=str, default=str(CKPT / "stage1.pt"), help="Output checkpoint path")
    parser.add_argument("--metrics_out", type=str, default=str(REPORTS / "stage1_metrics.json"), help="Output metrics JSON path")
    parser.add_argument("--train_csv", type=str, default=None, help="Path to train split CSV")
    parser.add_argument("--val_csv", type=str, default=None, help="Path to val split CSV")
    parser.add_argument("--max_steps", type=int, default=None, help="Cap steps per epoch (for smoke testing)")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained backbone weights")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false", help="Do not download pretrained weights")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, mps, cpu)")
    parser.add_argument("--data_dir", type=str, default=str(ROOT), help="Root directory for resolving relative image paths in manifest (default: repo ROOT)")
    parser.add_argument("--path_prefix_strip", type=str, default="", help="Prefix to strip from every path in split CSVs at dataset construction (e.g. 'data/packaged_min/')")
    parser.add_argument("--splits_dir", type=str, default=None, help="Directory containing split CSVs (val.csv, test_indist.csv, etc.) for evaluation")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt) to resume training from")
    parser.add_argument("--class_weights_path", type=str, default=str(SPLITS / "class_weights.json"), help="Path to splits/class_weights.json")
    parser.add_argument("--eval_splits", action="store_true", default=True, help="Evaluate test splits on best checkpoint after training")
    parser.add_argument("--no_eval_splits", dest="eval_splits", action="store_false", help="Skip test split evaluation")
    parser.add_argument("--eval_checkpoint", type=str, default=str(CKPT / "stage1.pt"), help="Path to checkpoint (.pt) to evaluate")
    parser.add_argument("--eval_weights", type=str, default="auto", choices=["auto", "ema", "model"], help="Weights to evaluate: auto, ema, or model")
    parser.add_argument("--test_metrics_out", type=str, default=str(REPORTS / "stage1_test_metrics.json"), help="Output path for test metrics JSON")
    parser.add_argument("--eval_only", action="store_true", default=False, help="Run test-split evaluation only without training")
    parser.add_argument("--deterministic", action="store_true", default=False, help="Enable cuDNN deterministic mode (default: False to preserve throughput)")
    parser.add_argument("--no_wandb", action="store_true", default=False, help="Disable W&B tracking")
    parser.add_argument("--wandb_entity", type=str, default="prism-team", help="W&B entity/team")
    parser.add_argument("--wandb_project", type=str, default="SIH", help="W&B project name")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Explicit W&B run ID override")
    args = parser.parse_args(argv)

    set_seed(args.seed, deterministic=args.deterministic)

    # 1. Device resolution
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    use_amp = (device.type == "cuda")
    print(f"[train_model_a] Stage: {args.stage} | Device: {device} | AMP: {use_amp} | Data Dir: {args.data_dir} | Deterministic: {args.deterministic}")

    # 2. Build DataLoaders (reads CSV manifests)
    train_loader, val_loader = build_loaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        stage=args.stage,
        root_dir=args.data_dir,
        seed=args.seed,
        path_prefix_strip=args.path_prefix_strip,
    )

    # Dynamic class taxonomy from loader's dataset
    class_to_idx = train_loader.dataset.class_to_idx
    num_classes = len(class_to_idx)
    # Reconstruct class_names in index order
    class_names = [None] * num_classes
    for name, idx in class_to_idx.items():
        class_names[idx] = name
    print(f"[train_model_a] Resolved {num_classes} taxonomy classes from manifest.")

    # 3. Build Model
    model = build_model(
        num_classes=num_classes,
        pretrained=args.pretrained,
    )
    model = model.to(device)

    if args.eval_only:
        ckpt_eval_path = Path(args.eval_checkpoint)
        if not ckpt_eval_path.exists():
            raise FileNotFoundError(f"Evaluation checkpoint not found at: {ckpt_eval_path}")
        eval_model, weights_used = load_checkpoint_for_eval(
            checkpoint_path=ckpt_eval_path,
            model=model,
            eval_weights=args.eval_weights,
            device=device,
        )
        evaluate_test_splits(
            model=eval_model,
            checkpoint_path=ckpt_eval_path,
            class_names=class_names,
            splits_dir=args.splits_dir,
            output_report_path=args.test_metrics_out,
            reports_dir=Path(args.test_metrics_out).parent,
            data_dir=args.data_dir,
            path_prefix_strip=args.path_prefix_strip,
            batch_size=args.batch_size,
            device=device,
            weights_used=weights_used,
            max_steps=args.max_steps,
        )
        return

    # Discriminative parameter groups
    param_groups = get_parameter_groups(
        model,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
    )
    optimizer = AdamW(param_groups)

    # Load class weights reordered strictly to class_names, with sqrt transform and mean=1.0 normalisation
    class_weights = load_class_weights(
        weights_path=args.class_weights_path,
        class_names=class_names,
        device=device,
    )
    print(
        f"[train_model_a] Loaded class weights on {device}: "
        f"mean={class_weights.mean().item():.4f}, "
        f"min={class_weights.min().item():.4f}, "
        f"max={class_weights.max().item():.4f}, "
        f"dynamic_range={class_weights.max().item() / class_weights.min().item():.4f}"
    )

    # Criterion: weighted CrossEntropy with label smoothing (LABEL_SMOOTH from configs/train_config.py)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

    # Steps per epoch
    steps_per_epoch = len(train_loader) if args.max_steps is None else min(len(train_loader), args.max_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.lr_backbone, args.lr_head],
        total_steps=total_steps,
        pct_start=min(0.3, WARMUP_EPOCHS / max(1, args.epochs)),
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    ema = ModelEmaV2(model, decay=EMA_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_epoch = 1
    best_macro_f1 = -1.0
    best_metrics = {}

    if args.resume is not None:
        print(f"[train_model_a] Resuming from checkpoint: {args.resume}")
        start_epoch, best_macro_f1 = resume_from_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            expected_total_steps=total_steps,
            device=device,
        )
        print(f"[train_model_a] Resumed successfully. Next epoch: {start_epoch}, prior best val_macro_f1: {best_macro_f1:.4f}")

        if start_epoch > args.epochs:
            print(f"[train_model_a] Checkpoint completed epoch {start_epoch - 1} >= requested epochs {args.epochs}. Nothing to train.")
            return

    # 6.2 W&B Run ID resolution
    resume_wandb_run_id = None
    if args.resume is not None:
        saved_id = get_checkpoint_wandb_id(args.resume)
        if saved_id is not None:
            resume_wandb_run_id = saved_id
            print(f"[train_model_a] Reusing W&B run ID from checkpoint: {resume_wandb_run_id}")
        else:
            print("[train_model_a] WARNING: No wandb_run_id found in checkpoint; starting fresh W&B run.")
            resume_wandb_run_id = generate_wandb_id()
    else:
        resume_wandb_run_id = generate_wandb_id()

    if args.wandb_run_id is not None:
        resume_wandb_run_id = args.wandb_run_id

    # 6.1 W&B full config preparation
    wandb_cfg = dict(get_train_config_dict(args))
    wandb_cfg.update({
        "seed": args.seed,
        "total_steps": total_steps,
        "deterministic": args.deterministic,
        "git_commit": get_git_commit_hash(),
        "class_weights": [round(float(w), 5) for w in class_weights.detach().cpu().tolist()],
        "dataset_rows": {
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset),
        },
    })
    splits_base_for_cfg = Path(args.splits_dir) if args.splits_dir is not None else SPLITS
    for s_name, s_file in [
        ("test_indist", splits_base_for_cfg / "test_indist.csv"),
        ("test_sourceheldout", splits_base_for_cfg / "test_sourceheldout.csv"),
        ("test_crossdomain", splits_base_for_cfg / "test_crossdomain.csv"),
        ("test_external_riceblast", splits_base_for_cfg / "test_external_riceblast.csv"),
    ]:
        if s_file.exists():
            try:
                wandb_cfg["dataset_rows"][s_name] = len(pd.read_csv(s_file))
            except Exception:
                pass

    active_wandb_run_id = init_wandb(
        args=args,
        config=wandb_cfg,
        run_id=resume_wandb_run_id,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_path = out_path.parent / "last.pt"
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[train_model_a] Starting training (epochs {start_epoch} to {args.epochs}, {steps_per_epoch} steps/epoch)...")
    metrics_history: List[Dict[str, Any]] = []
    global_step = (start_epoch - 1) * steps_per_epoch

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss, step_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            device=device,
            use_amp=use_amp,
            max_steps=args.max_steps,
            scaler=scaler,
            global_step_start=global_step,
            base_decay=EMA_DECAY,
        )
        global_step += len(step_losses)

        # 1. Evaluate RAW model (the ground truth optimization signal)
        raw_val_macro_f1, raw_val_top1, raw_per_class_rec = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            class_names=class_names,
            max_steps=args.max_steps,
        )

        # 2. Evaluate EMA model if present
        ema_val_macro_f1, ema_val_top1, ema_per_class_rec = None, None, None
        if ema is not None:
            ema_val_macro_f1, ema_val_top1, ema_per_class_rec = evaluate(
                model=ema.module,
                loader=val_loader,
                device=device,
                class_names=class_names,
                max_steps=args.max_steps,
            )

        # Clearly labelled console logging for raw and EMA models
        if ema is not None and ema_val_top1 is not None and ema_val_macro_f1 is not None:
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | "
                f"Train Loss: {epoch_loss:.4f} | "
                f"Raw Val Top-1: {raw_val_top1 * 100:.2f}% | "
                f"Raw Val Macro-F1: {raw_val_macro_f1:.4f} | "
                f"EMA Val Top-1: {ema_val_top1 * 100:.2f}% | "
                f"EMA Val Macro-F1: {ema_val_macro_f1:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | "
                f"Train Loss: {epoch_loss:.4f} | "
                f"Raw Val Top-1: {raw_val_top1 * 100:.2f}% | "
                f"Raw Val Macro-F1: {raw_val_macro_f1:.4f}"
            )

        if step_losses:
            print(f"  Step Losses: initial={step_losses[0]:.4f} -> final={step_losses[-1]:.4f}")

        # Per-class visibility: compact single line for 5 lowest-recall classes
        worst_5_raw_str = format_lowest_recall_classes(raw_per_class_rec, n=5)
        print(f"  Lowest Recall (Raw): {worst_5_raw_str}")
        if ema is not None and ema_per_class_rec is not None:
            worst_5_ema_str = format_lowest_recall_classes(ema_per_class_rec, n=5)
            print(f"  Lowest Recall (EMA): {worst_5_ema_str}")

        epoch_record = {
            'epoch': epoch,
            'train_loss': float(epoch_loss),
            'raw_val_top1': float(raw_val_top1),
            'raw_val_macro_f1': float(raw_val_macro_f1),
            'raw_per_class_recall': raw_per_class_rec,
            # Legacy metrics mirror the raw model metrics
            'val_top1': float(raw_val_top1),
            'val_macro_f1': float(raw_val_macro_f1),
            'per_class_recall': raw_per_class_rec,
        }
        if ema is not None and ema_val_top1 is not None and ema_val_macro_f1 is not None:
            epoch_record['ema_val_top1'] = float(ema_val_top1)
            epoch_record['ema_val_macro_f1'] = float(ema_val_macro_f1)
            epoch_record['ema_per_class_recall'] = ema_per_class_rec
        metrics_history.append(epoch_record)

        # 6.1 Log per epoch: train loss, raw/ema val top-1, raw/ema val macro-F1, LR, and per-class recall
        lr_head = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]["lr"]
        wandb_epoch_payload = {
            "epoch": epoch,
            "train/loss": float(epoch_loss),
            "val/raw_top1_accuracy": float(raw_val_top1),
            "val/raw_macro_f1": float(raw_val_macro_f1),
            "val/top1_accuracy": float(raw_val_top1),
            "val/macro_f1": float(raw_val_macro_f1),
            "lr/backbone": float(optimizer.param_groups[0]["lr"]),
            "lr/head": float(lr_head),
            "per_class_recall": raw_per_class_rec,
        }
        if ema is not None and ema_val_top1 is not None and ema_val_macro_f1 is not None:
            wandb_epoch_payload["val/ema_top1_accuracy"] = float(ema_val_top1)
            wandb_epoch_payload["val/ema_macro_f1"] = float(ema_val_macro_f1)

        for cls_name, rec in raw_per_class_rec.items():
            wandb_epoch_payload[f"recall/{cls_name}"] = float(rec)
            wandb_epoch_payload[f"recall/raw_{cls_name}"] = float(rec)
        if ema is not None and ema_per_class_rec is not None:
            for cls_name, rec in ema_per_class_rec.items():
                wandb_epoch_payload[f"recall/ema_{cls_name}"] = float(rec)
        safe_wandb_log(wandb_epoch_payload)

        # Full checkpoint state dictionary preserving complete execution state
        checkpoint_state = {
            'epoch': epoch,
            'stage': args.stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'ema_state_dict': ema.state_dict() if ema is not None else None,
            'raw_val_macro_f1': raw_val_macro_f1,
            'raw_val_top1': raw_val_top1,
            'ema_val_macro_f1': ema_val_macro_f1,
            'ema_val_top1': ema_val_top1,
            'val_macro_f1': raw_val_macro_f1,  # driven strictly by raw model
            'val_top1': raw_val_top1,          # driven strictly by raw model
            'class_names': class_names,
            'num_classes': num_classes,
            'config': get_train_config_dict(args),
            'total_steps': total_steps,
            'best_macro_f1': best_macro_f1,
            'class_weights': class_weights.detach().cpu(),
            'wandb_run_id': active_wandb_run_id if active_wandb_run_id is not None else resume_wandb_run_id,
        }

        # 1. last.pt is saved every single epoch, unconditionally and atomically
        save_checkpoint_atomic(checkpoint_state, last_path)

        # 2. Macro-F1 checkpoint selection for best model (stage1.pt) - DRIVEN STRICTLY BY RAW MODEL
        if raw_val_macro_f1 > best_macro_f1:
            best_macro_f1 = raw_val_macro_f1
            checkpoint_state['best_macro_f1'] = best_macro_f1
            save_checkpoint_atomic(checkpoint_state, out_path)

            best_metrics = {
                'stage': args.stage,
                'epoch': epoch,
                'val_macro_f1': float(raw_val_macro_f1),
                'val_top1': float(raw_val_top1),
                'raw_val_macro_f1': float(raw_val_macro_f1),
                'raw_val_top1': float(raw_val_top1),
                'ema_val_macro_f1': float(ema_val_macro_f1) if ema_val_macro_f1 is not None else None,
                'ema_val_top1': float(ema_val_top1) if ema_val_top1 is not None else None,
                'per_class_recall': raw_per_class_rec,
                'num_classes': num_classes,
            }

        # 4.4 Full per-class table goes to JSON every epoch, not just on improvement
        metrics_payload = {
            'stage': args.stage,
            'best_epoch': best_metrics.get('epoch', epoch),
            'best_macro_f1': float(best_macro_f1),
            'best_val_top1': float(best_metrics.get('raw_val_top1', raw_val_top1)),
            'val_macro_f1': float(raw_val_macro_f1),
            'val_top1': float(raw_val_top1),
            'raw_val_macro_f1': float(raw_val_macro_f1),
            'raw_val_top1': float(raw_val_top1),
            'ema_val_macro_f1': float(ema_val_macro_f1) if ema_val_macro_f1 is not None else None,
            'ema_val_top1': float(ema_val_top1) if ema_val_top1 is not None else None,
            'per_class_recall': raw_per_class_rec,
            'num_classes': num_classes,
            'best_metrics': best_metrics,
            'history': metrics_history,
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_payload, f, indent=2)

    print(f"[train_model_a] Finished. Best Val Macro-F1: {best_macro_f1:.4f}")
    print(f"[train_model_a] Checkpoint saved: {out_path} (exists: {out_path.exists()})")
    print(f"[train_model_a] Last checkpoint saved: {last_path} (exists: {last_path.exists()})")
    print(f"[train_model_a] Metrics saved: {metrics_path} (exists: {metrics_path.exists()})")

    # 4. Evaluate test splits on best checkpoint
    if args.eval_splits and out_path.exists():
        print(f"\n[train_model_a] Evaluating test splits on best checkpoint: {out_path}")
        eval_model, weights_used = load_checkpoint_for_eval(
            checkpoint_path=out_path,
            model=model,
            eval_weights=args.eval_weights,
            device=device,
        )
        test_results = evaluate_test_splits(
            model=eval_model,
            checkpoint_path=out_path,
            class_names=class_names,
            splits_dir=args.splits_dir,
            output_report_path=args.test_metrics_out,
            reports_dir=Path(args.test_metrics_out).parent,
            data_dir=args.data_dir,
            path_prefix_strip=args.path_prefix_strip,
            batch_size=args.batch_size,
            device=device,
            weights_used=weights_used,
            max_steps=args.max_steps,
        )

        # 6.1 Log test metrics per split as separate summary entries in W&B (never merged)
        wandb_test_summary = {}
        for split_key, s_data in test_results.items():
            wandb_test_summary[f"{split_key}/top1_accuracy"] = float(s_data["top1_accuracy"])
            wandb_test_summary[f"{split_key}/macro_f1"] = float(s_data["macro_f1"])
            wandb_test_summary[f"{split_key}/micro_f1"] = float(s_data["micro_f1"])
            wandb_test_summary[f"{split_key}/rows_evaluated"] = int(s_data["rows_evaluated"])
            wandb_test_summary[f"{split_key}_summary"] = s_data
        safe_wandb_summary(wandb_test_summary)

    safe_wandb_finish()


if __name__ == "__main__":
    main()
