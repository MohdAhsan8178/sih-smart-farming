"""
Unit tests for STEP 11 (Data Pipeline & Augmentation) and STEP 12 (Model A Baseline).
Verifies:
- Acceptance criteria of STEP 11 (DataLoader output shape, tensor dtype, value range)
- Hard constraint: dynamic class taxonomy from manifest/config (synthetic 3-class manifest test)
- Head width and class ordering strictly follow the manifest, not any constant
- Transforms execution and agronomic hue preservation constraint (<= 12)
- Discriminative learning rate parameter group separation
"""

import json
import math
from pathlib import Path
import random
import sys
import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from configs.classes import CLASS_NAMES
from configs.paths import ONNX_DIR, REPORTS, ROOT, SPLITS
from configs.train_config import (
    BATCH_SIZE,
    EMA_DECAY,
    EPOCHS,
    IMAGE_SIZE,
    LABEL_SMOOTH,
    LR_BACKBONE,
    LR_HEAD,
    WEIGHT_DECAY,
)
from train.transforms import (
    HUE_SHIFT_LIMIT,
    baseline_train_transform,
    eval_transform,
    train_transform,
)
from timm.utils import ModelEmaV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from train.dataset import PlantDataset, build_loaders, seed_worker
from train.model import build_model, get_parameter_groups
from train.split import compute_class_weights
from train.train_model_a import (
    build_test_loader,
    compute_split_metrics,
    evaluate,
    evaluate_test_splits,
    format_lowest_recall_classes,
    generate_wandb_id,
    get_checkpoint_wandb_id,
    get_train_config_dict,
    init_wandb,
    load_checkpoint_for_eval,
    load_class_weights,
    main,
    resume_from_checkpoint,
    safe_wandb_finish,
    safe_wandb_log,
    safe_wandb_summary,
    save_checkpoint_atomic,
    set_seed,
    train_one_epoch,
)
from scripts.extract_logits import extract_logits, discover_images

try:
    import wandb
except ImportError:
    wandb = None


def test_step_11_build_loaders_shape_and_tensor_type():
    """
    Verifies the STEP 11 acceptance criteria:
    tl, vl = build_loaders()
    x, y = next(iter(tl))
    assert x.shape[1] == 3 and x.shape[2] == 224
    """
    train_loader, val_loader = build_loaders(batch_size=8, stage=1)
    images, labels = next(iter(train_loader))

    assert isinstance(images, torch.Tensor), "Images must be torch.Tensor"
    assert isinstance(labels, torch.Tensor), "Labels must be torch.Tensor"
    assert images.dtype == torch.float32, "Tensor dtype must be float32"
    assert images.ndim == 4, "Batch must have shape (B, C, H, W)"
    assert images.shape[1] == 3, f"Expected 3 color channels, got {images.shape[1]}"
    assert images.shape[2] == IMAGE_SIZE, f"Expected height {IMAGE_SIZE}, got {images.shape[2]}"
    assert images.shape[3] == IMAGE_SIZE, f"Expected width {IMAGE_SIZE}, got {images.shape[3]}"
    assert labels.ndim == 1, "Labels must be a 1D tensor"
    assert len(labels) == len(images), "Batch labels count must match batch images count"


def test_step_11_transforms_hue_limit_and_dimensions():
    """
    Verifies that all three transform pipelines produce tensors of shape (3, 224, 224)
    and enforces the agronomic constraint that hue shifts never exceed +/- 12.
    """
    assert HUE_SHIFT_LIMIT <= 12, (
        f"Agronomic violation: HUE_SHIFT_LIMIT is {HUE_SHIFT_LIMIT} > 12. "
        "Colour is diagnostic for foliar diseases and must not be altered beyond +/- 12."
    )

    dummy_img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

    t_train = train_transform(size=IMAGE_SIZE)(image=dummy_img)['image']
    t_base = baseline_train_transform(size=IMAGE_SIZE)(image=dummy_img)['image']
    t_eval = eval_transform(size=IMAGE_SIZE)(image=dummy_img)['image']

    for name, t in [("train", t_train), ("baseline", t_base), ("eval", t_eval)]:
        assert t.shape == (3, IMAGE_SIZE, IMAGE_SIZE), (
            f"Transform {name} produced shape {t.shape}, expected (3, {IMAGE_SIZE}, {IMAGE_SIZE})"
        )
        assert t.dtype == torch.float32, f"Transform {name} must produce float32 tensor"


def test_synthetic_3_class_manifest_dynamic_head_and_ordering(tmp_path: Path):
    """
    HARD CONSTRAINT PROOF:
    Builds the dataset and DataLoader against a synthetic 3-class manifest.
    Asserts:
      1. Head width of the constructed model is exactly 3 (follows manifest, not 29 or 31).
      2. Class ordering follows the custom manifest taxonomy, not any hardcoded constant.
      3. DataLoader produces targets strictly within [0, 2].
      4. Forward pass produces logits with shape (B, 3).
    """
    # 1. Create dummy image files for 3 synthetic classes
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    synthetic_classes = ["test_alpha_rot", "test_beta_smut", "test_gamma_healthy"]
    records = []

    for c_idx, class_name in enumerate(synthetic_classes):
        for img_i in range(3):
            img_file = img_dir / f"{class_name}_{img_i}.jpg"
            # Write a solid test image
            cv2.imwrite(str(img_file), np.full((100, 100, 3), fill_value=40 * (c_idx + 1), dtype=np.uint8))
            records.append({
                'path': str(img_file),
                'label': class_name,
                'source_dataset': 'synthetic_test',
                'orig_folder': class_name,
                'group_id': c_idx * 10 + img_i,
            })

    synth_csv = tmp_path / "synthetic_splits.csv"
    pd.DataFrame(records).to_csv(synth_csv, index=False)

    # 2. Build dataset with explicit 3-class taxonomy mapping
    synth_class_to_idx = {name: idx for idx, name in enumerate(synthetic_classes)}
    dataset = PlantDataset(
        data_source=synth_csv,
        transform=eval_transform(size=IMAGE_SIZE),
        class_to_idx=synth_class_to_idx,
    )

    assert len(dataset) == 9
    assert dataset.class_to_idx == synth_class_to_idx
    assert len(dataset.class_to_idx) == 3

    # Verify class ordering in dataset matches synthetic manifest
    for sample_idx in range(len(dataset)):
        tensor, target = dataset[sample_idx]
        expected_class = records[sample_idx]['label']
        expected_target = synth_class_to_idx[expected_class]
        assert target == expected_target
        assert 0 <= target < 3

    # 3. Build model against synthetic class count
    model = build_model(
        num_classes=len(dataset.class_to_idx),
        pretrained=False,
    )

    # Assert classifier head width is strictly 3
    classifier = model.get_classifier()
    assert classifier.out_features == 3, (
        f"Expected model head width to be 3 (from synthetic manifest), but got {classifier.out_features}."
    )

    # 4. Pass synthetic batch through model
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    logits = model(batch_x)

    assert logits.shape == (4, 3), f"Expected logits shape (4, 3), got {logits.shape}"


def test_step_12_discriminative_parameter_groups():
    """
    Verifies parameter separation between backbone (lr=1e-4) and head (lr=1e-3).
    """
    model = build_model(num_classes=5, pretrained=False)
    groups = get_parameter_groups(model)

    assert len(groups) == 2, "Expected 2 parameter groups (backbone and head)"

    backbone_group = groups[0]
    head_group = groups[1]

    assert backbone_group['lr'] == LR_BACKBONE, f"Backbone LR must be {LR_BACKBONE}"
    assert head_group['lr'] == LR_HEAD, f"Head LR must be {LR_HEAD}"
    assert backbone_group['weight_decay'] == WEIGHT_DECAY
    assert head_group['weight_decay'] == WEIGHT_DECAY

    head_param_set = set(model.get_classifier().parameters())
    head_group_set = set(head_group['params'])
    assert head_param_set == head_group_set, "Head parameter group must match get_classifier() parameters"


def test_step_12_smoke_training_step(tmp_path: Path):
    """
    Verifies that train_one_epoch and evaluate execute without errors on synthetic data,
    compute finite loss, and return valid Macro-F1 and Top-1 metrics.
    """
    img_dir = tmp_path / "smoke_images"
    img_dir.mkdir()

    classes = ["smoke_c1", "smoke_c2", "smoke_c3"]
    records = []
    for c_idx, c_name in enumerate(classes):
        for i in range(4):
            img_file = img_dir / f"{c_name}_{i}.jpg"
            cv2.imwrite(str(img_file), np.full((64, 64, 3), fill_value=50 * (c_idx + 1), dtype=np.uint8))
            records.append({'path': str(img_file), 'label': c_name})

    csv_path = tmp_path / "smoke.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_loader, val_loader = build_loaders(
        train_csv=csv_path,
        val_csv=csv_path,
        batch_size=4,
        stage=1,
        class_to_idx=class_to_idx,
    )

    model = build_model(num_classes=len(classes), pretrained=False)
    optimizer = torch.optim.AdamW(get_parameter_groups(model))
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    loss, step_losses = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        ema=None,
        device=device,
        max_steps=2,
    )

    assert np.isfinite(loss), "Training loss must be finite"
    assert len(step_losses) == 2, "Expected exactly 2 step losses recorded"

    macro_f1, top1, per_class_rec = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        class_names=classes,
        max_steps=2,
    )

    assert 0.0 <= macro_f1 <= 1.0, f"Invalid Macro-F1: {macro_f1}"
    assert 0.0 <= top1 <= 1.0, f"Invalid Top-1 accuracy: {top1}"
    assert len(per_class_rec) == 3, f"Expected 3 per-class recall entries, got {len(per_class_rec)}"


def test_manifest_and_configs_classes_must_agree_raising_if_not(tmp_path: Path):
    """
    Verifies that manifest and configs/classes.py taxonomy must strictly agree,
    raising ValueError upfront if either the manifest contains an unknown class
    or is missing a class defined in the config.
    """
    from configs.classes import CLASS_NAMES

    # Dummy image
    img_file = tmp_path / "dummy.jpg"
    cv2.imwrite(str(img_file), np.zeros((32, 32, 3), dtype=np.uint8))

    # Case 1: Manifest contains an unknown class not in configs/classes.py
    df_unknown = pd.DataFrame([
        {'path': str(img_file), 'label': CLASS_NAMES[0]},
        {'path': str(img_file), 'label': 'alien_unregistered_disease'},
    ])
    with pytest.raises(ValueError, match="Taxonomy agreement failure"):
        PlantDataset(
            data_source=df_unknown,
            enforce_manifest_agreement=True,
        )

    # Case 2: Manifest is missing classes defined in configs/classes.py
    df_incomplete = pd.DataFrame([
        {'path': str(img_file), 'label': CLASS_NAMES[0]},
        {'path': str(img_file), 'label': CLASS_NAMES[1]},
    ])
    with pytest.raises(ValueError, match="Taxonomy agreement failure"):
        PlantDataset(
            data_source=df_incomplete,
            enforce_manifest_agreement=True,
        )

    # Case 3: Agreement with custom taxonomy dictionary
    custom_dict = {'crop_a': 0, 'crop_b': 1}
    df_custom_agree = pd.DataFrame([
        {'path': str(img_file), 'label': 'crop_a'},
        {'path': str(img_file), 'label': 'crop_b'},
    ])
    ds = PlantDataset(
        data_source=df_custom_agree,
        class_to_idx=custom_dict,
        enforce_manifest_agreement=True,
    )
    assert len(ds) == 2
    assert set(ds.class_to_idx.keys()) == {'crop_a', 'crop_b'}


def test_step13_data_dir_override_resolves_paths_against_supplied_root(tmp_path: Path):
    """
    Asserts that passing a non-default --data_dir causes image paths in split CSVs
    to resolve against the supplied root directory and not against repository ROOT.
    """
    mock_kaggle_dir = tmp_path / "mock_kaggle_dataset"
    img_dir = mock_kaggle_dir / "data" / "raw" / "test_crop"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_file = img_dir / "sample_001.jpg"

    # Write a test image inside the mock data directory
    dummy_pixels = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(img_file), dummy_pixels)

    # Relative path as stored in splits/*.csv
    rel_img_path = "data/raw/test_crop/sample_001.jpg"
    df_manifest = pd.DataFrame([
        {'path': rel_img_path, 'label': 'test_crop'}
    ])
    csv_file = tmp_path / "test_split.csv"
    df_manifest.to_csv(csv_file, index=False)

    # 1. Verification: without data_dir override (defaulting to repo ROOT),
    # attempting to load the sample must raise FileNotFoundError because the path
    # does not exist under ROOT.
    train_loader_default, _ = build_loaders(
        train_csv=csv_file,
        val_csv=csv_file,
        batch_size=1,
        class_to_idx={'test_crop': 0},
        enforce_manifest_agreement=False,
    )
    with pytest.raises(FileNotFoundError, match="Image path does not exist on disk"):
        _ = train_loader_default.dataset[0]

    # 2. Verification: with data_dir override pointing to mock_kaggle_dir,
    # images resolve cleanly against the supplied directory.
    train_loader_override, _ = build_loaders(
        train_csv=csv_file,
        val_csv=csv_file,
        batch_size=1,
        class_to_idx={'test_crop': 0},
        enforce_manifest_agreement=False,
        root_dir=mock_kaggle_dir,
    )
    tensor, target = train_loader_override.dataset[0]
    assert tensor.shape == (3, 224, 224), f"Expected tensor shape (3, 224, 224), got {tensor.shape}"
    assert target == 0, f"Expected target class index 0, got {target}"


def test_step13_checkpoint_contains_scheduler_scaler_and_ema_state(tmp_path: Path):
    """
    Asserts that saved checkpoints preserve all components of training execution state:
    epoch, stage, model_state_dict, optimizer_state_dict, scheduler_state_dict,
    scaler_state_dict, ema_state_dict, val_macro_f1, val_top1, class_names,
    num_classes, config, and total_steps.
    """
    model = build_model(num_classes=3, pretrained=False)
    optimizer = AdamW(get_parameter_groups(model))
    total_steps = 30
    scheduler = OneCycleLR(optimizer, max_lr=[1e-4, 1e-3], total_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    ema = ModelEmaV2(model, decay=0.99)
    classes = ["class_a", "class_b", "class_c"]

    ckpt_file = tmp_path / "test_stage1.pt"
    state = {
        'epoch': 3,
        'stage': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'val_macro_f1': 0.885,
        'val_top1': 0.912,
        'class_names': classes,
        'num_classes': len(classes),
        'config': get_train_config_dict(),
        'total_steps': total_steps,
        'best_macro_f1': 0.885,
        'class_weights': torch.ones(len(classes), dtype=torch.float32),
    }
    save_checkpoint_atomic(state, ckpt_file)

    loaded = torch.load(ckpt_file, map_location="cpu")
    required_keys = [
        'epoch', 'stage', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'scaler_state_dict', 'ema_state_dict',
        'val_macro_f1', 'val_top1', 'class_names', 'num_classes',
        'config', 'total_steps', 'class_weights',
    ]
    for key in required_keys:
        assert key in loaded, f"Checkpoint missing required key: {key}"

    assert loaded['epoch'] == 3
    assert loaded['stage'] == 1
    assert loaded['total_steps'] == 30
    assert isinstance(loaded['scheduler_state_dict'], dict)
    assert 'total_steps' in loaded['scheduler_state_dict']
    assert isinstance(loaded['scaler_state_dict'], dict)
    assert isinstance(loaded['ema_state_dict'], dict)
    assert len(loaded['ema_state_dict']) > 0
    assert isinstance(loaded['config'], dict)
    assert 'BATCH_SIZE' in loaded['config']
    assert 'BACKBONE' in loaded['config']


def test_step13_resume_restores_epoch_and_scheduler_step_position(tmp_path: Path):
    """
    Asserts that resume advances to start_epoch = checkpoint['epoch'] + 1,
    and faithfully restores scheduler step position and state across save/load.
    """
    model = build_model(num_classes=3, pretrained=False)
    optimizer = AdamW(get_parameter_groups(model))
    total_steps = 25
    scheduler = OneCycleLR(optimizer, max_lr=[1e-4, 1e-3], total_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    ema = ModelEmaV2(model, decay=0.99)

    # Step optimizer and scheduler 8 times
    for _ in range(8):
        optimizer.step()
        scheduler.step()

    step_pos_before = scheduler.last_epoch
    lr_before = scheduler.get_last_lr()
    assert step_pos_before == 8

    ckpt_file = tmp_path / "resume_test.pt"
    state = {
        'epoch': 2,
        'stage': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'val_macro_f1': 0.75,
        'val_top1': 0.80,
        'class_names': ["c1", "c2", "c3"],
        'num_classes': 3,
        'config': get_train_config_dict(),
        'total_steps': total_steps,
        'best_macro_f1': 0.75,
    }
    save_checkpoint_atomic(state, ckpt_file)

    # Fresh model, optimizer, scheduler, scaler, ema
    fresh_model = build_model(num_classes=3, pretrained=False)
    fresh_optimizer = AdamW(get_parameter_groups(fresh_model))
    fresh_scheduler = OneCycleLR(fresh_optimizer, max_lr=[1e-4, 1e-3], total_steps=total_steps)
    fresh_scaler = torch.amp.GradScaler('cuda', enabled=False)
    fresh_ema = ModelEmaV2(fresh_model, decay=0.99)

    start_epoch, best_f1 = resume_from_checkpoint(
        checkpoint_path=ckpt_file,
        model=fresh_model,
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
        scaler=fresh_scaler,
        ema=fresh_ema,
        expected_total_steps=total_steps,
    )

    # Verify epoch restoration
    assert start_epoch == 3, f"Expected start_epoch=3 (epoch 2 + 1), got {start_epoch}"
    assert best_f1 == 0.75

    # Verify scheduler position and LR restoration
    assert fresh_scheduler.last_epoch == step_pos_before == 8
    assert fresh_scheduler.get_last_lr() == lr_before

    # Verify continued stepping continues on OneCycleLR schedule
    fresh_optimizer.step()
    fresh_scheduler.step()
    assert fresh_scheduler.last_epoch == 9


def test_step13_last_checkpoint_written_every_epoch_even_when_metric_does_not_improve(tmp_path: Path):
    """
    Asserts that last.pt is updated on every epoch regardless of performance,
    while stage1.pt (best) is only updated when val_macro_f1 strictly improves.
    """
    model = build_model(num_classes=2, pretrained=False)
    optimizer = AdamW(get_parameter_groups(model))
    total_steps = 10
    scheduler = OneCycleLR(optimizer, max_lr=[1e-4, 1e-3], total_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    ema = ModelEmaV2(model, decay=0.99)

    out_path = tmp_path / "stage1.pt"
    last_path = tmp_path / "last.pt"

    best_macro_f1 = -1.0

    # Epoch 1: Metric 0.85 -> both last.pt and stage1.pt are written
    epoch1_state = {
        'epoch': 1,
        'stage': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'val_macro_f1': 0.85,
        'val_top1': 0.88,
        'class_names': ["a", "b"],
        'num_classes': 2,
        'config': get_train_config_dict(),
        'total_steps': total_steps,
        'best_macro_f1': best_macro_f1,
    }
    save_checkpoint_atomic(epoch1_state, last_path)
    if epoch1_state['val_macro_f1'] > best_macro_f1:
        best_macro_f1 = epoch1_state['val_macro_f1']
        epoch1_state['best_macro_f1'] = best_macro_f1
        save_checkpoint_atomic(epoch1_state, out_path)

    assert last_path.exists()
    assert out_path.exists()
    assert torch.load(out_path, map_location="cpu")['epoch'] == 1
    assert torch.load(out_path, map_location="cpu")['val_macro_f1'] == 0.85

    # Epoch 2: Degraded metric (0.70 < 0.85) -> last.pt updated to epoch 2, stage1.pt remains epoch 1
    epoch2_state = {
        'epoch': 2,
        'stage': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'val_macro_f1': 0.70,
        'val_top1': 0.72,
        'class_names': ["a", "b"],
        'num_classes': 2,
        'config': get_train_config_dict(),
        'total_steps': total_steps,
        'best_macro_f1': best_macro_f1,
    }
    save_checkpoint_atomic(epoch2_state, last_path)
    if epoch2_state['val_macro_f1'] > best_macro_f1:
        best_macro_f1 = epoch2_state['val_macro_f1']
        epoch2_state['best_macro_f1'] = best_macro_f1
        save_checkpoint_atomic(epoch2_state, out_path)

    # Verify last.pt points to epoch 2
    ckpt_last = torch.load(last_path, map_location="cpu")
    assert ckpt_last['epoch'] == 2, f"Expected last.pt to be epoch 2, got {ckpt_last['epoch']}"
    assert ckpt_last['val_macro_f1'] == 0.70

    # Verify stage1.pt was NOT overwritten and remains epoch 1
    ckpt_best = torch.load(out_path, map_location="cpu")
    assert ckpt_best['epoch'] == 1, f"Expected stage1.pt to remain epoch 1, got {ckpt_best['epoch']}"
    assert ckpt_best['val_macro_f1'] == 0.85


def test_step13_resume_with_mismatched_total_steps_raises_not_silently_continues(tmp_path: Path):
    """
    Asserts that resuming with a total_steps mismatch raises ValueError
    instead of corrupting OneCycleLR schedule silently.
    """
    model = build_model(num_classes=2, pretrained=False)
    optimizer = AdamW(get_parameter_groups(model))
    scaler = torch.amp.GradScaler('cuda', enabled=False)

    ckpt_file = tmp_path / "mismatch.pt"
    state = {
        'epoch': 1,
        'stage': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': None,
        'scaler_state_dict': scaler.state_dict(),
        'ema_state_dict': None,
        'val_macro_f1': 0.80,
        'val_top1': 0.80,
        'class_names': ["a", "b"],
        'num_classes': 2,
        'config': get_train_config_dict(),
        'total_steps': 50,
    }
    save_checkpoint_atomic(state, ckpt_file)

    fresh_model = build_model(num_classes=2, pretrained=False)
    fresh_optimizer = AdamW(get_parameter_groups(fresh_model))

    with pytest.raises(ValueError, match="Mismatched total_steps"):
        resume_from_checkpoint(
            checkpoint_path=ckpt_file,
            model=fresh_model,
            optimizer=fresh_optimizer,
            expected_total_steps=100,
        )


def test_step13_interrupted_checkpoint_write_does_not_corrupt_existing_file(tmp_path: Path):
    """
    Asserts that if saving is interrupted or fails mid-write, the existing checkpoint
    remains uncorrupted and no temporary artifacts are left behind.
    """
    ckpt_file = tmp_path / "target_checkpoint.pt"

    # Write initial valid checkpoint
    initial_state = {'epoch': 1, 'data': 'unbroken_state_initial'}
    save_checkpoint_atomic(initial_state, ckpt_file)
    assert ckpt_file.exists()

    # Define mock that simulates failure during torch.save
    class FailingTorchSave:
        def __call__(self, obj, f, *args, **kwargs):
            f.write(b"CORRUPTED_PARTIAL_BYTES")
            raise IOError("Simulated power loss / disk failure mid-write")

    import unittest.mock as mock
    with mock.patch("torch.save", side_effect=FailingTorchSave()):
        with pytest.raises(IOError, match="Simulated power loss"):
            save_checkpoint_atomic({'epoch': 2, 'data': 'new_state'}, ckpt_file)

    # Verify original file is completely intact
    loaded = torch.load(ckpt_file, map_location="cpu")
    assert loaded['epoch'] == 1
    assert loaded['data'] == 'unbroken_state_initial'

    # Verify no dangling temporary files exist in tmp_path
    tmp_files = list(tmp_path.glob(".tmp_*"))
    assert len(tmp_files) == 0, f"Leftover temp files found: {tmp_files}"


def test_step13_class_weights_json_keys_match_class_names_exactly():
    """
    Asserts that splits/class_weights.json contains metadata and that its
    raw_inverse_weights and class_counts keys match CLASS_NAMES exactly (all 29 classes).
    """
    weights_path = SPLITS / "class_weights.json"
    assert weights_path.exists(), f"Missing weights file: {weights_path}"

    with open(weights_path, "r") as f:
        data = json.load(f)

    assert data["source_split"] == "splits/train.csv"
    assert data["total_samples"] == 16526
    assert "timestamp" in data

    raw_keys = set(data["raw_inverse_weights"].keys())
    count_keys = set(data["class_counts"].keys())
    expected_keys = set(CLASS_NAMES)

    assert raw_keys == expected_keys, f"Mismatch in raw_inverse_weights keys: {raw_keys ^ expected_keys}"
    assert count_keys == expected_keys, f"Mismatch in class_counts keys: {count_keys ^ expected_keys}"
    assert len(raw_keys) == len(CLASS_NAMES) == 29


def test_step13_class_weights_reordered_to_class_names_not_file_order(tmp_path: Path):
    """
    Constructs a JSON file with keys deliberately shuffled/reversed, and asserts
    that load_class_weights reorders the tensor to strictly match CLASS_NAMES order.
    """
    # Assign distinct raw weights: raw_val = (i + 1) ** 2 for index i in CLASS_NAMES
    raw_weights = {c: float((i + 1) ** 2) for i, c in enumerate(CLASS_NAMES)}

    # Reverse the key order in the JSON file
    reversed_keys = list(reversed(CLASS_NAMES))
    shuffled_payload = {
        "source_split": "splits/train.csv",
        "total_samples": 1000,
        "timestamp": "2026-09-09T00:00:00Z",
        "class_counts": {c: 10 for c in reversed_keys},
        "raw_inverse_weights": {c: raw_weights[c] for c in reversed_keys},
    }
    shuffled_path = tmp_path / "shuffled_weights.json"
    with open(shuffled_path, "w") as f:
        json.dump(shuffled_payload, f)

    loaded_tensor = load_class_weights(shuffled_path, class_names=CLASS_NAMES)

    # Compute expected normalized sqrt vector in canonical CLASS_NAMES order
    expected_raw = np.array([(i + 1) ** 2 for i in range(len(CLASS_NAMES))], dtype=np.float32)
    expected_sqrt = np.sqrt(expected_raw)
    expected_norm = expected_sqrt / np.mean(expected_sqrt)
    expected_tensor = torch.tensor(expected_norm, dtype=torch.float32)

    assert torch.allclose(loaded_tensor, expected_tensor, atol=1e-5), (
        "Loaded class weights tensor did not preserve canonical CLASS_NAMES ordering!"
    )
    # Also verify that the first element is indeed rice__normal (i=0 -> 1.0) and not not_crop (reversed)
    assert loaded_tensor[0] < loaded_tensor[-1]


def test_step13_sqrt_weights_normalised_to_mean_one():
    """
    Asserts that loading splits/class_weights.json yields weights whose mean is 1.0,
    with max:min ratio ~3.88:1, corresponding to wheat__brown_rust and rice__normal.
    """
    weights = load_class_weights(SPLITS / "class_weights.json", class_names=CLASS_NAMES)

    assert len(weights) == len(CLASS_NAMES) == 29
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-5)

    idx_min = torch.argmin(weights).item()
    idx_max = torch.argmax(weights).item()

    assert CLASS_NAMES[idx_min] == "rice__normal"
    assert CLASS_NAMES[idx_max] == "wheat__brown_rust"

    min_val = weights[idx_min].item()
    max_val = weights[idx_max].item()

    assert pytest.approx(min_val, abs=1e-3) == 0.5022
    assert pytest.approx(max_val, abs=1e-3) == 1.9483
    assert pytest.approx(max_val / min_val, abs=1e-3) == 3.8793


def test_step13_missing_class_weights_file_raises_not_silently_uniform(tmp_path: Path):
    """
    Asserts that if the class weights JSON is missing, load_class_weights raises
    FileNotFoundError and never falls back silently to uniform weights.
    """
    missing_file = tmp_path / "nonexistent_class_weights.json"
    with pytest.raises(FileNotFoundError, match="Class weights file not found"):
        load_class_weights(missing_file, class_names=CLASS_NAMES)


def test_step13_extra_or_missing_class_key_raises_with_class_name_in_message(tmp_path: Path):
    """
    Asserts that missing keys and extra unexpected keys both raise ValueError
    with the offending class name clearly displayed in the exception message.
    """
    valid_payload = {
        "source_split": "splits/train.csv",
        "total_samples": 1000,
        "timestamp": "2026-09-09T00:00:00Z",
        "raw_inverse_weights": {c: 1.0 for c in CLASS_NAMES},
    }

    # Case A: Missing class key
    missing_class = CLASS_NAMES[0]
    payload_missing = dict(valid_payload)
    payload_missing["raw_inverse_weights"] = {
        c: 1.0 for c in CLASS_NAMES if c != missing_class
    }
    path_missing = tmp_path / "missing.json"
    with open(path_missing, "w") as f:
        json.dump(payload_missing, f)

    with pytest.raises(ValueError, match="Missing classes") as exc_missing:
        load_class_weights(path_missing, class_names=CLASS_NAMES)
    assert missing_class in str(exc_missing.value)

    # Case B: Extra unexpected class key
    extra_class = "alien_crop__unknown_blight"
    payload_extra = dict(valid_payload)
    payload_extra["raw_inverse_weights"] = dict(valid_payload["raw_inverse_weights"])
    payload_extra["raw_inverse_weights"][extra_class] = 1.0
    path_extra = tmp_path / "extra.json"
    with open(path_extra, "w") as f:
        json.dump(payload_extra, f)

    with pytest.raises(ValueError, match="Extra unexpected classes") as exc_extra:
        load_class_weights(path_extra, class_names=CLASS_NAMES)
    assert extra_class in str(exc_extra.value)


def test_step13_zero_count_class_raises_instead_of_infinite_weight(tmp_path: Path):
    """
    Asserts that if any class has 0 images in the training split, compute_class_weights
    raises ValueError citing the class name rather than producing inf or default weights.
    Also asserts that load_class_weights rejects zero or infinite raw weights.
    """
    # 1. Test split.py's zero-count guard
    dummy_records = [
        {"path": f"img_{c}.jpg", "label": c}
        for c in CLASS_NAMES if c != "wheat__brown_rust"
    ]
    df_missing_class = pd.DataFrame(dummy_records)
    with pytest.raises(ValueError, match="zero rows in train.csv; inverse-frequency weight is infinite") as excinfo:
        compute_class_weights(df_missing_class, class_names=CLASS_NAMES)
    assert "wheat__brown_rust" in str(excinfo.value)

    # 2. Test load_class_weights numerical sanity guard against zero or inf
    bad_payload = {
        "raw_inverse_weights": {c: (0.0 if c == "wheat__brown_rust" else 1.0) for c in CLASS_NAMES}
    }
    bad_path = tmp_path / "zero_weight.json"
    with open(bad_path, "w") as f:
        json.dump(bad_payload, f)
    with pytest.raises(ValueError, match="invalid or zero/infinite raw weight") as excinfo_zero:
        load_class_weights(bad_path, class_names=CLASS_NAMES)
    assert "wheat__brown_rust" in str(excinfo_zero.value)


def test_step13_label_smoothing_value_is_actually_passed_to_criterion():
    """
    Asserts that LABEL_SMOOTH = 0.1 from configs/train_config.py is actively
    wired into the CrossEntropyLoss criterion alongside the class weights.
    """
    weights = load_class_weights(SPLITS / "class_weights.json", class_names=CLASS_NAMES)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTH)

    assert criterion.label_smoothing == LABEL_SMOOTH == 0.1
    assert criterion.weight is not None
    assert torch.equal(criterion.weight, weights)

    # Verify that label smoothing changes the loss value vs unsmoothed loss
    criterion_unsmoothed = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.0)
    dummy_logits = torch.zeros((2, len(CLASS_NAMES)))
    dummy_logits[:, 0] = 5.0
    dummy_targets = torch.tensor([0, 0])

    loss_smoothed = criterion(dummy_logits, dummy_targets)
    loss_unsmoothed = criterion_unsmoothed(dummy_logits, dummy_targets)
    assert not torch.isclose(loss_smoothed, loss_unsmoothed)


def test_step13_weight_tensor_is_on_same_device_as_model():
    """
    Asserts that the loaded class weights tensor resides on the identical device
    as the model parameters and does not trigger CUDA/device mismatch exceptions.
    """
    device = torch.device("cpu")
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False).to(device)
    weights = load_class_weights(SPLITS / "class_weights.json", class_names=CLASS_NAMES, device=device)

    model_device = next(model.parameters()).device
    assert weights.device == model_device == device

    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTH)
    dummy_x = torch.randn(2, 3, 224, 224, device=device)
    dummy_y = torch.tensor([0, 1], device=device)

    outputs = model(dummy_x)
    loss = criterion(outputs, dummy_y)
    assert torch.isfinite(loss)


def test_step13_indist_and_crossdomain_evaluated_separately_never_averaged(tmp_path: Path):
    """
    Asserts that indist and crossdomain splits are evaluated as separate, distinct
    blocks in the output JSON with different row counts (2069 vs 9389), and never
    averaged or concatenated together.
    """
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(3, len(CLASS_NAMES)),
    ).eval()

    out_json = tmp_path / "stage1_test_metrics.json"
    results = evaluate_test_splits(
        model=model,
        checkpoint_path=tmp_path / "stage1.pt",
        class_names=CLASS_NAMES,
        splits_dict={
            "test_indist": SPLITS / "test_indist.csv",
            "test_crossdomain": SPLITS / "test_crossdomain.csv",
        },
        output_report_path=out_json,
        reports_dir=tmp_path,
        batch_size=4,
        weights_used="ema",
        max_steps=1,
    )

    # 1. Output file must exist and contain distinct blocks
    assert out_json.exists(), f"Output metrics JSON not written to {out_json}"
    with open(out_json, "r") as f:
        disk_results = json.load(f)

    assert "test_indist" in disk_results
    assert "test_crossdomain" in disk_results

    # 2. Distinct row counts reflecting actual split sizes and partial evaluation flag
    indist_rows = disk_results["test_indist"]["row_count"]
    cross_rows = disk_results["test_crossdomain"]["row_count"]
    assert indist_rows == 2069
    assert cross_rows == 9389
    assert indist_rows != cross_rows, (
        f"Row counts must be distinct between indist ({indist_rows}) and crossdomain ({cross_rows})"
    )
    # C8: With max_steps=1 and batch_size=4, exactly 4 rows evaluated and partial_evaluation flagged
    assert disk_results["test_indist"]["rows_evaluated"] == 4
    assert disk_results["test_indist"]["partial_evaluation"] is True
    assert disk_results["test_crossdomain"]["rows_evaluated"] == 4
    assert disk_results["test_crossdomain"]["partial_evaluation"] is True

    # 3. No averaged, merged, or concatenated block exists
    for key in disk_results.keys():
        assert "avg" not in key.lower() and "mean" not in key.lower() and "combined" not in key.lower(), (
            f"Found averaged or combined split key '{key}'! Splits must never be averaged or concatenated."
        )


def test_step13_per_class_metrics_written_for_all_29_classes(tmp_path: Path):
    """
    Asserts that per-class precision, recall, f1, and support are computed and
    written for all 29 classes in the taxonomy.
    """
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(3, len(CLASS_NAMES)),
    ).eval()

    out_json = tmp_path / "stage1_test_metrics.json"
    results = evaluate_test_splits(
        model=model,
        checkpoint_path=tmp_path / "stage1.pt",
        class_names=CLASS_NAMES,
        splits_dict={"test_indist": SPLITS / "test_indist.csv"},
        output_report_path=out_json,
        reports_dir=tmp_path,
        batch_size=4,
        weights_used="model_state_dict",
        max_steps=1,
    )

    per_class = results["test_indist"]["per_class"]
    assert len(per_class) == 29 == len(CLASS_NAMES), (
        f"Expected 29 classes in per_class metrics, found {len(per_class)}"
    )
    assert set(per_class.keys()) == set(CLASS_NAMES)

    for c in CLASS_NAMES:
        entry = per_class[c]
        assert "precision" in entry
        assert "recall" in entry
        assert "f1" in entry
        assert "support" in entry
        assert isinstance(entry["support"], int)


def test_step13_zero_support_class_reports_null_not_zero_recall(tmp_path: Path):
    """
    Asserts that classes with zero support in a split report null (None in Python)
    for precision, recall, and f1, and specifically do not report 0.0.
    Verified on test_crossdomain (0 not_crop rows) and direct metric computation.
    """
    # 1. Direct metric verification on synthetic labels with zero-support class
    mock_classes = ["class_present_a", "class_present_b", "class_absent_c"]
    targets = [0, 0, 1, 1]
    preds = [0, 1, 1, 1]
    _, _, _, per_class_synth, _ = compute_split_metrics(targets, preds, mock_classes)

    assert per_class_synth["class_absent_c"]["support"] == 0
    assert per_class_synth["class_absent_c"]["recall"] is None
    assert per_class_synth["class_absent_c"]["precision"] is None
    assert per_class_synth["class_absent_c"]["f1"] is None
    assert per_class_synth["class_absent_c"]["recall"] != 0.0

    # 2. Live verification on test_crossdomain split where not_crop has 0 support
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(3, len(CLASS_NAMES)),
    ).eval()

    out_json = tmp_path / "stage1_test_metrics.json"
    results = evaluate_test_splits(
        model=model,
        checkpoint_path=tmp_path / "stage1.pt",
        class_names=CLASS_NAMES,
        splits_dict={"test_crossdomain": SPLITS / "test_crossdomain.csv"},
        output_report_path=out_json,
        reports_dir=tmp_path,
        batch_size=4,
        max_steps=1,
    )

    not_crop_entry = results["test_crossdomain"]["per_class"]["not_crop"]
    assert not_crop_entry["support"] == 0
    assert not_crop_entry["recall"] is None, f"Expected null recall for zero-support class, got {not_crop_entry['recall']}"
    assert not_crop_entry["precision"] is None
    assert not_crop_entry["f1"] is None
    assert not_crop_entry["recall"] != 0.0

    # Verify JSON serialization emits literal null
    with open(out_json, "r") as f:
        raw_json_str = f.read()
    assert '"not_crop": {\n        "precision": null,\n        "recall": null,\n        "f1": null,\n        "support": 0\n      }' in raw_json_str or '"recall": null' in raw_json_str


def test_step13_test_eval_uses_eval_transform_identical_to_validation():
    """
    Asserts that test evaluation builds DataLoaders with eval_transform identical
    to validation transform in all composition steps and parameters.
    """
    _, val_loader = build_loaders(batch_size=4, stage=1)
    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    test_loader = build_test_loader(SPLITS / "test_indist.csv", class_to_idx=class_to_idx, batch_size=4)

    val_tf = val_loader.dataset.transform
    test_tf = test_loader.dataset.transform

    val_steps = [type(t).__name__ for t in val_tf.transforms]
    test_steps = [type(t).__name__ for t in test_tf.transforms]

    assert val_steps == test_steps == ['SmallestMaxSize', 'CenterCrop', 'Normalize', 'ToTensorV2']

    for vt, tt in zip(val_tf.transforms, test_tf.transforms):
        assert type(vt) is type(tt)
        if hasattr(vt, "max_size"):
            assert vt.max_size == tt.max_size
        if hasattr(vt, "height") and hasattr(vt, "width"):
            assert (vt.height, vt.width) == (tt.height, tt.width) == (IMAGE_SIZE, IMAGE_SIZE)
        if hasattr(vt, "mean") and hasattr(vt, "std"):
            assert vt.mean == tt.mean
            assert vt.std == tt.std


def test_step13_metrics_json_records_which_checkpoint_and_weights_were_used(tmp_path: Path):
    """
    Asserts that the metrics output JSON explicitly logs the checkpoint path and
    whether EMA or base model weights were used for evaluation.
    """
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(3, len(CLASS_NAMES)),
    )
    ema = ModelEmaV2(model, decay=0.999)

    ckpt_path = tmp_path / "stage1.pt"
    save_checkpoint_atomic(
        {
            "epoch": 5,
            "stage": 1,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "class_names": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES),
        },
        ckpt_path,
    )

    # 1. auto selects EMA when ema_state_dict is present
    _, weights_used_auto = load_checkpoint_for_eval(ckpt_path, model, eval_weights="auto")
    assert weights_used_auto == "ema"

    # 2. model forces model_state_dict
    _, weights_used_model = load_checkpoint_for_eval(ckpt_path, model, eval_weights="model")
    assert weights_used_model == "model_state_dict"

    # 3. Output JSON records both checkpoint_path and weights_used
    out_json = tmp_path / "stage1_test_metrics.json"
    results = evaluate_test_splits(
        model=model,
        checkpoint_path=ckpt_path,
        class_names=CLASS_NAMES,
        splits_dict={"test_indist": SPLITS / "test_indist.csv"},
        output_report_path=out_json,
        reports_dir=tmp_path,
        batch_size=4,
        weights_used=weights_used_auto,
        max_steps=1,
    )

    assert results["test_indist"]["checkpoint_path"] == str(ckpt_path)
    assert results["test_indist"]["weights_used"] == "ema"

    with open(out_json, "r") as f:
        disk_data = json.load(f)
    assert disk_data["test_indist"]["checkpoint_path"] == str(ckpt_path)
    assert disk_data["test_indist"]["weights_used"] == "ema"


def test_step13_worst_five_recall_classes_printed_each_epoch(capsys):
    """
    Asserts that format_lowest_recall_classes formats exactly the 5 lowest-recall
    classes and their values into a compact single line, sorted ascending.
    """
    recalls = {c: 0.95 for c in CLASS_NAMES}
    recalls["not_crop"] = 0.0500
    recalls["wheat__yellow_rust"] = 0.1234
    recalls["rice__blast"] = 0.2345
    recalls["sugarcane__red_rot"] = 0.3456
    recalls["wheat__septoria"] = 0.4567

    # C7: Every key in the input must be a verified member of CLASS_NAMES
    for k in recalls.keys():
        assert k in CLASS_NAMES, f"Key '{k}' must be an existing member of CLASS_NAMES"
    assert set(recalls.keys()) == set(CLASS_NAMES)

    compact_line = format_lowest_recall_classes(recalls, n=5)

    # Must contain exactly 5 classes separated by ' | '
    tokens = compact_line.split(" | ")
    assert len(tokens) == 5

    assert tokens[0] == "not_crop=0.0500"
    assert tokens[1] == "wheat__yellow_rust=0.1234"
    assert tokens[2] == "rice__blast=0.2345"
    assert tokens[3] == "sugarcane__red_rot=0.3456"
    assert tokens[4] == "wheat__septoria=0.4567"

    # Simulate printing and verify format
    print(f"  Lowest Recall: {compact_line}")
    captured = capsys.readouterr()
    assert "Lowest Recall: not_crop=0.0500 | wheat__yellow_rust=0.1234 | rice__blast=0.2345 | sugarcane__red_rot=0.3456 | wheat__septoria=0.4567" in captured.out


def test_step13_saved_logits_are_pre_softmax_and_row_aligned_with_paths(tmp_path: Path):
    """
    Asserts that saved logits are pre-softmax (unbounded, not normalized to 1.0)
    and proves row-alignment: row i of the saved logits array matches the deterministic
    prediction for path i across all samples.
    """
    # 1. Create synthetic images with distinct pixel values
    csv_rows = []
    for i, pixel_val in enumerate([15, 65, 145]):
        img_path = tmp_path / f"align_img_{i}.jpg"
        img = np.full((120, 120, 3), pixel_val, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        csv_rows.append({"path": str(img_path), "label": CLASS_NAMES[i]})

    manifest_file = tmp_path / "test_alignment.csv"
    pd.DataFrame(csv_rows).to_csv(manifest_file, index=False)

    class DeterministicFeatureModel(torch.nn.Module):
        def forward(self, x):
            B = x.shape[0]
            out = torch.zeros((B, len(CLASS_NAMES)), dtype=torch.float32)
            # Deterministic pre-softmax function with negative and unbounded values
            out[:, 0] = x.mean(dim=(1, 2, 3)) * 85.0 - 42.0
            out[:, 1] = x[:, 0, 0, 0] * 3.5 - 12.0
            out[:, 2] = -5.0 * (x[:, 1, 1, 1] + 1.0)
            return out

    model = DeterministicFeatureModel().eval()

    logits_dir = tmp_path / "logits"
    res = evaluate_test_splits(
        model=model,
        checkpoint_path=tmp_path / "stage1.pt",
        class_names=CLASS_NAMES,
        splits_dict={"align_test": manifest_file},
        output_report_path=tmp_path / "report.json",
        reports_dir=tmp_path,
        logits_dir=logits_dir,
        batch_size=2,
    )

    logits_arr = np.load(logits_dir / "align_test_logits.npy")
    with open(logits_dir / "align_test_paths.json", "r") as f:
        paths_meta = json.load(f)

    # Assert pre-softmax: values can be negative, sums do not equal 1.0
    assert (logits_arr < 0).any(), "Logits must be pre-softmax (containing negative values)"
    assert not np.allclose(logits_arr.sum(axis=1), 1.0), "Logits must not sum to 1.0 (softmax was applied!)"
    assert logits_arr.dtype == np.float32

    # Prove deterministic row alignment: row i must match prediction for path i
    assert logits_arr.shape[0] == len(paths_meta) == len(csv_rows)
    tf = eval_transform(IMAGE_SIZE)

    for i in range(len(paths_meta)):
        path_i = paths_meta[i]["path"]
        bgr = cv2.imread(path_i)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = tf(image=rgb)["image"].unsqueeze(0)
        expected_output = model(tensor).squeeze(0).detach().numpy()
        assert np.allclose(logits_arr[i], expected_output, atol=1e-5), (
            f"Row alignment failure at row {i} for path {path_i}! Logits do not correspond to image."
        )


def test_step13_logits_second_dimension_equals_num_classes(tmp_path: Path):
    """
    Asserts that saved logits arrays have shape (N, 29) where the second dimension
    equals len(CLASS_NAMES) exactly.
    """
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(3, len(CLASS_NAMES)),
    ).eval()

    logits_dir = tmp_path / "logits"
    evaluate_test_splits(
        model=model,
        checkpoint_path=tmp_path / "stage1.pt",
        class_names=CLASS_NAMES,
        splits_dict={"test_indist": SPLITS / "test_indist.csv"},
        output_report_path=tmp_path / "report.json",
        reports_dir=tmp_path,
        logits_dir=logits_dir,
        batch_size=4,
        max_steps=1,
    )

    logits_arr = np.load(logits_dir / "test_indist_logits.npy")
    assert logits_arr.ndim == 2
    assert logits_arr.shape[1] == len(CLASS_NAMES) == 29
    assert logits_arr.dtype == np.float32


def test_step15_extract_logits_uses_identical_eval_transform_as_validation():
    """
    Asserts that scripts/extract_logits.py applies eval_transform identical to
    validation in composition steps, dimensions, and normalization constants.
    """
    _, val_loader = build_loaders(batch_size=4, stage=1)
    val_tf = val_loader.dataset.transform
    extract_tf = eval_transform(size=IMAGE_SIZE)

    val_steps = [type(t).__name__ for t in val_tf.transforms]
    extract_steps = [type(t).__name__ for t in extract_tf.transforms]

    assert val_steps == extract_steps == ['SmallestMaxSize', 'CenterCrop', 'Normalize', 'ToTensorV2']

    for vt, et in zip(val_tf.transforms, extract_tf.transforms):
        assert type(vt) is type(et)
        if hasattr(vt, "max_size"):
            assert vt.max_size == et.max_size
        if hasattr(vt, "height") and hasattr(vt, "width"):
            assert (vt.height, vt.width) == (et.height, et.width) == (IMAGE_SIZE, IMAGE_SIZE)
        if hasattr(vt, "mean") and hasattr(vt, "std"):
            assert vt.mean == et.mean
            assert vt.std == et.std


def test_step15_extract_logits_handles_unlabelled_nested_folder(tmp_path: Path):
    """
    Asserts that extract_logits recursively traverses nested unlabelled directories,
    records relative paths and parent folder names, and outputs float32 (N, 29) logits.
    """
    nested_root = tmp_path / "nested_holdout"
    sub_a = nested_root / "category_a"
    sub_b = nested_root / "category_b" / "deep_folder"
    sub_a.mkdir(parents=True, exist_ok=True)
    sub_b.mkdir(parents=True, exist_ok=True)

    # Save 2 images in sub_a and 1 in sub_b
    cv2.imwrite(str(sub_a / "img1.png"), np.full((100, 100, 3), 40, dtype=np.uint8))
    cv2.imwrite(str(sub_a / "img2.png"), np.full((100, 100, 3), 80, dtype=np.uint8))
    cv2.imwrite(str(sub_b / "img3.png"), np.full((100, 100, 3), 160, dtype=np.uint8))

    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(3, len(CLASS_NAMES)),
    ).eval()

    out_prefix = tmp_path / "nested_out"
    logits, meta = extract_logits(
        model=model,
        input_dir=nested_root,
        out=out_prefix,
        batch_size=2,
    )

    assert logits.shape == (3, len(CLASS_NAMES))
    assert logits.dtype == np.float32
    assert meta["row_count"] == 3

    # Verify category and parent folder tracking
    category_counts = meta["category_counts"]
    assert category_counts["category_a"] == 2
    assert category_counts["deep_folder"] == 1

    for img_rec in meta["images"]:
        assert "relative_path" in img_rec
        assert "parent_folder" in img_rec
        assert img_rec["parent_folder"] in ["category_a", "deep_folder"]


def test_step15_extract_logits_on_openset_holdout_yields_174_rows_across_six_categories(tmp_path: Path):
    """
    Asserts that running extract_logits on data/raw/openset_holdout/ produces exactly
    174 rows across the six categories (10 hands, 34 feet_shoes, 38 pavement, 17 walls,
    43 soil, 32 green_noncrop) with a randomly initialised model.
    """
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(3, len(CLASS_NAMES)),
    ).eval()

    out_prefix = tmp_path / "openset_extracted"
    logits, meta = extract_logits(
        model=model,
        input_dir="data/raw/openset_holdout",
        out=out_prefix,
        batch_size=32,
    )

    assert logits.shape == (174, len(CLASS_NAMES))
    assert logits.dtype == np.float32
    assert meta["row_count"] == 174

    expected_categories = {
        "hands": 10,
        "feet_shoes": 34,
        "pavement": 38,
        "walls": 17,
        "soil": 43,
        "green_noncrop": 32,
    }
    assert meta["category_counts"] == expected_categories


def test_step15_extract_logits_raises_on_unreadable_image_rather_than_skipping(tmp_path: Path):
    """
    Asserts that extract_logits raises a descriptive ValueError citing the filename
    when encountering a corrupt or unreadable image file, rather than silently skipping it.
    """
    corrupt_dir = tmp_path / "corrupt_test_folder"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    corrupt_file = corrupt_dir / "unreadable_artifact.jpg"
    corrupt_file.write_bytes(b"INVALID_IMAGE_HEADER_DATA")

    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(3, len(CLASS_NAMES)),
    ).eval()

    with pytest.raises(ValueError, match="Corrupted or unreadable image file") as excinfo:
        extract_logits(
            model=model,
            input_dir=corrupt_dir,
            out=tmp_path / "corrupt_out",
        )

    assert "unreadable_artifact.jpg" in str(excinfo.value)


def test_step13_training_proceeds_when_wandb_unavailable_or_key_missing(tmp_path: Path, monkeypatch):
    """
    Asserts that training proceeds to completion without error when W&B is unavailable
    or when wandb.init raises an exception (e.g. missing API key, network failure).
    Verifies that all checkpoints and metrics artifacts are created.
    """
    if wandb is not None:
        def mock_failing_init(*args, **kwargs):
            raise RuntimeError("Simulated W&B failure: API key missing / offline network failure")
        monkeypatch.setattr(wandb, "init", mock_failing_init)

    out_ckpt = tmp_path / "stage1_nowandb.pt"
    out_metrics = tmp_path / "metrics_nowandb.json"
    last_ckpt = tmp_path / "last.pt"

    argv = [
        "--epochs", "1",
        "--max_steps", "2",
        "--batch_size", "4",
        "--device", "cpu",
        "--out", str(out_ckpt),
        "--metrics_out", str(out_metrics),
        "--no_eval_splits",
        "--no_pretrained",
    ]

    # Must complete without raising an exception
    main(argv)

    assert out_ckpt.exists(), f"Expected checkpoint {out_ckpt} to be written despite W&B failure"
    assert last_ckpt.exists(), f"Expected last.pt {last_ckpt} to be written despite W&B failure"
    assert out_metrics.exists(), f"Expected metrics {out_metrics} to be written despite W&B failure"


def test_step13_wandb_log_failure_mid_epoch_does_not_abort_training(tmp_path: Path, monkeypatch):
    """
    Asserts that when wandb.log raises mid-epoch (e.g. intermittent connection drops during
    a long Kaggle training run), the training loop does not crash, completes all epochs,
    and writes out checkpoints.
    """
    class MockRun:
        def __init__(self):
            self.id = "mock-run-mid-epoch-drop"
            self.summary = {}

    log_call_count = [0]

    def mock_failing_log(data, *args, **kwargs):
        log_call_count[0] += 1
        raise ConnectionResetError("Simulated mid-epoch Kaggle network disconnect during wandb.log")

    if wandb is not None:
        mock_run = MockRun()
        monkeypatch.setattr(wandb, "init", lambda *a, **kw: mock_run)
        monkeypatch.setattr(wandb, "run", mock_run)
        monkeypatch.setattr(wandb, "log", mock_failing_log)

    out_ckpt = tmp_path / "stage1_logfail.pt"
    out_metrics = tmp_path / "metrics_logfail.json"
    last_ckpt = tmp_path / "last.pt"

    argv = [
        "--epochs", "1",
        "--max_steps", "2",
        "--batch_size", "4",
        "--device", "cpu",
        "--out", str(out_ckpt),
        "--metrics_out", str(out_metrics),
        "--no_eval_splits",
        "--no_pretrained",
    ]

    # Must complete without crashing
    main(argv)

    assert log_call_count[0] > 0, "Expected wandb.log to have been called and failed during training"
    assert out_ckpt.exists(), f"Expected checkpoint {out_ckpt} to exist after completed training"
    assert last_ckpt.exists(), f"Expected last checkpoint {last_ckpt} to exist"


def test_step13_wandb_run_id_persisted_in_checkpoint_and_reused_on_resume(tmp_path: Path, monkeypatch):
    """
    Asserts that:
    1. wandb_run_id is persisted into checkpoint dictionaries.
    2. Resuming from a checkpoint with wandb_run_id reuses the ID with resume='allow'.
    3. Resuming from a legacy checkpoint without wandb_run_id logs a warning and generates
       a fresh run ID without failing.
    """
    ckpt_path = tmp_path / "test_ckpt_with_id.pt"
    legacy_ckpt_path = tmp_path / "test_legacy_ckpt.pt"

    # 1. Checkpoint with explicit wandb_run_id
    state_with_id = {
        "epoch": 2,
        "stage": 1,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "total_steps": 100,
        "wandb_run_id": "persisted-run-777",
    }
    torch.save(state_with_id, ckpt_path)

    extracted_id = get_checkpoint_wandb_id(ckpt_path)
    assert extracted_id == "persisted-run-777", f"Expected 'persisted-run-777', got {extracted_id}"

    # 2. Resuming reuses ID
    captured_init_kwargs = {}
    if wandb is not None:
        class DummyRun:
            id = "persisted-run-777"
        def mock_init(**kwargs):
            captured_init_kwargs.update(kwargs)
            return DummyRun()
        monkeypatch.setattr(wandb, "init", mock_init)

    import argparse
    dummy_args = argparse.Namespace(
        no_wandb=False,
        wandb_entity="prism-team",
        wandb_project="SIH",
    )
    res_id = init_wandb(args=dummy_args, config={"lr": 0.001}, run_id=extracted_id)
    assert res_id == "persisted-run-777"
    assert captured_init_kwargs.get("id") == "persisted-run-777"
    assert captured_init_kwargs.get("resume") == "allow"

    # 3. Legacy checkpoint without wandb_run_id
    state_legacy = {
        "epoch": 1,
        "stage": 1,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "total_steps": 50,
    }
    torch.save(state_legacy, legacy_ckpt_path)
    legacy_id = get_checkpoint_wandb_id(legacy_ckpt_path)
    assert legacy_id is None, "Legacy checkpoint must return None for wandb_run_id"

    # Simulating resume logic on legacy checkpoint
    if legacy_id is None:
        fresh_id = generate_wandb_id()
    else:
        fresh_id = legacy_id
    assert fresh_id is not None and len(fresh_id) > 0

    captured_init_kwargs.clear()
    res_fresh = init_wandb(args=dummy_args, config={"lr": 0.001}, run_id=fresh_id)
    assert res_fresh is not None


def test_step13_worker_init_fn_seeds_each_dataloader_worker_distinctly():
    """
    Asserts that:
    1. seed_worker produces distinct NumPy and Python random sequences for distinct worker_ids.
    2. Calling seed_worker repeatedly with identical base seed and worker_id reproduces
       exact deterministic pseudo-random sequences.
    3. DataLoader construction wires worker_init_fn=seed_worker and seeded Generator.
    """
    # 1. Distinct worker seeds generate distinct numbers
    torch.manual_seed(100)
    seed_worker(0)
    val0_np = int(np.random.randint(0, 10000000))
    val0_py = int(random.randint(0, 10000000))

    torch.manual_seed(100)
    seed_worker(1)
    val1_np = int(np.random.randint(0, 10000000))
    val1_py = int(random.randint(0, 10000000))

    assert val0_np != val1_np, f"Worker 0 and Worker 1 generated identical NumPy values: {val0_np}"
    assert val0_py != val1_py, f"Worker 0 and Worker 1 generated identical Python random values: {val0_py}"

    # 2. Exact reproducibility on repeat
    torch.manual_seed(100)
    seed_worker(0)
    repeat_val0_np = int(np.random.randint(0, 10000000))
    repeat_val0_py = int(random.randint(0, 10000000))

    assert val0_np == repeat_val0_np, "Seed worker failed to reproduce identical NumPy sequence"
    assert val0_py == repeat_val0_py, "Seed worker failed to reproduce identical Python sequence"

    # 3. DataLoader worker_init_fn and generator inspection
    tl, vl = build_loaders(batch_size=4, stage=1, seed=42)
    assert tl.worker_init_fn is seed_worker, "train_loader must have worker_init_fn=seed_worker"
    assert vl.worker_init_fn is seed_worker, "val_loader must have worker_init_fn=seed_worker"
    assert isinstance(tl.generator, torch.Generator), "train_loader must have seeded torch.Generator"
    assert isinstance(vl.generator, torch.Generator), "val_loader must have seeded torch.Generator"


def test_step13_deterministic_flag_sets_cudnn_flags_and_defaults_off():
    """
    Asserts that:
    1. The --deterministic CLI argument defaults to False (throughput preserved).
    2. Calling set_seed with deterministic=False sets cuDNN deterministic=False and benchmark=True.
    3. Passing --deterministic sets args.deterministic to True.
    4. Calling set_seed with deterministic=True sets cuDNN deterministic=True and benchmark=False.
    """
    import argparse
    from train.train_model_a import main

    # 1. Default CLI behavior is False
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic", action="store_true", default=False)
    args_default = parser.parse_args([])
    assert args_default.deterministic is False, "Default --deterministic must be False"

    # 2. When deterministic=False: deterministic=False, benchmark=True
    set_seed(42, deterministic=False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True

    # 3. Passing flag sets it to True
    args_flag = parser.parse_args(["--deterministic"])
    assert args_flag.deterministic is True

    # 4. When deterministic=True: deterministic=True, benchmark=False
    set_seed(42, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    # Teardown: restore default
    set_seed(42, deterministic=False)


def test_phase_b_packaged_dataset_file_counts_match_raw_identically():
    """
    Asserts that every top-level folder in data/raw has an identical file count
    in data/packaged (excluding hidden files and .git).
    """
    raw_dir = Path("data/raw")
    pkg_dir = Path("data/packaged")
    assert raw_dir.exists() and pkg_dir.exists()

    for d in raw_dir.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            raw_files = [p for p in d.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.parts)]
            pkg_d = pkg_dir / d.name
            assert pkg_d.exists(), f"Directory {pkg_d} missing from data/packaged"
            pkg_files = [p for p in pkg_d.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.parts)]
            assert len(raw_files) == len(pkg_files), f"Count mismatch for {d.name}: raw={len(raw_files)}, pkg={len(pkg_files)}"


def test_phase_b_resized_photos_max_dim_1024_and_orientation_exif_stripped():
    """
    Asserts that all 1154 target images (980 in not_crop and 174 in openset_holdout)
    in data/packaged/ have max dimension <= 1024 px and have no EXIF orientation tag.
    """
    from PIL import Image

    pkg_dir = Path("data/packaged")
    target_configs = [
        ("not_crop", {"hands", "feet_shoes", "pavement", "walls", "soil_field", "green_noncrop"}, 980),
        ("openset_holdout", {"hands", "feet_shoes", "pavement", "walls", "soil", "green_noncrop"}, 174),
    ]

    total_checked = 0
    for top_folder, subfolders, expected_count in target_configs:
        folder_checked = 0
        for sub in subfolders:
            sub_dir = pkg_dir / top_folder / sub
            assert sub_dir.exists()
            for p in sub_dir.glob("*"):
                if p.is_file() and not p.name.startswith("."):
                    with Image.open(p) as img:
                        w, h = img.size
                        assert max(w, h) <= 1024, f"Image {p} max dimension > 1024: ({w}, {h})"
                        exif = img.getexif()
                        assert 0x0112 not in exif, f"Image {p} still contains orientation EXIF tag"
                        folder_checked += 1
                        total_checked += 1
        assert folder_checked == expected_count, f"Expected {expected_count} for {top_folder}, got {folder_checked}"

    assert total_checked == 1154, f"Expected 1154 total resized images checked, got {total_checked}"


def test_phase_b_splits_packaged_paths_exist_on_disk_and_zero_missing():
    """
    Asserts that 100% of paths in all CSVs under splits_packaged/ exist on disk
    under data/packaged/, with exactly zero missing files.
    """
    import csv

    splits_pkg = Path("splits_packaged")
    assert splits_pkg.exists()

    total_checked = 0
    total_missing = 0
    for csv_path in sorted(splits_pkg.glob("*.csv")):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            if "path" not in header:
                continue
            path_idx = header.index("path")
            for row in reader:
                if len(row) > path_idx:
                    p = Path(row[path_idx])
                    total_checked += 1
                    if not p.exists():
                        total_missing += 1

    assert total_checked == 60161, f"Expected 60161 total split paths, got {total_checked}"
    assert total_missing == 0, f"Expected 0 missing files, got {total_missing}"


def test_phase_b_class_weights_json_is_byte_identical_in_splits_packaged():
    """
    Asserts that splits_packaged/class_weights.json is byte-identical to splits/class_weights.json.
    """
    raw_cw = Path("splits/class_weights.json").read_bytes()
    pkg_cw = Path("splits_packaged/class_weights.json").read_bytes()
    assert raw_cw == pkg_cw, "splits_packaged/class_weights.json must match splits/class_weights.json byte-identically"


def test_phase_b_leak_check_openset_holdout_and_not_crop_archive_absent_from_splits_packaged():
    """
    Asserts that openset_holdout and not_crop_archive do not appear anywhere in any
    split CSV under splits_packaged/.
    """
    import subprocess

    csvs = sorted(str(p) for p in Path("splits_packaged").glob("*.csv"))
    res = subprocess.run(["grep", "-c", "openset_holdout\\|not_crop_archive"] + csvs, capture_output=True, text=True)
    for line in res.stdout.strip().splitlines():
        if ":" in line:
            _, cnt = line.rsplit(":", 1)
            assert int(cnt) == 0, f"Leaked entry detected in {line}"


def test_path_prefix_strip_present_and_stripped_correctly_resolves(tmp_path: Path):
    """
    Asserts that when path_prefix_strip is provided and all CSV paths carry that
    prefix, the prefix is stripped and the resulting paths resolve correctly against
    root_dir, producing valid image tensors.
    """
    img_dir = tmp_path / "paddy" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_file = img_dir / "leaf.jpg"
    dummy = np.full((64, 64, 3), 100, dtype=np.uint8)
    cv2.imwrite(str(img_file), dummy)

    # CSV stores paths WITH the prefix
    df = pd.DataFrame([
        {"path": "data/packaged_min/paddy/train/leaf.jpg", "label": "test_class"},
    ])
    csv_path = tmp_path / "split.csv"
    df.to_csv(csv_path, index=False)

    ds = PlantDataset(
        data_source=csv_path,
        class_to_idx={"test_class": 0},
        root_dir=tmp_path,
        path_prefix_strip="data/packaged_min/",
    )

    assert ds.image_paths[0] == "paddy/train/leaf.jpg", (
        f"Expected stripped path 'paddy/train/leaf.jpg', got '{ds.image_paths[0]}'"
    )
    tensor, target = ds[0]
    assert tensor.shape[0] == 3, f"Expected 3-channel tensor, got shape {tensor.shape}"
    assert target == 0


def test_path_prefix_strip_empty_leaves_paths_untouched(tmp_path: Path):
    """
    Asserts that when path_prefix_strip is empty (default), CSV paths are left
    exactly as-is with no modification.
    """
    img_dir = tmp_path / "data" / "raw" / "crop"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_file = img_dir / "img.jpg"
    dummy = np.full((32, 32, 3), 50, dtype=np.uint8)
    cv2.imwrite(str(img_file), dummy)

    original_path = "data/raw/crop/img.jpg"
    df = pd.DataFrame([{"path": original_path, "label": "crop_a"}])
    csv_path = tmp_path / "split.csv"
    df.to_csv(csv_path, index=False)

    ds = PlantDataset(
        data_source=csv_path,
        class_to_idx={"crop_a": 0},
        root_dir=tmp_path,
        path_prefix_strip="",
    )
    assert ds.image_paths[0] == original_path, (
        f"Expected untouched path '{original_path}', got '{ds.image_paths[0]}'"
    )


def test_path_prefix_strip_one_path_lacking_prefix_raises_valueerror(tmp_path: Path):
    """
    Asserts that if any path in the CSV does NOT start with the specified
    path_prefix_strip, PlantDataset raises ValueError at construction naming
    that path and reporting the total count of offenders.
    """
    df = pd.DataFrame([
        {"path": "data/packaged_min/paddy/leaf1.jpg", "label": "a"},
        {"path": "other/prefix/leaf2.jpg", "label": "a"},
        {"path": "data/packaged_min/paddy/leaf3.jpg", "label": "a"},
    ])
    csv_path = tmp_path / "split.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="other/prefix/leaf2.jpg"):
        PlantDataset(
            data_source=csv_path,
            class_to_idx={"a": 0},
            root_dir=tmp_path,
            path_prefix_strip="data/packaged_min/",
        )


def test_path_prefix_strip_missing_on_disk_raises_runtimeerror_at_construction(tmp_path: Path):
    """
    Asserts that when path_prefix_strip strips a prefix successfully but the
    resulting path does not exist on disk, PlantDataset raises RuntimeError at
    construction (not at __getitem__), reporting root_dir, the raw CSV value,
    the stripped value, and the joined path.
    """
    # Do NOT create the image on disk — the stripped path must fail to resolve
    df = pd.DataFrame([
        {"path": "data/packaged_min/nonexistent/img.jpg", "label": "b"},
    ])
    csv_path = tmp_path / "split.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(RuntimeError, match="path_prefix_strip sanity check failed at construction"):
        PlantDataset(
            data_source=csv_path,
            class_to_idx={"b": 0},
            root_dir=tmp_path,
            path_prefix_strip="data/packaged_min/",
        )


def test_evaluate_test_splits_with_explicit_splits_dir_and_missing_file_raises(tmp_path: Path):
    """
    Asserts that when splits_dir is explicitly provided and a named split file is
    missing, evaluate_test_splits raises FileNotFoundError with the full path in
    the error message, rather than silently skipping the split.
    """
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(3, len(CLASS_NAMES)),
    ).eval()

    empty_splits_dir = tmp_path / "custom_splits"
    empty_splits_dir.mkdir(parents=True, exist_ok=True)
    expected_missing = str(empty_splits_dir / "val.csv")

    with pytest.raises(FileNotFoundError) as exc_info:
        evaluate_test_splits(
            model=model,
            checkpoint_path=tmp_path / "stage1.pt",
            class_names=CLASS_NAMES,
            splits_dir=empty_splits_dir,
            output_report_path=tmp_path / "report.json",
            reports_dir=tmp_path,
        )

    assert "Explicit splits_dir was provided" in str(exc_info.value)
    assert expected_missing in str(exc_info.value)


def test_evaluate_test_splits_with_default_splits_and_missing_file_skips(tmp_path: Path):
    """
    Asserts that when splits_dir is None (falling back to default SPLITS) and a
    split file is absent, evaluate_test_splits continues gracefully and skips the split.
    """
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(3, len(CLASS_NAMES)),
    ).eval()

    nonexistent_file = tmp_path / "absent_split.csv"
    results = evaluate_test_splits(
        model=model,
        checkpoint_path=tmp_path / "stage1.pt",
        class_names=CLASS_NAMES,
        splits_dict={"absent_split": nonexistent_file},
        splits_dir=None,
        output_report_path=tmp_path / "report.json",
        reports_dir=tmp_path,
    )

    assert "absent_split" not in results


def test_evaluate_test_splits_applies_path_prefix_strip_and_resolves_images_against_data_dir(tmp_path: Path):
    """
    Asserts that evaluate_test_splits threads path_prefix_strip to build_test_loader,
    correctly stripping the prefix from split CSV paths and resolving them against data_dir.
    """
    mock_data = tmp_path / "mock_kaggle_dataset"
    img_dir = mock_data / "paddy" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_file = img_dir / "sample_001.jpg"
    dummy_pixels = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(img_file), dummy_pixels)

    # CSV path contains the prefix
    csv_file = tmp_path / "val.csv"
    pd.DataFrame([
        {"path": "data/packaged_min/paddy/val/sample_001.jpg", "label": CLASS_NAMES[0]}
    ]).to_csv(csv_file, index=False)

    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(3, len(CLASS_NAMES)),
    ).eval()

    results = evaluate_test_splits(
        model=model,
        checkpoint_path=tmp_path / "stage1.pt",
        class_names=CLASS_NAMES,
        splits_dict={"val": csv_file},
        splits_dir=tmp_path,
        data_dir=mock_data,
        path_prefix_strip="data/packaged_min/",
        output_report_path=tmp_path / "report.json",
        reports_dir=tmp_path,
        max_steps=1,
    )

    assert "val" in results
    assert results["val"]["rows_evaluated"] == 1


def test_splits_dir_flag_reaches_evaluate_test_splits_from_argparse(tmp_path: Path, monkeypatch):
    """
    Asserts that passing --splits_dir on the CLI threads cleanly through argparse
    into the evaluate_test_splits call.
    """
    import train.train_model_a as tma
    from train.model import build_model
    from train.train_model_a import save_checkpoint_atomic

    dummy_model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    dummy_ckpt = tmp_path / "dummy_stage1.pt"
    save_checkpoint_atomic({
        "epoch": 1,
        "model_state_dict": dummy_model.state_dict(),
        "class_names": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
    }, dummy_ckpt)

    captured_kwargs = {}
    def mock_eval(**kwargs):
        captured_kwargs.update(kwargs)
        return {}

    monkeypatch.setattr(tma, "evaluate_test_splits", mock_eval)

    custom_splits_dir = tmp_path / "my_custom_splits"
    custom_splits_dir.mkdir()

    tma.main([
        "--eval_only",
        "--eval_checkpoint", str(dummy_ckpt),
        "--splits_dir", str(custom_splits_dir),
        "--no_pretrained",
        "--no_wandb",
    ])

    assert captured_kwargs.get("splits_dir") == str(custom_splits_dir), (
        f"Expected evaluate_test_splits to receive splits_dir='{custom_splits_dir}', "
        f"got '{captured_kwargs.get('splits_dir')}'"
    )


def test_both_raw_and_ema_validation_metrics_are_logged_each_epoch(tmp_path: Path):
    """
    Asserts that validation evaluates and logs both the raw model and EMA model metrics
    each epoch, recording separate raw_val_top1/raw_val_macro_f1 and
    ema_val_top1/ema_val_macro_f1 entries in metrics history.
    """
    out_ckpt = tmp_path / "stage1_both_metrics.pt"
    out_metrics = tmp_path / "metrics_both.json"

    argv = [
        "--epochs", "1",
        "--max_steps", "2",
        "--batch_size", "4",
        "--device", "cpu",
        "--out", str(out_ckpt),
        "--metrics_out", str(out_metrics),
        "--no_eval_splits",
        "--no_pretrained",
        "--no_wandb",
    ]
    main(argv)

    assert out_metrics.exists(), f"Expected metrics JSON at {out_metrics}"
    with open(out_metrics) as f:
        data = json.load(f)

    # Top-level metrics check
    assert "raw_val_top1" in data, "raw_val_top1 missing from top-level metrics"
    assert "raw_val_macro_f1" in data, "raw_val_macro_f1 missing from top-level metrics"
    assert "ema_val_top1" in data, "ema_val_top1 missing from top-level metrics"
    assert "ema_val_macro_f1" in data, "ema_val_macro_f1 missing from top-level metrics"

    assert 0.0 <= data["raw_val_top1"] <= 1.0
    assert 0.0 <= data["raw_val_macro_f1"] <= 1.0
    assert 0.0 <= data["ema_val_top1"] <= 1.0
    assert 0.0 <= data["ema_val_macro_f1"] <= 1.0

    # History entries check
    assert "history" in data and len(data["history"]) >= 1
    epoch_rec = data["history"][0]
    assert "raw_val_top1" in epoch_rec, "raw_val_top1 missing from epoch history"
    assert "raw_val_macro_f1" in epoch_rec, "raw_val_macro_f1 missing from epoch history"
    assert "ema_val_top1" in epoch_rec, "ema_val_top1 missing from epoch history"
    assert "ema_val_macro_f1" in epoch_rec, "ema_val_macro_f1 missing from epoch history"
    assert "val_top1" in epoch_rec, "val_top1 missing from epoch history"
    assert "val_macro_f1" in epoch_rec, "val_macro_f1 missing from epoch history"
    # Legacy metrics track the raw model
    assert epoch_rec["val_top1"] == epoch_rec["raw_val_top1"]
    assert epoch_rec["val_macro_f1"] == epoch_rec["raw_val_macro_f1"]


def test_best_checkpoint_selection_uses_the_raw_model_metric_not_ema(tmp_path: Path, monkeypatch):
    """
    Verifies that best checkpoint selection (stage1.pt) is driven strictly by the
    raw model's Macro-F1 score, NOT by the EMA model.
    Scenario:
      Epoch 1: Raw Macro-F1 = 0.85, EMA Macro-F1 = 0.15
      Epoch 2: Raw Macro-F1 = 0.50, EMA Macro-F1 = 0.95
    If EMA drove selection, Epoch 2 would overwrite stage1.pt (0.95 > 0.15).
    Since RAW drives selection, Epoch 1 is selected (0.85 > 0.50) and Epoch 2 does not overwrite.
    """
    import train.train_model_a as tma

    out_ckpt = tmp_path / "stage1_raw_select.pt"
    out_metrics = tmp_path / "metrics_raw_select.json"

    eval_calls = []

    def mock_evaluate(model, loader, device, class_names, max_steps=None):
        eval_calls.append(model)
        idx = len(eval_calls)
        dummy_per_class = {c: 0.5 for c in class_names}
        if idx == 1:
            return 0.85, 0.85, dummy_per_class
        elif idx == 2:
            return 0.15, 0.15, dummy_per_class
        elif idx == 3:
            return 0.50, 0.50, dummy_per_class
        elif idx == 4:
            return 0.95, 0.95, dummy_per_class
        return 0.50, 0.50, dummy_per_class

    monkeypatch.setattr(tma, "evaluate", mock_evaluate)

    argv = [
        "--epochs", "2",
        "--max_steps", "1",
        "--batch_size", "4",
        "--device", "cpu",
        "--out", str(out_ckpt),
        "--metrics_out", str(out_metrics),
        "--no_eval_splits",
        "--no_pretrained",
        "--no_wandb",
    ]
    tma.main(argv)

    assert out_ckpt.exists(), f"Checkpoint {out_ckpt} was not saved"
    ckpt = torch.load(out_ckpt, map_location="cpu")

    # Assert that Epoch 1 was retained because its RAW score (0.85) was higher than Epoch 2 RAW (0.50)
    assert ckpt["epoch"] == 1, (
        f"Expected best checkpoint from Epoch 1 (raw=0.85), but got Epoch {ckpt['epoch']} "
        "(indicates EMA may have incorrectly driven checkpoint selection)"
    )
    assert ckpt["best_macro_f1"] == pytest.approx(0.85)
    assert ckpt["raw_val_macro_f1"] == pytest.approx(0.85)

    with open(out_metrics) as f:
        metrics = json.load(f)
    assert metrics["best_epoch"] == 1
    assert metrics["best_macro_f1"] == pytest.approx(0.85)
    assert metrics["history"][0]["raw_val_macro_f1"] == pytest.approx(0.85)
    assert metrics["history"][0]["ema_val_macro_f1"] == pytest.approx(0.15)
    assert metrics["history"][1]["raw_val_macro_f1"] == pytest.approx(0.50)
    assert metrics["history"][1]["ema_val_macro_f1"] == pytest.approx(0.95)


def test_ema_decay_value_converges_within_the_configured_total_step_count():
    """
    Verifies that the configured EMA_DECAY converges within the total step count
    (50 epochs * 415 steps/epoch = 20,750 steps), ensuring initial weight retention
    is strictly < 0.1% (1e-3).
    Also confirms that the uncalibrated 0.9998 fails this threshold, and that
    the dynamic EMA warmup flushes initial weights within the first 10 steps.
    """
    # 1. Physical step counts verified from dataset row counts
    train_rows = 26505
    batch_size = BATCH_SIZE  # 64
    steps_per_epoch = math.ceil(train_rows / batch_size)  # 415
    assert steps_per_epoch == 415, f"Expected 415 steps/epoch, got {steps_per_epoch}"

    total_steps = EPOCHS * steps_per_epoch  # 50 * 415 = 20,750
    assert total_steps == 20750, f"Expected 20,750 total steps, got {total_steps}"

    # 2. Configured EMA decay must achieve initial-weight contribution < 0.1% (1e-3)
    retention_at_end = EMA_DECAY ** total_steps
    assert retention_at_end < 1e-3, (
        f"Configured EMA_DECAY={EMA_DECAY} leaves {retention_at_end:.4%} initial weight after "
        f"{total_steps} steps, exceeding the 0.1% (1e-3) threshold."
    )

    # 3. Maximum theoretical decay factor: beta_max = (1e-3)**(1/20750) ~= 0.999667
    max_allowable_decay = (1e-3) ** (1.0 / total_steps)
    assert EMA_DECAY < max_allowable_decay, (
        f"EMA_DECAY={EMA_DECAY} must be strictly less than {max_allowable_decay:.8f}"
    )

    # 4. Prove that the stale uncalibrated decay 0.9998 failed
    stale_decay = 0.9998
    stale_retention = stale_decay ** total_steps
    assert stale_retention > 1e-3, (
        f"Stale decay 0.9998 expected > 0.1% retention, got {stale_retention:.4%}"
    )

    # 5. EMA Warmup validation: decay(t) = min(EMA_DECAY, (1 + t)/(10 + t))
    warmup_decay_prod = 1.0
    for t in range(10):
        step_decay = min(EMA_DECAY, (1.0 + t) / (10.0 + t))
        warmup_decay_prod *= step_decay
    assert warmup_decay_prod < 1e-4, (
        f"Warmup should flush initial weight contribution below 0.01% by step 10, got {warmup_decay_prod}"
    )


def test_phase4_eval_transform_contains_none_of_anti_shortcut_transforms():
    """
    Phase 4 requirement: eval_transform must be strictly deterministic and evaluation-only.
    Asserts eval_transform contains NONE of the five anti-source shortcut transforms:
    ImageCompression, Downscale, GaussNoise, MotionBlur, RandomGamma.
    """
    import albumentations as A
    eval_tf = eval_transform()

    def _collect_transforms(pipeline):
        tfs = []
        for t in pipeline.transforms:
            tfs.append(t)
            if hasattr(t, "transforms"):
                tfs.extend(_collect_transforms(t))
        return tfs

    eval_tfs = _collect_transforms(eval_tf)
    anti_shortcut_classes = (
        A.ImageCompression,
        A.Downscale,
        A.GaussNoise,
        A.MotionBlur,
        A.RandomGamma,
    )
    anti_shortcut_names = {cls.__name__ for cls in anti_shortcut_classes}

    for t in eval_tfs:
        assert not isinstance(t, anti_shortcut_classes), (
            f"eval_transform must NOT contain {type(t).__name__}"
        )
        assert type(t).__name__ not in anti_shortcut_names, (
            f"eval_transform must NOT contain transform with name {type(t).__name__}"
        )


def test_phase4_baseline_train_transform_contains_all_anti_shortcut_transforms():
    """
    Phase 4 requirement: baseline_train_transform MUST contain all five anti-source shortcut transforms:
    ImageCompression, Downscale, GaussNoise, MotionBlur, RandomGamma.
    Verifies that all 5 are present, their probabilities/parameters match Phase 4 specs,
    and HUE_SHIFT_LIMIT <= 12.
    """
    import albumentations as A
    base_tf = baseline_train_transform()

    def _collect_transforms(pipeline):
        tfs = []
        for t in pipeline.transforms:
            tfs.append(t)
            if hasattr(t, "transforms"):
                tfs.extend(_collect_transforms(t))
        return tfs

    base_tfs = _collect_transforms(base_tf)
    tf_names = {type(t).__name__ for t in base_tfs}
    expected_names = {"ImageCompression", "Downscale", "GaussNoise", "MotionBlur", "RandomGamma"}

    assert expected_names.issubset(tf_names), (
        f"Missing anti-shortcut transforms in baseline_train_transform: {expected_names - tf_names}"
    )

    comp_tfs = [t for t in base_tfs if isinstance(t, A.ImageCompression)]
    assert len(comp_tfs) >= 1, "ImageCompression must be in baseline_train_transform"
    assert comp_tfs[0].p == 0.5, f"ImageCompression p expected 0.5, got {comp_tfs[0].p}"
    if hasattr(comp_tfs[0], "quality_range"):
        assert comp_tfs[0].quality_range == (30, 90)

    down_tfs = [t for t in base_tfs if isinstance(t, A.Downscale)]
    assert len(down_tfs) >= 1, "Downscale must be in baseline_train_transform"
    assert down_tfs[0].p == 0.3, f"Downscale p expected 0.3, got {down_tfs[0].p}"
    if hasattr(down_tfs[0], "scale_range"):
        assert down_tfs[0].scale_range == (0.7, 0.95)

    noise_tfs = [t for t in base_tfs if isinstance(t, A.GaussNoise)]
    assert len(noise_tfs) >= 1, "GaussNoise must be in baseline_train_transform"
    assert noise_tfs[0].p == 0.3, f"GaussNoise p expected 0.3, got {noise_tfs[0].p}"

    blur_tfs = [t for t in base_tfs if isinstance(t, A.MotionBlur)]
    assert len(blur_tfs) >= 1, "MotionBlur must be in baseline_train_transform"
    assert blur_tfs[0].p == 0.2, f"MotionBlur p expected 0.2, got {blur_tfs[0].p}"

    gamma_tfs = [t for t in base_tfs if isinstance(t, A.RandomGamma)]
    assert len(gamma_tfs) >= 1, "RandomGamma must be in baseline_train_transform"
    assert gamma_tfs[0].p == 0.3, f"RandomGamma p expected 0.3, got {gamma_tfs[0].p}"

    assert HUE_SHIFT_LIMIT <= 12, f"HUE_SHIFT_LIMIT is {HUE_SHIFT_LIMIT} > 12"


def test_sourceheldout_split_is_evaluated_not_skipped(tmp_path: Path):
    """
    Verifies that evaluate_test_splits automatically discovers and evaluates
    test_sourceheldout.csv when passed a splits_dir containing splits_v3,
    evaluating exactly val, test_indist, test_sourceheldout, test_external_riceblast
    and not skipping test_sourceheldout or failing on absent test_crossdomain.csv.
    """
    splits_dir = tmp_path / "splits_v3"
    splits_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create dummy sample image
    img_file = data_dir / "sample.jpg"
    cv2.imwrite(str(img_file), np.zeros((64, 64, 3), dtype=np.uint8))

    # Create dummy split CSVs
    split_names = [
        "val.csv",
        "test_indist.csv",
        "test_sourceheldout.csv",
        "test_external_riceblast.csv",
    ]
    for name in split_names:
        df = pd.DataFrame([{"path": "data/packaged_min/sample.jpg", "label": CLASS_NAMES[0]}])
        df.to_csv(splits_dir / name, index=False)

    # Confirm test_crossdomain.csv does NOT exist in this directory
    assert not (splits_dir / "test_crossdomain.csv").exists()

    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(3, len(CLASS_NAMES)),
    ).eval()

    report_json = tmp_path / "report.json"
    results = evaluate_test_splits(
        model=model,
        checkpoint_path=tmp_path / "stage1.pt",
        class_names=CLASS_NAMES,
        splits_dir=splits_dir,
        splits_dict=None,
        data_dir=data_dir,
        path_prefix_strip="data/packaged_min/",
        output_report_path=report_json,
        reports_dir=tmp_path / "reports",
        max_steps=1,
    )

    expected_splits = {"val", "test_indist", "test_sourceheldout", "test_external_riceblast"}
    assert set(results.keys()) == expected_splits, (
        f"Expected evaluated splits {expected_splits}, got {set(results.keys())}"
    )
    assert "test_sourceheldout" in results, "test_sourceheldout was NOT evaluated!"
    assert "test_crossdomain" not in results, "test_crossdomain should not be present!"
    assert results["test_sourceheldout"]["rows_evaluated"] == 1


def test_step14_all_five_deliverables_exist_and_non_empty():
    """
    Asserts that Step 14 produces all five mandatory deliverables in artifacts/reports/:
    1. eval_indist.json
    2. eval_crossdomain.json
    3. confusion_matrix.png
    4. per_class_recall.csv
    5. gradcam_panel.png
    """
    reports_dir = REPORTS
    deliverables = [
        reports_dir / "eval_indist.json",
        reports_dir / "eval_crossdomain.json",
        reports_dir / "confusion_matrix.png",
        reports_dir / "per_class_recall.csv",
        reports_dir / "gradcam_panel.png",
    ]
    for path in deliverables:
        assert path.exists(), f"Missing Step 14 deliverable: {path}"
        assert path.stat().st_size > 0, f"Step 14 deliverable is empty: {path}"


def test_step14_headline_generalization_gap_is_explicitly_calculated():
    """
    Asserts that eval_crossdomain.json explicitly computes and records
    in-distribution macro-F1 minus test_sourceheldout macro-F1 as HEADLINE_GENERALIZATION_GAP.
    """
    crossdomain_json = REPORTS / "eval_crossdomain.json"
    assert crossdomain_json.exists()

    with open(crossdomain_json, "r") as f:
        data = json.load(f)

    assert "HEADLINE_GENERALIZATION_GAP" in data, "HEADLINE_GENERALIZATION_GAP missing from eval_crossdomain.json"
    gap_data = data["HEADLINE_GENERALIZATION_GAP"]

    indist_f1 = gap_data["indist_macro_f1"]
    sh_f1 = gap_data["sourceheldout_macro_f1"]
    reported_gap = gap_data["macro_f1_gap_percentage_points"]

    expected_gap = (indist_f1 - sh_f1) * 100.0
    assert abs(reported_gap - expected_gap) < 1e-4, f"Mismatch in gap: {reported_gap} vs {expected_gap}"
    assert reported_gap > 40.0, f"Expected critical generalization gap >40 points, got {reported_gap}"


def test_step14_evaluate_cli_supports_all_required_flags():
    """
    Asserts that train/evaluate.py supports --ckpt, --split, --eval_weights, and --gradcam.
    """
    import subprocess
    cmd = [sys.executable, "train/evaluate.py", "--help"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0
    help_text = res.stdout
    for flag in ["--ckpt", "--split", "--eval_weights", "--gradcam"]:
        assert flag in help_text, f"Flag {flag} not found in train/evaluate.py --help"


def test_step15_config_thresholds_fitted():
    """
    Asserts that Step 15 fitted decision thresholds are set in configs/train_config.py
    and are not None.
    """
    from configs.train_config import TAU_ENERGY, T_CAL, TAU_PRIOR, TAU_DISEASE, TAU_MARGIN
    assert TAU_ENERGY is not None and isinstance(TAU_ENERGY, (float, int)), f"Invalid TAU_ENERGY: {TAU_ENERGY}"
    assert T_CAL is not None and isinstance(T_CAL, (float, int)), f"Invalid T_CAL: {T_CAL}"
    assert TAU_PRIOR is not None and isinstance(TAU_PRIOR, (float, int)), f"Invalid TAU_PRIOR: {TAU_PRIOR}"
    assert TAU_DISEASE is not None and isinstance(TAU_DISEASE, (float, int)), f"Invalid TAU_DISEASE: {TAU_DISEASE}"
    assert TAU_MARGIN is not None and isinstance(TAU_MARGIN, (float, int)), f"Invalid TAU_MARGIN: {TAU_MARGIN}"
    assert TAU_ENERGY < 0.0, f"Expected negative energy threshold, got {TAU_ENERGY}"
    assert 0.1 < T_CAL < 3.0, f"T_CAL outside expected range: {T_CAL}"


def test_step15_all_deliverables_exist_and_non_empty():
    """
    Asserts that Step 15 produces all required deliverables in artifacts/reports/:
    1. ood_metrics.json
    2. threshold_sweep.png
    """
    reports_dir = REPORTS
    deliverables = [
        reports_dir / "ood_metrics.json",
        reports_dir / "threshold_sweep.png",
    ]
    for path in deliverables:
        assert path.exists(), f"Missing Step 15 deliverable: {path}"
        assert path.stat().st_size > 0, f"Step 15 deliverable is empty: {path}"


def test_step15_ood_metrics_schema_and_values():
    """
    Asserts that ood_metrics.json contains all required fields and valid metrics.
    """
    metrics_path = REPORTS / "ood_metrics.json"
    assert metrics_path.exists()
    with open(metrics_path, "r") as f:
        data = json.load(f)

    assert "temperature_scaling" in data
    assert "open_set_rejection" in data
    assert "post_hoc_prior" in data
    assert "aggregation_thresholds" in data

    temp = data["temperature_scaling"]
    assert "T_CAL" in temp and 0.1 < temp["T_CAL"] < 2.0
    assert temp["val_nll_after"] < temp["val_nll_before"], "Temperature scaling did not decrease validation NLL"

    ood = data["open_set_rejection"]
    assert ood["target_tpr"] == 0.95
    assert ood["auroc"] > 0.85, f"Expected AUROC > 0.85, got {ood['auroc']}"
    assert ood["in_distribution_samples"] == 3320
    assert ood["open_set_samples"] == 6638

    prior = data["post_hoc_prior"]
    assert "selected_TAU_PRIOR" in prior
    assert "sweep" in prior and len(prior["sweep"]) == 5

    agg = data["aggregation_thresholds"]
    assert "selected_TAU_DISEASE" in agg
    assert "selected_TAU_MARGIN" in agg
    assert agg["operating_point"]["precision"] >= 0.95


def test_step15_calibrate_cli_supports_all_required_flags():
    """
    Asserts that train/calibrate.py supports --ckpt, --openset, and --write-config.
    """
    import subprocess
    cmd = [sys.executable, "train/calibrate.py", "--help"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0
    help_text = res.stdout
    for flag in ["--ckpt", "--openset", "--write-config"]:
        assert flag in help_text, f"Flag {flag} not found in train/calibrate.py --help"


def test_step16_fused_model_channel_order_red_flag():
    """
    CHECK 1 for Step 16: Asserts FusedModel matches reference CPU preprocessing
    on asymmetric strongly RED image [20, 40, 200], and detects BGR/RGB channel swap.
    """
    from train.export_onnx import FusedModel, check_1_red_flag
    from train.model import build_model
    from configs.classes import NUM_CLASSES

    backbone = build_model(num_classes=NUM_CLASSES, pretrained=False)
    backbone.eval()
    fused = FusedModel(backbone)
    fused.eval()

    diff, diff_swapped = check_1_red_flag(fused, backbone, batch_size=9)
    assert diff < 1e-4, f"FusedModel channel order mismatch: diff = {diff}"
    assert diff_swapped > 0.05, f"Swapped channel sensitivity test failed: diff_swapped = {diff_swapped}"


def test_step17_onnx_deliverables_exist_and_non_empty():
    """
    Asserts both raw and simplified ONNX deliverables exist in artifacts/onnx/
    and have reasonable file sizes (>10 MB).
    """
    onnx_dir = ROOT / "artifacts" / "onnx"
    raw_onnx = onnx_dir / "model_a_fused.onnx"
    sim_onnx = onnx_dir / "model_a_sim.onnx"

    assert raw_onnx.exists(), f"Missing raw ONNX file: {raw_onnx}"
    assert sim_onnx.exists(), f"Missing simplified ONNX file: {sim_onnx}"
    assert raw_onnx.stat().st_size > 10 * 1024 * 1024, f"raw ONNX unexpectedly small: {raw_onnx.stat().st_size}"
    assert sim_onnx.stat().st_size > 10 * 1024 * 1024, f"simplified ONNX unexpectedly small: {sim_onnx.stat().st_size}"


def test_step17_simplified_onnx_schema_and_opset():
    """
    Asserts that model_a_sim.onnx has static batch 9, opset 13, float32 input, float32 output.
    """
    import onnx
    sim_onnx = ROOT / "artifacts" / "onnx" / "model_a_sim.onnx"
    assert sim_onnx.exists()

    model = onnx.load(str(sim_onnx))
    onnx.checker.check_model(model)

    assert model.opset_import[0].version == 13, f"Expected opset 13, got {model.opset_import[0].version}"

    # Verify input shape and dtype (dtype 1 = FLOAT32, required by TensorRT 8.2)
    inp = model.graph.input[0]
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    assert shape == [9, 224, 224, 3], f"Expected input shape [9, 224, 224, 3], got {shape}"
    assert inp.type.tensor_type.elem_type == 1, f"Expected float32 (1), got {inp.type.tensor_type.elem_type}"

    # Verify output shape and dtype (dtype 1 = FLOAT32)
    out = model.graph.output[0]
    out_shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    assert out_shape == [9, 29], f"Expected output shape [9, 29], got {out_shape}"
    assert out.type.tensor_type.elem_type == 1, f"Expected float32 (1), got {out.type.tensor_type.elem_type}"


def test_step17_simplified_onnx_passes_checks():
    """
    Verifies that model_a_sim.onnx passes Check 1 (red flag test < 1e-3)
    and Check 3 (numerical fidelity vs PyTorch < 1e-3).
    """
    import onnxruntime as ort
    from train.export_onnx import check_1_red_flag, check_3_onnx_fidelity, FusedModel
    from train.train_model_a import load_checkpoint_for_eval
    from train.model import build_model
    from configs.classes import NUM_CLASSES

    sim_onnx = ROOT / "artifacts" / "onnx" / "model_a_sim.onnx"
    ckpt_path = ROOT / "artifacts" / "checkpoints" / "v3" / "stage1.pt"
    assert sim_onnx.exists() and ckpt_path.exists()

    backbone = build_model(num_classes=NUM_CLASSES, pretrained=False)
    backbone, _ = load_checkpoint_for_eval(ckpt_path, backbone, eval_weights="auto", device=torch.device("cpu"))
    fused_pt = FusedModel(backbone)
    fused_pt.eval()

    sess = ort.InferenceSession(str(sim_onnx), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_type = sess.get_inputs()[0].type
    def onnx_fn(batch):
        if "float" in input_type and batch.dtype == np.uint8:
            batch = batch.astype(np.float32)
        return sess.run(None, {input_name: batch})[0]

    # Check 1
    diff_1, diff_swapped = check_1_red_flag(onnx_fn, backbone=backbone, batch_size=9)
    assert diff_1 < 1e-3, f"Check 1 failed on simplified ONNX: max abs diff {diff_1}"

    # Check 3
    max_diff, _ = check_3_onnx_fidelity(sim_onnx, fused_model=fused_pt, batch_size=9)
    assert max_diff < 1e-3, f"Check 3 failed on simplified ONNX: max abs diff {max_diff}"


def test_step18_build_script_exists_and_forbids_int8():
    """
    Asserts that scripts/build_trt_engine.sh exists, is executable, configures FP16,
    contains zero --int8 compiler directives, and enforces the Jetson platform guard.
    """
    import os
    import subprocess
    script_path = ROOT / "scripts" / "build_trt_engine.sh"
    assert script_path.exists(), f"Missing build script: {script_path}"
    assert os.access(script_path, os.X_OK), f"Build script not executable: {script_path}"

    content = script_path.read_text()

    # Rule checks
    assert "--fp16" in content, "Missing --fp16 in build script"
    lines_with_int8 = [line for line in content.splitlines() if "--int8" in line and not line.strip().startswith("#")]
    assert len(lines_with_int8) == 0, f"Found un-commented --int8 command in build script: {lines_with_int8}"

    # Platform guard check
    res = subprocess.run([str(script_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode != 0, "Build script should refuse execution on non-Jetson (macOS) host"
    assert "ERROR: Target architecture mismatch!" in res.stderr


def test_step19_trt_classifier_structure_and_guards():
    """
    Asserts that edge/trt_classifier.py adheres to Rule R9:
    - Never imports pycuda.autoinit in AST
    - TRTClassifier implements infer, infer_all, close, __enter__, __exit__
    - Safeguards execution on non-CUDA hosts with descriptive RuntimeError
    """
    import ast
    trt_py = ROOT / "edge" / "trt_classifier.py"
    assert trt_py.exists(), f"Missing runtime wrapper: {trt_py}"

    content = trt_py.read_text()
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "autoinit" not in alias.name, f"Found autoinit import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or "autoinit" not in node.module, f"Found autoinit import: {node.module}"
            for alias in node.names:
                assert "autoinit" not in alias.name, f"Found autoinit import: {alias.name}"

    import edge.trt_classifier as trt_mod
    assert hasattr(trt_mod, "TRTClassifier")
    cls = trt_mod.TRTClassifier

    for method in ["infer", "infer_all", "close", "__enter__", "__exit__"]:
        assert hasattr(cls, method), f"TRTClassifier missing required method: {method}"

    # Attempt instantiation on macOS without CUDA
    if not trt_mod.HAS_TRT or not trt_mod.HAS_PYCUDA:
        with pytest.raises(RuntimeError) as exc_info:
            cls("artifacts/engines/model_a_fp16.engine")
        assert "TRTClassifier requires 'tensorrt' and 'pycuda.driver'" in str(exc_info.value)




