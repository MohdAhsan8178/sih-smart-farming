"""
Model definition for aerial crop disease classification.
Reference: ULTIMATE_IMPLEMENTATION_PLAN_1.md STEP 12.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
import timm

from configs.train_config import (
    BACKBONE,
    DROP_PATH,
    DROP_RATE,
    LR_BACKBONE,
    LR_HEAD,
    WEIGHT_DECAY,
)


def build_model(
    backbone: Optional[str] = None,
    num_classes: Optional[int] = None,
    pretrained: bool = True,
    drop_rate: Optional[float] = None,
    drop_path_rate: Optional[float] = None,
) -> nn.Module:
    """
    Constructs the classifier model using timm.
    Class count is resolved at runtime from configs/classes.py or passed explicitly
    by the caller (e.g. for synthetic manifests or taxonomy expansions).
    """
    bb = backbone if backbone is not None else BACKBONE

    if num_classes is None:
        from configs.classes import CLASS_NAMES
        n_classes = len(CLASS_NAMES)
    else:
        n_classes = int(num_classes)

    dr = float(drop_rate if drop_rate is not None else DROP_RATE)
    dpr = float(drop_path_rate if drop_path_rate is not None else DROP_PATH)

    model = timm.create_model(
        bb,
        pretrained=pretrained,
        num_classes=n_classes,
        drop_rate=dr,
        drop_path_rate=dpr,
    )
    return model


def get_parameter_groups(
    model: nn.Module,
    lr_backbone: Optional[float] = None,
    lr_head: Optional[float] = None,
    weight_decay: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Separates classifier head parameters from backbone parameters for discriminative LR.
    Head receives lr_head (1e-3) while backbone receives lr_backbone (1e-4).
    """
    lr_bb = float(lr_backbone if lr_backbone is not None else LR_BACKBONE)
    lr_hd = float(lr_head if lr_head is not None else LR_HEAD)
    wd = float(weight_decay if weight_decay is not None else WEIGHT_DECAY)

    # In timm efficientnet models, get_classifier() retrieves the linear head
    head = model.get_classifier()
    head_params = list(head.parameters())
    head_ids = {id(p) for p in head_params}

    backbone_params = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]

    return [
        {'params': backbone_params, 'lr': lr_bb, 'weight_decay': wd},
        {'params': head_params, 'lr': lr_hd, 'weight_decay': wd},
    ]
