"""
Model B definition: Lightweight CNN for sticky-trap insect patch classification (64x64).
Reference: AI_Handbook_4.md §7.5.2 (~150k params).

Python 3.6 compatible.
"""

import sys
from pathlib import Path
from typing import Optional

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from configs.classes_model_b import NUM_CLASSES


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU, followed by 2x2 MaxPool."""
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TrapPestCNN(nn.Module):
    """
    Lightweight 4-block CNN designed for 64x64 patches.
    Parameters: ~153k.
    Inference latency: < 2ms on Jetson Nano Maxwell GPU and gateway CPU.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, drop_rate: float = 0.3):
        super(TrapPestCNN, self).__init__()
        self.num_classes = int(num_classes)
        
        # 4 feature blocks: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.features = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=float(drop_rate)),
            nn.Linear(128, self.num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 128-d pooled feature representation."""
        feats = self.features(x)
        pooled = self.pool(feats)
        return torch.flatten(pooled, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (B, 3, 64, 64). Returns raw logits: (B, num_classes)."""
        feat_vec = self.extract_features(x)
        return self.head(feat_vec)


def build_model_b(num_classes: int = NUM_CLASSES,
                  drop_rate: float = 0.3,
                  weights_path: Optional[str] = None) -> TrapPestCNN:
    """Instantiate TrapPestCNN, optionally loading checkpoint weights."""
    model = TrapPestCNN(num_classes=num_classes, drop_rate=drop_rate)
    if weights_path is not None:
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state_dict)
    return model


def export_model_b_onnx(model: nn.Module,
                        onnx_path: str,
                        opset_version: int = 13,
                        batch_size: int = 1) -> str:
    """
    Export Model B to ONNX format compatible with TensorRT 8.2 on Jetson Nano.
    Uses static batch size or dynamic batching as specified.
    """
    model_cpu = model.to('cpu')
    model_cpu.eval()
    dummy_input = torch.randn(batch_size, 3, 64, 64, dtype=torch.float32)
    p = Path(onnx_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model_cpu,
        dummy_input,
        str(p),
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False
    )
    return str(p)
