"""
configs/reliability.py

Cross-source generalization reliability tiers for Model A classes
based on held-out camera acquisition benchmarks (Wageningen 4TU / multi-source).

Tier definitions (Exact Spec Rule):
- TESTED_ROBUST: recall >= 0.60 on held-out source
- TESTED_WEAK:   0.30 <= recall < 0.60 on held-out source
- TESTED_FAILED: recall < 0.30 on held-out source
- UNTESTED:      no source-heldout data (support == 0)
"""

MODEL_A_CROSS_SOURCE_RELIABILITY = {
    "rice__normal": "TESTED_WEAK",
    "rice__bacterial_leaf_blight": "TESTED_FAILED",
    "rice__bacterial_leaf_streak": "UNTESTED",
    "rice__bacterial_panicle_blight": "UNTESTED",
    "rice__blast": "TESTED_FAILED",
    "rice__brown_spot": "TESTED_FAILED",
    "rice__downy_mildew": "UNTESTED",
    "rice__tungro": "TESTED_FAILED",
    "rice__hispa": "UNTESTED",
    "rice__leaf_roller": "UNTESTED",
    "rice__yellow_stem_borer": "UNTESTED",
    "sugarcane__healthy": "TESTED_ROBUST",
    "sugarcane__dried_leaf": "UNTESTED",
    "sugarcane__mosaic": "UNTESTED",
    "sugarcane__red_rot": "UNTESTED",
    "sugarcane__rust": "UNTESTED",
    "sugarcane__yellow_leaf": "UNTESTED",
    "sugarcane__smut": "UNTESTED",
    "sugarcane__pokkah_boeng": "UNTESTED",
    "sugarcane__grassy_shoot": "UNTESTED",
    "sugarcane__brown_spot": "UNTESTED",
    "sugarcane__banded_chlorosis": "UNTESTED",
    "sugarcane__sett_rot": "UNTESTED",
    "wheat__healthy": "UNTESTED",
    "wheat__yellow_rust": "TESTED_ROBUST",
    "wheat__brown_rust": "UNTESTED",
    "wheat__septoria": "TESTED_FAILED",
    "wheat__powdery_mildew": "TESTED_WEAK",
    "not_crop": "UNTESTED",
}


def get_cross_source_reliability(class_name: str) -> str:
    """Returns cross-source generalization reliability tier for Model A classes."""
    return MODEL_A_CROSS_SOURCE_RELIABILITY.get(class_name, "UNTESTED")
