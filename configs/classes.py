"""Single source of truth for the class taxonomy. Import this everywhere."""
import numpy as np

CLASS_NAMES = [
    # rice (11)
    'rice__normal', 'rice__bacterial_leaf_blight', 'rice__bacterial_leaf_streak',
    'rice__bacterial_panicle_blight', 'rice__blast', 'rice__brown_spot',
    'rice__downy_mildew', 'rice__tungro', 'rice__hispa', 'rice__leaf_roller',
    'rice__yellow_stem_borer',
    # sugarcane (12)
    'sugarcane__healthy', 'sugarcane__dried_leaf', 'sugarcane__mosaic',
    'sugarcane__red_rot', 'sugarcane__rust', 'sugarcane__yellow_leaf',
    'sugarcane__smut', 'sugarcane__pokkah_boeng', 'sugarcane__grassy_shoot',
    'sugarcane__brown_spot', 'sugarcane__banded_chlorosis', 'sugarcane__sett_rot',
    # wheat (5)
    'wheat__healthy', 'wheat__yellow_rust', 'wheat__brown_rust',
    'wheat__septoria', 'wheat__powdery_mildew',
    # reject
    'not_crop',
]

NUM_CLASSES = len(CLASS_NAMES)                       # dynamically resolved from len(CLASS_NAMES)
IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

HEALTHY_NAMES = ['rice__normal', 'sugarcane__healthy',
                 'sugarcane__dried_leaf', 'wheat__healthy']
HEALTHY_COLS = np.array([IDX[n] for n in HEALTHY_NAMES], dtype=int)
NOTCROP_COL = IDX['not_crop']

# Crop diagnostic columns = everything EXCEPT not_crop.
# Energy is computed over these only (see rejection.open_set_energy).
CROP_COLS = np.array([i for i in range(NUM_CLASSES) if i != NOTCROP_COL], dtype=int)

DISEASE_COLS = np.array(
    [i for i in range(NUM_CLASSES)
     if i != NOTCROP_COL and i not in set(HEALTHY_COLS.tolist())], dtype=int)

assert len(CROP_COLS) == NUM_CLASSES - 1
assert len(DISEASE_COLS) == NUM_CLASSES - 1 - len(HEALTHY_COLS)

# ---- Cross-Source Reliability Tiers (F7 / Step 14 Held-Out Evaluation) ----
# TESTED_ROBUST: F1 >= 0.85 on held-out source (or recall >= 0.60)
# TESTED_WEAK:   0.50 <= F1 < 0.85 (or 0.30 <= recall < 0.60)
# TESTED_FAILED: F1 < 0.50 (or recall < 0.30)
# UNTESTED:      Class not present in held-out source evaluation (support == 0)
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
    "sugarcane__healthy": "TESTED_WEAK",
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
    "wheat__yellow_rust": "TESTED_WEAK",
    "wheat__brown_rust": "UNTESTED",
    "wheat__septoria": "TESTED_FAILED",
    "wheat__powdery_mildew": "TESTED_FAILED",
    "not_crop": "UNTESTED",
}

def get_cross_source_reliability(class_name: str) -> str:
    """Returns cross-source generalization reliability tier for Model A classes."""
    return MODEL_A_CROSS_SOURCE_RELIABILITY.get(class_name, "UNTESTED")
