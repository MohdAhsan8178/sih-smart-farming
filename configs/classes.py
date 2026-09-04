"""Single source of truth for the class taxonomy. Import this everywhere."""
import numpy as np

CLASS_NAMES = [
    # rice (Paddy Doctor, 13)
    'rice__normal', 'rice__bacterial_leaf_blight', 'rice__bacterial_leaf_streak',
    'rice__bacterial_panicle_blight', 'rice__blast', 'rice__brown_spot',
    'rice__downy_mildew', 'rice__tungro', 'rice__hispa', 'rice__leaf_roller',
    'rice__black_stem_borer', 'rice__white_stem_borer', 'rice__yellow_stem_borer',
    # sugarcane (12)
    'sugarcane__healthy', 'sugarcane__dried_leaf', 'sugarcane__mosaic',
    'sugarcane__red_rot', 'sugarcane__rust', 'sugarcane__yellow_leaf',
    'sugarcane__smut', 'sugarcane__pokkah_boeng', 'sugarcane__grassy_shoot',
    'sugarcane__brown_spot', 'sugarcane__banded_chlorosis', 'sugarcane__sett_rot',
    # wheat (5, optional)
    'wheat__healthy', 'wheat__yellow_rust', 'wheat__brown_rust',
    'wheat__septoria', 'wheat__powdery_mildew',
    # reject
    'not_crop',
]

NUM_CLASSES = len(CLASS_NAMES)                       # 31
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
