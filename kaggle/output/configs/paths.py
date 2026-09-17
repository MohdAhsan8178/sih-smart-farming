from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
RAW, INTERIM, PROCESSED = DATA / 'raw', DATA / 'interim', DATA / 'processed'
SPLITS = ROOT / 'splits'
ARTIFACTS = ROOT / 'artifacts'
CKPT, ONNX_DIR, ENGINE_DIR, REPORTS = (ARTIFACTS / 'checkpoints',
                                       ARTIFACTS / 'onnx',
                                       ARTIFACTS / 'engines',
                                       ARTIFACTS / 'reports')

# Dataset subdirectories under RAW
PADDY = RAW / 'paddy_doctor'
SUGAR_THITE = RAW / 'sugarcane_thite'
SUGAR_DAPHAL = RAW / 'sugarcane_daphal'
PLANTDOC = RAW / 'plantdoc'
PLANTWILD = RAW / 'plantwild'
NOTCROP = RAW / 'not_crop'
OPENSET = RAW / 'openset'
WHEAT_SMALL = RAW / 'wheat_small'
WHEAT_MENDELEY = RAW / 'wheat_mendeley'
RICE_RIFAT = RAW / 'rice_pest_rifat'
RICE_SETHY = RAW / 'rice_sethy'
RICE_HASAN = RAW / 'rice_hasan'

for p in (RAW, INTERIM, PROCESSED, SPLITS, CKPT, ONNX_DIR, ENGINE_DIR, REPORTS,
          PADDY, SUGAR_THITE, SUGAR_DAPHAL, PLANTDOC, PLANTWILD, NOTCROP, OPENSET,
          WHEAT_SMALL, WHEAT_MENDELEY, RICE_RIFAT, RICE_SETHY, RICE_HASAN):
    p.mkdir(parents=True, exist_ok=True)
