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

for p in (RAW, INTERIM, PROCESSED, SPLITS, CKPT, ONNX_DIR, ENGINE_DIR, REPORTS,
          PADDY, SUGAR_THITE, SUGAR_DAPHAL, PLANTDOC, PLANTWILD, NOTCROP, OPENSET):
    p.mkdir(parents=True, exist_ok=True)
