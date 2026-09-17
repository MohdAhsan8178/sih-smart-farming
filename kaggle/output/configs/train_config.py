# ---- SHARED between training and edge. Changing these breaks the engine. ----
IMAGE_SIZE   = 224      # model input
TILE_SIZE    = 320      # tile cut from the frame before resize to IMAGE_SIZE
TILE_GRID    = 3        # 3x3
N_TILES      = TILE_GRID * TILE_GRID          # 9  <- MUST equal engine batch
TILE_OVERLAP = 0.20
ENGINE_BATCH = N_TILES                        # 9. Never hardcode 8 anywhere.
ONNX_OPSET   = 13                             # TensorRT 8.2 ceiling

# ---- training only ----
BACKBONE      = 'tf_efficientnet_lite0'
BATCH_SIZE    = 64
EPOCHS        = 30
LR_HEAD       = 1e-3
LR_BACKBONE   = 1e-4
WEIGHT_DECAY  = 0.05
WARMUP_EPOCHS = 3
LABEL_SMOOTH  = 0.1
DROP_RATE     = 0.2
DROP_PATH     = 0.1
EMA_DECAY     = 0.9998
GRAD_CLIP     = 1.0
SEED          = 42

# ---- decision thresholds. Fitted in STEP 15, written back here. ----
TAU_DISEASE = 0.55      # per-tile disease probability floor
TAU_MARGIN  = 0.10      # disease must beat healthy on the SAME tile by this
TAU_CONF    = 0.60      # abstain below this (on UNADJUSTED probabilities)
TAU_ENERGY  = None      # <- fit in STEP 15 on the OPEN-SET data, not not_crop
T_CAL       = None      # <- fit in STEP 15 (temperature scaling)
TAU_PRIOR   = 1.0       # post-hoc logit adjustment strength
CELL_K, CELL_N, CELL_MIN_SCORE = 2, 3, 0.55
