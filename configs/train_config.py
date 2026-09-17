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
EPOCHS        = 50
LR_HEAD       = 1e-3
LR_BACKBONE   = 1e-4
WEIGHT_DECAY  = 0.05
WARMUP_EPOCHS = 3
LABEL_SMOOTH  = 0.1
DROP_RATE     = 0.2
DROP_PATH     = 0.1
EMA_DECAY     = 0.9995  # Calibrated for 20,750 steps (50 epochs * 415 steps/epoch): 0.9995^20750 = 3.11e-5 < 0.1% (ceiling 0.999667)
GRAD_CLIP     = 1.0
SEED          = 42

# ---- decision thresholds. Fitted in STEP 15, written back here. ----
TAU_DISEASE = 0.4       # per-tile disease probability floor (calibrated STEP 15)
TAU_MARGIN  = 0.15      # disease must beat healthy on the SAME tile by this (calibrated STEP 15)
TAU_CONF    = 0.60      # abstain below this (on UNADJUSTED probabilities)
TAU_ENERGY  = -2.7957   # <- refitted on SSIM-confirmed clean val set (N=2,616); original STEP 15: -2.8529, pHash-clean: -2.7424
T_CAL       = 0.6162    # <- refitted on SSIM-confirmed clean val set (N=2,616); original STEP 15: 0.597, pHash-clean: 0.610
TAU_PRIOR   = 0.0       # <- swept in STEP 15 (best validation macro-F1 with weighted loss)
CELL_K, CELL_N, CELL_MIN_SCORE = 2, 3, 0.55

# ---- Model B (Sticky-Trap Pest Patch Classifier) Calibration Constants ----
MODEL_B_T_CAL       = 0.9541    # Temperature scaling fitted on validation set (LBFGS on NLL)
MODEL_B_TAU_ENERGY  = -3.8054   # Numerically stable energy threshold on val set (95% ID TPR)
MODEL_B_TAU_CONF    = 0.60      # Confidence floor for patch classification abstention

# ---- provisional spatial & temporal aggregation thresholds (uncalibrated engineering defaults) ----
PROVISIONAL_TAU_HEALTHY          = 0.50    # Mean probability floor for healthy frame classification
PROVISIONAL_NOTCROP_FRAC         = 0.50    # Non-crop probability fraction threshold to declare frame NOT_CROP
PROVISIONAL_MIN_TILES            = 2       # Minimum valid tiles required for frame aggregation
PROVISIONAL_CELL_MIN_FRAMES      = 2       # Minimum valid frames in temporal window before reporting cell verdict

# ---- Edge Frame Gate Thresholds (Step 20) ----
GATE_ALTITUDE_MIN_M            = 1.5     # Rangefinder inspection band lower bound (meters)
GATE_ALTITUDE_MAX_M            = 2.5     # Rangefinder inspection band upper bound (meters)
PROVISIONAL_GATE_MAX_ROLL_DEG  = 15.0    # IMU attitude smear limit (|roll| <= 15 deg, uncalibrated engineering default)
PROVISIONAL_GATE_MAX_PITCH_DEG = 15.0    # IMU attitude smear limit (|pitch| <= 15 deg, uncalibrated engineering default)
PROVISIONAL_TAU_BLUR           = 100.0   # Variance-of-Laplacian sharpness floor (uncalibrated engineering default)
PROVISIONAL_GATE_CLIPPING_DARK_DN     = 5       # Underexposed digital number floor (uncalibrated engineering default)
PROVISIONAL_GATE_CLIPPING_BRIGHT_DN   = 250     # Overexposed digital number ceiling (aligns with edge/camera.py, uncalibrated default)
PROVISIONAL_GATE_CLIPPING_MAX_FRACTION = 0.02   # Maximum tolerated clipped pixel fraction (2%, uncalibrated engineering default)
GATE_MIN_NOVELTY_DISPLACEMENT          = 0.60    # Minimum scene displacement vs last kept frame (60%)

# ---- Edge Tiler Thresholds (Step 20) ----
TILE_MIN_VEG_FRACTION          = 0.40    # Minimum vegetation fraction to keep a tile (>40%)

# ---- Vegetation Indices & Canopy Gating Thresholds (Step 21 & Step 26) ----
PROVISIONAL_EXG_VEG_THRESHOLD      = 20      # ExG threshold for vegetation segmentation (uncalibrated heuristic)
PROVISIONAL_MIN_CANOPY_FRACTION    = 0.15    # Minimum canopy fraction floor before withholding indices
PROVISIONAL_DGCI_MAX_OOD_FRACTION  = 0.30    # Maximum out-of-domain fraction before withholding DGCI mean

# ---- Growth Stage Phenology Defaults (Step 28 / FAO-56 Chapter 5 & 6) ----
# Default variety cycle lengths in days (PROVISIONAL / RECALLED_UNVERIFIED cultivar defaults).
# FAO-56 Table 11 benchmark values for Tropical/Subtropical India.
PROVISIONAL_DEFAULT_CYCLE_DAYS = {
    "rice": 150,       # FAO-56 Table 11: Rice (Tropics) 30/30/60/30
    "wheat": 120,      # FAO-56 Table 11: Wheat (Central India) 15/25/50/30
    "sugarcane": 280,  # FAO-56 Table 11: Sugarcane (Ratoon) 25/70/135/50
}

