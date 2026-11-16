"""
HOSD Pipeline v10 — Configuration
===================================
4 Modes:
  baseline : No L1, No Instance Norm
  onlyL1   : Only L1 Normalization
  onlyIN   : Only Instance Normalization
  full     : L1 + Instance Norm (best)
"""

import os
from pathlib import Path

# ── Data Path ─────────────────────────────────────────────────
_DATA_ROOT_DEFAULT = r"G:\Jhamman\PHD Work\Oill_Spill_Gulf_of_Maxico\hosd_pipeline_v8"
DATA_ROOT = Path(os.environ.get("HOSD_DATA_ROOT", _DATA_ROOT_DEFAULT))

IMAGE_FILES = {
    "GM01": "GM01.mat", "GM02": "GM02.mat", "GM03": "GM03.mat",
    "GM04": "GM04.mat", "GM05": "GM05.mat", "GM06": "GM06.mat",
    "GM07": "GM07.mat", "GM08": "GM08.mat", "GM09": "GM09.mat",
    "GM10": "GM10.mat", "GM11": "GM11.mat", "GM12": "GM12.mat",
    "GM13": "GM13.mat", "GM14": "GM14.mat", "GM15": "GM15.mat",
    "GM16": "GM16.mat", "GM17": "GM17.mat", "GM18": "GM18.mat",
}

IMAGE_META = {
    "GM01": ("2010-05-17", 7.6), "GM02": ("2010-05-17", 7.6),
    "GM03": ("2010-05-17", 7.6), "GM04": ("2010-05-17", 7.6),
    "GM05": ("2010-05-17", 7.6), "GM06": ("2010-05-18", 8.1),
    "GM07": ("2010-07-09", 3.3), "GM08": ("2010-07-09", 3.3),
    "GM09": ("2010-07-09", 3.2), "GM10": ("2010-05-17", 7.6),
    "GM11": ("2010-05-17", 7.6), "GM12": ("2010-05-06", 7.6),
    "GM13": ("2010-07-09", 3.2), "GM14": ("2010-07-09", 3.2),
    "GM15": ("2010-07-09", 3.3), "GM16": ("2010-07-09", 3.2),
    "GM17": ("2010-05-17", 7.6), "GM18": ("2010-07-09", 3.3),
}

# ── Temporal Split ─────────────────────────────────────────────
TRAIN_IMAGES     = ["GM01","GM02","GM03","GM04","GM05",
                    "GM10","GM11","GM17"]
VAL_IMAGES       = ["GM06","GM12"]    # May → early stopping
THRESHOLD_IMAGES = ["GM07"]           # July → threshold calibration
TEST_IMAGES      = ["GM08","GM09","GM13","GM14",
                    "GM15","GM16","GM18"]

# ── Output (set dynamically by main.py) ───────────────────────
OUTPUT_ROOT = Path("./results/full")
LOG_DIR     = Path("./logs/full")
CKPT_DIR    = OUTPUT_ROOT / "checkpoints"

# ── Preprocessing Flags (set by main.py) ──────────────────────
L1_NORMALISE  = True
INSTANCE_NORM = True
CLIP_SIGMA    = 5.0
TOTAL_BANDS   = 224

# ── Band Selection ─────────────────────────────────────────────
DEAD_BAND_STD_THR   = 1e-6
COMPOSITE_SCORE_THR = 0.20   # ← fixed (was 0.05)
FISHER_WEIGHT       = 0.45
CONTINUITY_WEIGHT   = 0.25
STD_WEIGHT          = 0.20
KURT_WEIGHT         = 0.10
N_CONTINUITY_NBRS   = 3

# ── Sampling ───────────────────────────────────────────────────
USE_ALL_OIL           = True
N_PER_CLASS_PER_IMAGE = 2000
RANDOM_SEEDS          = [42, 123, 2024]
N_PER_CLASS_DL        = 3000

# ── DL Training ───────────────────────────────────────────────
BATCH_SIZE    = 8192
EPOCHS        = 150
LR            = 1e-3
WEIGHT_DECAY  = 5e-4
PATIENCE      = 25
LR_T_MAX      = 150
LABEL_SMOOTH  = 0.05
MIXUP_ALPHA   = 0.1
FOCAL_GAMMA   = 2.0
FOCAL_ALPHA   = 0.75
NUM_WORKERS   = 0
WARMUP_EPOCHS = 5

# ── Classical ─────────────────────────────────────────────────
RF_N_EST     = 300
RF_MAX_DEPTH = None

# ── Evaluation ────────────────────────────────────────────────
EVAL_BATCH_SIZE  = 131072
THRESHOLD_SEARCH = 50
OIL_CLASS        = 1

# ── Visualisation ─────────────────────────────────────────────
DPI       = 150
FIG_WIDTH = 20
