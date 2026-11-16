"""
Data loader — Mode-based preprocessing
=======================================
Modes (set via config flags):
  baseline : raw reflectance only
  onlyL1   : L1 norm only
  onlyIN   : Instance norm only
  full     : L1 + Instance norm
"""

import gc
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Tuple
from config import DATA_ROOT, IMAGE_FILES, CLIP_SIGMA
import config as _cfg


def load_mat(name: str) -> Tuple[np.ndarray, np.ndarray]:
    path = DATA_ROOT / IMAGE_FILES[name]
    mat  = sio.loadmat(str(path))
    img  = mat["img"].astype(np.float32)
    gt   = mat["map"].astype(np.uint8)
    return img, gt


def _l1_normalise(flat: np.ndarray) -> np.ndarray:
    """L1 norm — preserves spectral shape, resolution-invariant."""
    l1 = np.abs(flat).sum(axis=1, keepdims=True) + 1e-8
    return flat / l1


def _instance_normalise(flat: np.ndarray) -> np.ndarray:
    """Per-image instance norm — distribution alignment."""
    mu  = flat.mean(axis=0, keepdims=True)
    sig = flat.std(axis=0, keepdims=True) + 1e-8
    return (flat - mu) / sig


def compute_train_norm_stats(good_bands, logger=None):
    if logger:
        logger.info("  Norm stats: not needed (instance norm pipeline).")
    n = len(good_bands)
    return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)


def preprocess_image(img, good_bands, mean, std):
    """Apply mode-specific preprocessing."""
    H, W, _ = img.shape
    flat = img[:, :, good_bands].astype(np.float32).reshape(-1, len(good_bands))

    if _cfg.L1_NORMALISE:
        flat = _l1_normalise(flat)

    if _cfg.INSTANCE_NORM:
        flat = _instance_normalise(flat)

    flat = np.clip(flat, -CLIP_SIGMA, CLIP_SIGMA)
    return flat.reshape(H, W, -1).astype(np.float32)


def load_preprocessed(name, good_bands, mean, std):
    img, gt = load_mat(name)
    proc    = preprocess_image(img, good_bands, mean, std)
    del img; gc.collect()
    H, W, B = proc.shape
    return proc.reshape(-1, B), gt.flatten(), H, W
