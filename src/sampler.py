"""
Multi-image balanced pixel sampler.
Uses preprocess_image which now includes L1 normalisation.
"""

import gc
import numpy as np
from typing import List, Optional, Tuple

from config import N_PER_CLASS_PER_IMAGE, TRAIN_IMAGES, VAL_IMAGES
from src.data_loader import load_mat, preprocess_image


def sample_from_image(
    img_name:    str,
    good_bands:  np.ndarray,
    mean:        np.ndarray,
    std:         np.ndarray,
    n_per_class: int = N_PER_CLASS_PER_IMAGE,
    seed:        int = 42,
    logger=None,
) -> Tuple[np.ndarray, np.ndarray]:

    from config import USE_ALL_OIL   # ← ADD

    rng = np.random.default_rng(seed)
    img, gt = load_mat(img_name)
    proc    = preprocess_image(img, good_bands, mean, std)
    del img; gc.collect()

    H, W, B = proc.shape
    pixels  = proc.reshape(-1, B)
    labels  = gt.flatten()

    oil_idx   = np.where(labels == 1)[0]
    water_idx = np.where(labels == 0)[0]

    # ── NEW: Use ALL oil pixels ─────────────────────────────
    if USE_ALL_OIL:
        n_oil   = len(oil_idx)           # SARE oil pixels
        n_water = min(n_oil, len(water_idx))  # equal water
    else:
        n_oil   = min(n_per_class, len(oil_idx))
        n_water = min(n_per_class, len(water_idx))
    # ────────────────────────────────────────────────────────

    sel = np.concatenate([
        oil_idx,                                          # ALL oil (no sampling)
        rng.choice(water_idx, n_water, replace=False),   # random water
    ])
    rng.shuffle(sel)

    X = pixels[sel].copy()
    y = labels[sel].copy()

    if logger:
        logger.info(f"  {img_name}: {n_oil:,} oil + {n_water:,} water "
                    f"({'ALL oil' if USE_ALL_OIL else 'sampled'})")
    del proc, pixels; gc.collect()
    return X, y


def build_train_set(
    good_bands:  np.ndarray,
    mean:        np.ndarray,
    std:         np.ndarray,
    n_per_class: int = N_PER_CLASS_PER_IMAGE,
    seed:        int = 42,
    image_list:  Optional[List[str]] = None,
    logger=None,
) -> Tuple[np.ndarray, np.ndarray]:
    if image_list is None:
        image_list = TRAIN_IMAGES
    X_list, y_list = [], []
    for i, name in enumerate(image_list):
        X_i, y_i = sample_from_image(name, good_bands, mean, std,
                                      n_per_class, seed + i, logger)
        X_list.append(X_i); y_list.append(y_i)

    X = np.vstack(X_list); y = np.concatenate(y_list)
    if logger:
        logger.info(f"Train set: {X.shape[0]:,} samples, {X.shape[1]} bands  "
                    f"({(y==1).sum():,} oil, {(y==0).sum():,} water)")
    return X, y


def build_val_set(
    good_bands:  np.ndarray,
    mean:        np.ndarray,
    std:         np.ndarray,
    n_per_class: int = N_PER_CLASS_PER_IMAGE * 2,
    seed:        int = 42,
    logger=None,
) -> Tuple[np.ndarray, np.ndarray]:
    return build_train_set(good_bands, mean, std, n_per_class,
                           seed + 9999, VAL_IMAGES, logger)
