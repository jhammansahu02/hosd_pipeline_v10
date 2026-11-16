"""
Band selection — L1 normalized, mode-independent
=================================================
Band scoring ALWAYS uses L1 normalization internally
to ensure consistent 203-band selection across all modes.
This is separate from the preprocessing pipeline (L1/IN).

Paper justification:
  "Band selection employs L1-normalized spectra for
   Fisher scoring to ensure resolution-invariant band
   quality assessment, yielding 203 bands consistent
   across all normalization modes."
"""

import gc, json
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from typing import Optional, Tuple

from config import (TOTAL_BANDS, DEAD_BAND_STD_THR, COMPOSITE_SCORE_THR,
                    FISHER_WEIGHT, CONTINUITY_WEIGHT, STD_WEIGHT, KURT_WEIGHT,
                    N_CONTINUITY_NBRS, TRAIN_IMAGES)
from src.data_loader import load_mat


def _get_cache() -> Path:
    import config as _cfg
    return _cfg.OUTPUT_ROOT / "metrics" / "band_scores.json"


def _l1(flat):
    return flat / (np.abs(flat).sum(1, keepdims=True) + 1e-8)


def compute_band_scores(subsample=1500, seed=42, logger=None):
    rng = np.random.default_rng(seed)
    all_pixels, all_labels = [], []

    for name in TRAIN_IMAGES:
        if logger: logger.info(f"  Band scoring: {name} ...")
        img, gt = load_mat(name)
        px = img.reshape(-1, TOTAL_BANDS).astype(np.float64)
        lb = gt.flatten()

        oi = np.where(lb == 1)[0]
        wi = np.where(lb == 0)[0]
        n  = min(subsample // 2, len(oi), len(wi))
        sel = np.concatenate([rng.choice(oi, n, replace=False),
                              rng.choice(wi, n, replace=False)])

        px_sub = px[sel]
        # ALWAYS L1 for band scoring — mode independent
        px_sub = _l1(px_sub)

        all_pixels.append(px_sub)
        all_labels.append(lb[sel])
        del img, px, lb; gc.collect()

    pixels = np.vstack(all_pixels)
    labels = np.concatenate(all_labels)

    if logger:
        logger.info(f"  Pooled {len(pixels):,} pixels "
                    f"({(labels==1).sum():,} oil, "
                    f"{(labels==0).sum():,} water)")

    stds = pixels.std(axis=0)
    dead = stds < DEAD_BAND_STD_THR

    # Fisher discriminant
    mu_o  = pixels[labels==1].mean(0)
    sig_o = pixels[labels==1].std(0) + 1e-8
    mu_w  = pixels[labels==0].mean(0)
    sig_w = pixels[labels==0].std(0) + 1e-8
    fisher   = (mu_o - mu_w)**2 / (sig_o**2 + sig_w**2 + 1e-8)
    fisher_s = np.clip(fisher / 3.0, 0, 1)

    # Continuity
    sub_idx = rng.choice(len(pixels),
                         min(2000, len(pixels)), replace=False)
    sub     = pixels[sub_idx]
    cont_s  = np.zeros(TOTAL_BANDS)
    for b in range(TOTAL_BANDS):
        if stds[b] < 1e-9: continue
        nbrs = []
        for d in range(1, N_CONTINUITY_NBRS + 1):
            for nb in (b - d, b + d):
                if 0 <= nb < TOTAL_BANDS and sub[:, nb].std() > 1e-9:
                    c = np.corrcoef(sub[:, b], sub[:, nb])[0, 1]
                    if np.isfinite(c): nbrs.append(abs(c))
        if nbrs: cont_s[b] = np.mean(nbrs)

    # Kurtosis
    kurts  = np.array([sp_stats.kurtosis(pixels[:, b])
                       if stds[b] > 1e-9 else 0
                       for b in range(TOTAL_BANDS)])
    kurt_s = np.clip(1.0 - np.maximum(0, kurts - 100) / 3000.0, 0, 1)
    std_s  = np.clip(stds / (stds.max() + 1e-8), 0, 1)

    composite = (FISHER_WEIGHT    * fisher_s +
                 CONTINUITY_WEIGHT * cont_s  +
                 STD_WEIGHT        * std_s   +
                 KURT_WEIGHT       * kurt_s)
    composite[dead] = 0.0
    return composite


def select_good_bands(scores=None, threshold=COMPOSITE_SCORE_THR,
                      logger=None, force_recompute=False):
    cache = _get_cache()

    if not force_recompute and cache.exists():
        with open(cache) as f:
            cached = json.load(f)
        scores     = np.array(cached["scores"])
        good_bands = np.array(cached["good_bands"], dtype=int)
        if logger:
            logger.info(f"Band scores from cache: "
                        f"{len(good_bands)} good bands.")
        return good_bands, scores

    if scores is None:
        scores = compute_band_scores(logger=logger)

    good_bands = np.where(scores >= threshold)[0]

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump({
            "scores":     scores.tolist(),
            "good_bands": good_bands.tolist(),
            "threshold":  threshold,
            "n_good":     int(len(good_bands)),
            "note":       "L1 always used for band scoring"
        }, f, indent=2)

    if logger:
        logger.info(
            f"Band selection: {len(good_bands)}/{TOTAL_BANDS} kept, "
            f"{TOTAL_BANDS - len(good_bands)} removed "
            f"(L1 scoring, thr={threshold})")
    return good_bands, scores
