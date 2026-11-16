"""
Evaluator v8 — with IoU, MCC metrics + bold summary table
  1. Single-pass evaluation
  2. Per-model threshold calibration
  3. Extra metrics: IoU, MCC, Specificity
  4. Instance norm already applied in data_loader
"""

import gc, time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score,
                             accuracy_score, cohen_kappa_score,
                             confusion_matrix, matthews_corrcoef)

from config import TEST_IMAGES, VAL_IMAGES, EVAL_BATCH_SIZE, OUTPUT_ROOT, THRESHOLD_SEARCH
from src.data_loader import load_preprocessed


# ──────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob) -> Dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    # Core metrics
    oa   = float(accuracy_score(y_true, y_pred))
    f1   = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    kap  = float(cohen_kappa_score(y_true, y_pred))

    # Threshold-independent
    auc_roc = float(roc_auc_score(y_true, y_prob))
    auc_pr  = float(average_precision_score(y_true, y_prob))

    # Extra metrics for HSI literature comparison
    # IoU (Jaccard) — used in HMOSDN, HyperAUDI
    iou = float(tp / (tp + fp + fn + 1e-8))

    # MCC — Matthews Correlation Coefficient (robust to imbalance)
    mcc = float(matthews_corrcoef(y_true, y_pred))

    # Specificity (True Negative Rate) — water correctly classified
    spec = float(tn / (tn + fp + 1e-8))

    # False Alarm Rate
    far = float(fp / (fp + tn + 1e-8))

    return {
        "OA":          oa,
        "F1_oil":      f1,
        "IoU":         iou,
        "Precision":   prec,
        "Recall":      rec,
        "Specificity": spec,
        "FAR":         far,
        "AUC_ROC":     auc_roc,
        "AUC_PR":      auc_pr,
        "Kappa":       kap,
        "MCC":         mcc,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


# ──────────────────────────────────────────────────────────────
def find_optimal_threshold(
    predict_fn:  Callable,
    val_pixels:  np.ndarray,
    val_labels:  np.ndarray,
    n_search:    int = THRESHOLD_SEARCH,
) -> float:
    probs = []
    for i in range(0, len(val_pixels), EVAL_BATCH_SIZE):
        probs.append(predict_fn(val_pixels[i:i+EVAL_BATCH_SIZE]))
    probs = np.concatenate(probs)

    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.02, 0.98, n_search):
        preds = (probs >= thr).astype(int)
        f1    = f1_score(val_labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


# ──────────────────────────────────────────────────────────────
# def calibrate_all_thresholds(
#     predict_fns: Dict[str, Callable],
#     good_bands:  np.ndarray,
#     mean:        np.ndarray,
#     std:         np.ndarray,
#     logger=None,
# ) -> Dict[str, float]:
#     if logger: logger.info("  Calibrating thresholds on val set ...")

#     val_px, val_lb = [], []
#     for vname in VAL_IMAGES:
#         px, lb, H, W = load_preprocessed(vname, good_bands, mean, std)
#         rng = np.random.default_rng(42)
#         idx = rng.choice(len(px), min(20000, len(px)), replace=False)
#         val_px.append(px[idx]); val_lb.append(lb[idx])
#         del px, lb; gc.collect()

#     val_X = np.vstack(val_px)
#     val_y = np.concatenate(val_lb)

#     thresholds = {}
#     for mname, pred_fn in predict_fns.items():
#         thr = find_optimal_threshold(pred_fn, val_X, val_y)
#         thresholds[mname] = thr
#         if logger:
#             logger.info(f"    {mname:18s}: optimal threshold = {thr:.3f}")

#     del val_X, val_y; gc.collect()
#     return thresholds

def calibrate_all_thresholds(
    predict_fns: Dict[str, Callable],
    good_bands:  np.ndarray,
    mean:        np.ndarray,
    std:         np.ndarray,
    threshold_images: list = None,  # ← ADD
    logger=None,
) -> Dict[str, float]:

    if logger: logger.info("  Calibrating thresholds on val set ...")

    # PURANA: VAL_IMAGES
    # NAYA: threshold_images if provided else VAL_IMAGES
    cal_images = threshold_images if threshold_images else VAL_IMAGES

    val_px, val_lb = [], []
    for vname in cal_images:          # ← CHANGE
        px, lb, H, W = load_preprocessed(vname, good_bands, mean, std)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(px), min(20000, len(px)), replace=False)
        val_px.append(px[idx]); val_lb.append(lb[idx])
        del px, lb; gc.collect()

    val_X = np.vstack(val_px)
    val_y = np.concatenate(val_lb)

    thresholds = {}
    for mname, pred_fn in predict_fns.items():
        thr = find_optimal_threshold(pred_fn, val_X, val_y)
        thresholds[mname] = thr
        if logger:
            logger.info(f"    {mname:18s}: optimal threshold = {thr:.3f}")

    del val_X, val_y; gc.collect()
    return thresholds
# ──────────────────────────────────────────────────────────────
def _infer_batched(predict_fn, pixels, batch_size=EVAL_BATCH_SIZE):
    probs = []
    for i in range(0, len(pixels), batch_size):
        probs.append(predict_fn(pixels[i:i+batch_size]))
    return np.concatenate(probs).astype(np.float32)


def evaluate_all_models_on_image(
    predict_fns: Dict[str, Callable],
    img_name:    str,
    good_bands:  np.ndarray,
    mean:        np.ndarray,
    std:         np.ndarray,
    thresholds:  Optional[Dict[str, float]] = None,
    threshold:   float = 0.5,
    logger=None,
) -> Tuple[Dict, Dict, Dict, int, int]:
    t0 = time.time()
    pixels, gt_flat, H, W = load_preprocessed(img_name, good_bands, mean, std)
    if logger:
        logger.info(f"  [{img_name}] {H}x{W}={len(pixels):,}px "
                    f"({time.time()-t0:.1f}s load) — {len(predict_fns)} models")

    metrics_dict, prob_maps, pred_maps = {}, {}, {}
    for mname, pred_fn in predict_fns.items():
        t1   = time.time()
        prob = _infer_batched(pred_fn, pixels)
        thr  = thresholds.get(mname, threshold) if thresholds else threshold
        pred = (prob >= thr).astype(np.uint8)

        metrics_dict[mname] = compute_metrics(gt_flat, pred, prob)
        prob_maps[mname]    = prob
        pred_maps[mname]    = pred

        if logger:
            m = metrics_dict[mname]
            logger.info(f"    {mname:18s} F1={m['F1_oil']:.4f}  "
                        f"AUC-PR={m['AUC_PR']:.4f}  "
                        f"IoU={m['IoU']:.4f}  "
                        f"Kappa={m['Kappa']:.4f}  "
                        f"thr={thr:.3f}  ({time.time()-t1:.1f}s)")

    del pixels; gc.collect()
    return metrics_dict, prob_maps, pred_maps, H, W


# ──────────────────────────────────────────────────────────────
def evaluate_all_models_all_images(
    predict_fns: Dict[str, Callable],
    good_bands:  np.ndarray,
    mean:        np.ndarray,
    std:         np.ndarray,
    thresholds:  Optional[Dict[str, float]] = None,
    image_list:  Optional[List[str]] = None,
    logger=None,
) -> Tuple[pd.DataFrame, List[Dict], Dict, Dict]:
    if image_list is None:
        image_list = TEST_IMAGES

    model_names = list(predict_fns.keys())
    all_rows, prob_cache, pred_cache = [], {}, {}

    for img_name in image_list:
        if logger: logger.info("")
        m_dict, prob_maps, pred_maps, H, W = evaluate_all_models_on_image(
            predict_fns, img_name, good_bands, mean, std,
            thresholds=thresholds, logger=logger)

        prob_cache[img_name] = prob_maps
        pred_cache[img_name] = pred_maps

        for mname in model_names:
            row = dict(m_dict[mname])
            row.update({"image": img_name, "model": mname, "H": H, "W": W})
            if thresholds:
                row["threshold"] = thresholds.get(mname, 0.5)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    metric_cols = ["OA","F1_oil","IoU","Precision","Recall",
                   "Specificity","AUC_ROC","AUC_PR","Kappa","MCC"]
    agg_list = []
    for mname in model_names:
        sub = df[df["model"] == mname]
        agg = {"model": mname}
        for col in metric_cols:
            if col in sub.columns:
                agg[f"{col}_mean"] = float(sub[col].mean())
                agg[f"{col}_std"]  = float(sub[col].std())
        agg_list.append(agg)
        if logger:
            logger.info(f"  SUMMARY {mname}: "
                        f"F1={agg['F1_oil_mean']:.4f}+-{agg['F1_oil_std']:.4f}  "
                        f"IoU={agg['IoU_mean']:.4f}  "
                        f"AUC-PR={agg['AUC_PR_mean']:.4f}  "
                        f"Kappa={agg['Kappa_mean']:.4f}  "
                        f"MCC={agg['MCC_mean']:.4f}")

    return df, agg_list, prob_cache, pred_cache


# ──────────────────────────────────────────────────────────────
def save_results(per_image_df, agg_list, seed):
    out = OUTPUT_ROOT / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    per_image_df.to_csv(out / f"per_image_seed{seed}.csv", index=False)
    pd.DataFrame(agg_list).to_csv(out / f"aggregate_seed{seed}.csv", index=False)


def compile_multiseed_results(seeds):
    frames = []
    for s in seeds:
        p = OUTPUT_ROOT / "metrics" / f"aggregate_seed{s}.csv"
        if p.exists():
            df = pd.read_csv(p); df["seed"] = s; frames.append(df)
    if not frames: return pd.DataFrame()
    combined    = pd.concat(frames)
    metric_cols = [c for c in combined.columns if "_mean" in c or "_std" in c]
    agg = combined.groupby("model")[metric_cols].agg(["mean","std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()
