"""
Visualization suite — v8
==========================
Changes:
  1. visualize_all_images_gt() — all images with GT before evaluation
  2. comprehensive_prediction_maps() — row layout: RGB|GT|classical|DL|PhysSpec
  3. summary_table_plot() — bold highest value per column
  4. All existing functions updated with new model colours
  5. Extra metrics (IoU, MCC) in summary table
"""

import gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             roc_curve, auc, f1_score, average_precision_score)

from config import (OUTPUT_ROOT, DPI, FIG_WIDTH,
                    TEST_IMAGES, TRAIN_IMAGES, VAL_IMAGES, IMAGE_META)

FIG_DIR = OUTPUT_ROOT / "figures"
MAP_DIR = OUTPUT_ROOT / "maps"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLOURS = {
    "GNB_HOSD":      "#636EFA",
    "SVM_RBF":       "#EF553B",
    "RandomForest":  "#00CC96",
    "LDA":           "#AB63FA",
    "ExtraTrees":    "#19D3F3",
    "SpectralCNN1D": "#FFA15A",
    "HybridSN":      "#FF6692",
    "SpectralFormer":"#B6E880",
    "PhysSpecNet":   "#FF97FF",
    "MSSRAN":        "#FECB52",
}
PRED_CMAP = ListedColormap(["#1a6faf", "#d62728"])


def _save(fig, name: str, subdir: str = "figures") -> None:
    d = OUTPUT_ROOT / subdir
    d.mkdir(exist_ok=True)
    p = d / f"{name}.png"
    fig.savefig(str(p), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {p}")


def _make_rgb(img_raw: np.ndarray) -> np.ndarray:
    """Make pseudo-RGB from bands 29, 19, 9."""
    def _norm(ch):
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        return np.clip((ch.astype(np.float32) - lo) / (hi - lo + 1e-8), 0, 1)
    return np.stack([_norm(img_raw[:, :, 29]),
                     _norm(img_raw[:, :, 19]),
                     _norm(img_raw[:, :, 9])], axis=-1)


# ══════════════════════════════════════════════════════════════
# 1. VISUALIZE ALL IMAGES WITH GROUND TRUTH (before evaluation)
# ══════════════════════════════════════════════════════════════

def visualize_all_images_gt(logger=None) -> None:
    """
    Requirement 1: Visualize ALL 18 images with ground truth
    before any evaluation. Shows RGB + GT side by side.
    One figure per split (train/val/test).
    """
    from src.data_loader import load_mat

    all_splits = [
        ("Train", TRAIN_IMAGES),
        ("Val",   VAL_IMAGES),
        ("Test",  TEST_IMAGES),
    ]

    for split_name, img_list in all_splits:
        n = len(img_list)
        fig, axes = plt.subplots(n, 2, figsize=(10, n * 3.5))
        if n == 1:
            axes = axes.reshape(1, 2)

        for i, name in enumerate(img_list):
            try:
                img, gt = load_mat(name)
                rgb = _make_rgb(img)
                date, res = IMAGE_META.get(name, ("?", "?"))
                oil_pct = 100 * gt.sum() / gt.size

                axes[i, 0].imshow(rgb)
                axes[i, 0].set_title(
                    f"{name} ({date}, {res}m)\n{img.shape[0]}×{img.shape[1]}",
                    fontsize=8, fontweight="bold")
                axes[i, 0].axis("off")

                axes[i, 1].imshow(gt, cmap=PRED_CMAP, vmin=0, vmax=1)
                axes[i, 1].set_title(
                    f"Ground Truth\nOil={oil_pct:.1f}%",
                    fontsize=8, fontweight="bold")
                axes[i, 1].axis("off")

                del img, gt
                gc.collect()
            except Exception as e:
                if logger: logger.warning(f"  {name}: {e}")

        patches = [mpatches.Patch(color="#1a6faf", label="Water"),
                   mpatches.Patch(color="#d62728", label="Oil")]
        fig.legend(handles=patches, loc="lower center",
                   ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(
            f"HOSD {split_name} Images — Pseudo-RGB & Ground Truth",
            fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        _save(fig, f"gt_visualization_{split_name.lower()}")
        if logger:
            logger.info(f"  GT visualization saved: {split_name}")


# ══════════════════════════════════════════════════════════════
# 2. COMPREHENSIVE PREDICTION MAPS (Requirement 6)
# Row: RGB | GT | Classical models | DL models | PhysSpecNet
# 2 images per figure
# ══════════════════════════════════════════════════════════════

def comprehensive_prediction_maps(
    img_names:  List[str],
    prob_cache: Dict[str, Dict[str, np.ndarray]],
    pred_cache: Dict[str, Dict[str, np.ndarray]],
    gt_cache:   Dict[str, np.ndarray],
    H_cache:    Dict[str, int],
    W_cache:    Dict[str, int],
    rgb_cache:  Dict[str, np.ndarray],
    classical_models: List[str],
    dl_models:        List[str],
    logger=None,
) -> None:
    """
    Requirement 6: Row layout per image:
      Col 0: Pseudo-RGB
      Col 1: Ground Truth
      Col 2+: Classical models (GNB, RF, SVM, LDA, ExtraTrees)
      Col N+: DL models (SpectralCNN1D, HybridSN, SpectralFormer)
      Col M+: PhysSpecNet (highlighted)

    2 images per figure.
    """
    all_model_order = (["GT"] + classical_models + dl_models + ["PhysSpecNet"])
    # Remove GT from model lists
    model_order = classical_models + dl_models + ["PhysSpecNet"]

    # Process 2 images per figure
    for fig_idx in range(0, len(img_names), 2):
        pair = img_names[fig_idx: fig_idx + 2]
        n_rows = len(pair)
        n_cols = 2 + len(model_order)  # RGB + GT + models

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 2.8, n_rows * 3.5))
        if n_rows == 1:
            axes = axes.reshape(1, n_cols)

        for row_i, img_name in enumerate(pair):
            if img_name not in pred_cache:
                for ax in axes[row_i]:
                    ax.axis("off")
                continue

            H = H_cache[img_name]
            W = W_cache[img_name]
            gt = gt_cache[img_name].reshape(H, W)

            # Col 0: RGB
            if img_name in rgb_cache:
                axes[row_i, 0].imshow(rgb_cache[img_name])
            else:
                axes[row_i, 0].axis("off")
            date, res = IMAGE_META.get(img_name, ("?", "?"))
            oil_pct = 100 * gt.sum() / gt.size
            axes[row_i, 0].set_title(
                f"{img_name}\n{date},{res}m",
                fontsize=7, fontweight="bold")
            axes[row_i, 0].axis("off")

            # Col 1: Ground Truth
            axes[row_i, 1].imshow(gt, cmap=PRED_CMAP, vmin=0, vmax=1)
            axes[row_i, 1].set_title(
                f"Ground Truth\n{oil_pct:.1f}% oil",
                fontsize=7, fontweight="bold")
            axes[row_i, 1].axis("off")

            # Col 2+: Models
            for col_i, mname in enumerate(model_order, start=2):
                ax = axes[row_i, col_i]
                if mname not in pred_cache[img_name]:
                    ax.axis("off")
                    continue

                pred = pred_cache[img_name][mname].reshape(H, W)
                prob = prob_cache[img_name][mname]
                gt_flat = gt.flatten()

                f1  = f1_score(gt_flat, pred.flatten(),
                               pos_label=1, zero_division=0)
                ap  = average_precision_score(gt_flat, prob)

                ax.imshow(pred, cmap=PRED_CMAP, vmin=0, vmax=1)

                # PhysSpecNet highlighted with box
                is_phys = mname == "PhysSpecNet"
                col = MODEL_COLOURS.get(mname, "black")
                title = f"{mname}\nF1={f1:.2f} PR={ap:.2f}"
                ax.set_title(title, fontsize=6,
                             fontweight="bold", color=col)
                if is_phys:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#FF97FF")
                        spine.set_linewidth(3)
                ax.axis("off")

        # Column labels (top row)
        col_labels = ["Pseudo-RGB", "Ground Truth"] + model_order
        for ci, lbl in enumerate(col_labels):
            if ci < n_cols:
                axes[0, ci].set_xlabel(lbl, fontsize=7)

        patches = [mpatches.Patch(color="#1a6faf", label="Water"),
                   mpatches.Patch(color="#d62728", label="Oil")]
        fig.legend(handles=patches, loc="lower center",
                   ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.0))

        fig.suptitle(
            "Prediction Maps — Classical | DL | PhysSpecNet",
            fontsize=11, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        fname = "_".join([n for n in pair])
        _save(fig, f"pred_maps_comprehensive_{fname}", subdir="maps")
        if logger:
            logger.info(f"  Comprehensive maps saved: {pair}")


# ══════════════════════════════════════════════════════════════
# EXISTING FUNCTIONS (updated)
# ══════════════════════════════════════════════════════════════

def band_score_plot(scores: np.ndarray, good_bands: np.ndarray,
                    threshold: float) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, 6),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    colours = np.where(scores >= threshold, "#2ca02c", "#d62728")
    ax.bar(range(len(scores)), scores, color=colours, width=1.0, linewidth=0)
    ax.axhline(threshold, color="black", lw=2, ls="--",
               label=f"Threshold={threshold}")
    for region, colour, label in [
        ((107, 113), "#ffcccc", "H2O abs. ~1400nm"),
        ((153, 168), "#ffe0cc", "H2O/CO2 abs. ~1900nm"),
    ]:
        ax.axvspan(*region, alpha=0.35, color=colour, label=label)
    ax.set_xlabel("Band index", fontsize=12)
    ax.set_ylabel("Composite quality score", fontsize=12)
    ax.set_title("Band Quality: Gaussian + Task-Aware Fisher + Spectral Continuity",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, len(scores)); ax.grid(axis="y", alpha=0.3)
    ax2 = axes[1]
    mask = np.zeros(len(scores)); mask[good_bands] = 1
    ax2.imshow(mask.reshape(1, -1), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
               extent=[0, len(scores), 0, 1])
    ax2.set_yticks([])
    ax2.set_xlabel("Band index", fontsize=11)
    ax2.set_title(f"Selected: {len(good_bands)}/{len(scores)} bands "
                  f"({len(scores)-len(good_bands)} removed)", fontsize=10)
    patches = [mpatches.Patch(color="#2ca02c", label="Good"),
               mpatches.Patch(color="#d62728", label="Removed")]
    ax2.legend(handles=patches, loc="upper right", fontsize=8)
    plt.tight_layout()
    _save(fig, "band_quality_scores")


def training_curves(histories: Dict[str, Dict], seed: int) -> None:
    models = list(histories.keys())
    n = len(models)
    if n == 0: return
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1: axes = axes.reshape(1, 2)
    for i, mname in enumerate(models):
        h   = histories[mname]
        ep  = range(1, len(h["train_loss"]) + 1)
        col = MODEL_COLOURS.get(mname, "steelblue")
        axes[i, 0].plot(ep, h["train_loss"], color=col, lw=1.5, label="Train")
        axes[i, 0].plot(ep, h["val_loss"],   color=col, lw=1.5, ls="--", label="Val")
        axes[i, 0].set_title(f"{mname} - Loss", fontweight="bold")
        axes[i, 0].legend(); axes[i, 0].grid(alpha=0.3)
        axes[i, 1].plot(ep, h["train_acc"], color=col, lw=1.5, label="Train")
        axes[i, 1].plot(ep, h["val_acc"],   color=col, lw=1.5, ls="--", label="Val")
        axes[i, 1].set_title(f"{mname} - Accuracy", fontweight="bold")
        axes[i, 1].set_ylim(0, 1.05)
        axes[i, 1].legend(); axes[i, 1].grid(alpha=0.3)
    plt.suptitle(f"DL Training Curves (seed={seed})", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, f"training_curves_seed{seed}")


def metrics_heatmap(per_image_df: pd.DataFrame,
                    metric: str = "F1_oil", seed: int = 0) -> None:
    pivot = per_image_df.pivot(index="image", columns="model", values=metric)
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns)*1.8),
                                    max(5, len(pivot)*0.8)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_title(f"Per-Image {metric} - All Models (seed={seed})",
                 fontsize=13, fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            # Bold if max in row
            row_max = pivot.values[i, :].max()
            fw = "bold" if abs(v - row_max) < 1e-4 else "normal"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, fontweight=fw,
                    color="black" if v > 0.4 else "white")
    plt.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    _save(fig, f"heatmap_{metric}_seed{seed}")


def metrics_bar_chart(agg_list: List[Dict], seed: int) -> None:
    models    = [d["model"] for d in agg_list]
    f1_means  = [d["F1_oil_mean"]  for d in agg_list]
    f1_stds   = [d["F1_oil_std"]   for d in agg_list]
    auc_means = [d["AUC_PR_mean"]  for d in agg_list]
    auc_stds  = [d["AUC_PR_std"]   for d in agg_list]
    colours   = [MODEL_COLOURS.get(m, "steelblue") for m in models]
    x, w = np.arange(len(models)), 0.35
    fig, ax = plt.subplots(figsize=(max(12, len(models)*1.8), 6))
    ax.bar(x-w/2, f1_means,  w, yerr=f1_stds,  label="F1 (Oil)",
           color=colours, alpha=0.85, capsize=4)
    ax.bar(x+w/2, auc_means, w, yerr=auc_stds, label="AUC-PR",
           color=colours, alpha=0.55, capsize=4, hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Model Comparison (seed={seed})", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(ax.patches[:len(models)], f1_means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig, f"metrics_bar_seed{seed}")


def prediction_maps_row(
    img_name:   str,
    gt:         np.ndarray,
    prob_maps:  Dict[str, np.ndarray],
    pred_maps:  Dict[str, np.ndarray],
    H: int, W: int,
    rgb: Optional[np.ndarray] = None,
) -> None:
    model_names = list(prob_maps.keys())
    n_cols = 2 + len(model_names)
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(n_cols * 3.5, H/W * 3.5 + 1.5),
                             squeeze=False)
    axes = axes[0]
    if rgb is not None:
        axes[0].imshow(rgb)
        axes[0].set_title("Pseudo-RGB\n(B9/B19/B29)", fontsize=9, fontweight="bold")
    else:
        axes[0].axis("off")
    axes[0].axis("off")
    axes[1].imshow(gt, cmap=PRED_CMAP, vmin=0, vmax=1)
    oil_pct = 100 * gt.sum() / gt.size
    axes[1].set_title(f"Ground Truth\n{oil_pct:.1f}% oil",
                      fontsize=9, fontweight="bold")
    axes[1].axis("off")
    for ci, mname in enumerate(model_names, start=2):
        pred = pred_maps[mname].reshape(H, W)
        prob = prob_maps[mname]
        f1   = f1_score(gt.flatten(), pred.flatten(), pos_label=1, zero_division=0)
        ap   = average_precision_score(gt.flatten(), prob)
        col  = MODEL_COLOURS.get(mname, "black")
        axes[ci].imshow(pred, cmap=PRED_CMAP, vmin=0, vmax=1)
        axes[ci].set_title(f"{mname}\nF1={f1:.3f}  AUC={ap:.3f}",
                           fontsize=8, fontweight="bold", color=col)
        axes[ci].axis("off")
    patches = [mpatches.Patch(color="#1a6faf", label="Water"),
               mpatches.Patch(color="#d62728", label="Oil")]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    date, res = IMAGE_META.get(img_name, ("?", "?"))
    fig.suptitle(f"{img_name}  ({date}, {res}m)  |  All model predictions",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, f"pred_maps_{img_name}", subdir="maps")


def pr_roc_curves(
    img_name:  str,
    gt_flat:   np.ndarray,
    prob_maps: Dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for mname, prob in prob_maps.items():
        col = MODEL_COLOURS.get(mname, "steelblue")
        p, r, _ = precision_recall_curve(gt_flat, prob)
        axes[0].plot(r, p, label=f"{mname} ({auc(r,p):.3f})", color=col, lw=1.8)
        fpr, tpr, _ = roc_curve(gt_flat, prob)
        axes[1].plot(fpr, tpr, label=f"{mname} ({auc(fpr,tpr):.3f})",
                     color=col, lw=1.8)
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title(f"PR Curve [{img_name}]", fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].plot([0,1],[0,1],"k--",lw=1)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title(f"ROC Curve [{img_name}]", fontweight="bold")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.suptitle(f"PR & ROC — {img_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, f"pr_roc_{img_name}")


def confusion_matrix_grid(
    img_name:  str,
    gt_flat:   np.ndarray,
    pred_maps: Dict[str, np.ndarray],
) -> None:
    model_names = list(pred_maps.keys())
    n    = len(model_names)
    cols = min(4, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for ax, mname in zip(axes_flat, model_names):
        pred = pred_maps[mname]
        cm   = confusion_matrix(gt_flat, pred, labels=[0,1], normalize="true")
        im   = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:.3f}", ha="center", va="center",
                        fontsize=13, color="white" if cm[i,j]>0.5 else "black")
        ax.set_xticks([0,1]); ax.set_xticklabels(["Water","Oil"])
        ax.set_yticks([0,1]); ax.set_yticklabels(["Water","Oil"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(mname, fontweight="bold",
                     color=MODEL_COLOURS.get(mname,"black"))
        plt.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes_flat[n:]: ax.axis("off")
    plt.suptitle(f"Confusion Matrices (normalised) — {img_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, f"confusion_matrices_{img_name}")


# ══════════════════════════════════════════════════════════════
# 5. SUMMARY TABLE WITH BOLD HIGHEST VALUES (Requirement 5)
# ══════════════════════════════════════════════════════════════

def summary_table_plot(agg_list: List[Dict]) -> None:
    """
    Requirement 5: Bold highest value per column.
    Shows F1, IoU, AUC-PR, AUC-ROC, Kappa, MCC, Recall, Precision.
    """
    df = pd.DataFrame(agg_list).sort_values("F1_oil_mean", ascending=False)

    # Columns to show
    show_cols = [
        ("F1±std",      "F1_oil_mean",   "F1_oil_std"),
        ("IoU±std",     "IoU_mean",       "IoU_std"),
        ("AUC-PR±std",  "AUC_PR_mean",    "AUC_PR_std"),
        ("AUC-ROC",     "AUC_ROC_mean",   None),
        ("Recall",      "Recall_mean",    None),
        ("Precision",   "Precision_mean", None),
        ("Kappa",       "Kappa_mean",     None),
        ("MCC",         "MCC_mean",       None),
    ]

    col_labels = ["Model"] + [c[0] for c in show_cols]

    # Build data
    data = []
    for _, row in df.iterrows():
        r = [row["model"]]
        for col_name, mean_key, std_key in show_cols:
            val = row.get(mean_key, 0)
            if std_key:
                std = row.get(std_key, 0)
                r.append(f"{val:.4f}±{std:.4f}")
            else:
                r.append(f"{val:.4f}")
        data.append(r)

    # Find best value per metric column (for bolding)
    best_per_col = {}
    for ci, (_, mean_key, _) in enumerate(show_cols):
        vals = [row.get(mean_key, 0) for _, row in df.iterrows()]
        best_per_col[ci + 1] = max(vals)  # +1 for Model column

    n_rows = len(data)
    n_cols = len(col_labels)

    fig_h = max(4, n_rows * 0.7 + 2)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH + 2, fig_h))
    ax.axis("off")

    tbl = ax.table(cellText=data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 2.0)

    # Header style
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Row styling + bold best values
    row_colours = [
        "#FFE8F0",  # PhysSpecNet
        "#E8F4FF",  # DL models
        "#E8FFE8",  # Classical
        "#FFF8E8",  # GNB
    ]
    for ri, (_, row) in enumerate(df.iterrows(), start=1):
        # Row background
        mname = row["model"]
        bg = MODEL_COLOURS.get(mname, "#FFFFFF") + "33"
        for j in range(n_cols):
            tbl[ri, j].set_facecolor(bg)

        # Bold if best in column
        for ci, (_, mean_key, _) in enumerate(show_cols):
            col_idx = ci + 1
            val = row.get(mean_key, 0)
            if abs(val - best_per_col[col_idx]) < 1e-6:
                cell = tbl[ri, col_idx]
                cell.set_text_props(fontweight="bold", color="#B22222")

    ax.set_title(
        "Model Comparison Summary — mean across test images\n"
        "(Bold red = best per metric)",
        fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    _save(fig, "summary_table")
