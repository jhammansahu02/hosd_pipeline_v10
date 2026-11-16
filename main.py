"""
HOSD Oil Spill Pipeline v10
============================
4 Normalization Modes:
  baseline : No L1, No Instance Norm
  onlyL1   : Only L1 Normalization
  onlyIN   : Only Instance Normalization
  full     : L1 + Instance Norm (best)

Usage:
  python main.py --mode baseline --seed 42
  python main.py --mode onlyL1  --seed 42
  python main.py --mode onlyIN  --seed 42
  python main.py --mode full    --seed 42
  python main.py --mode full    --seed 42 --skip_dl
  python main.py --mode full    --status
"""

import os, sys, gc, time, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Parse args FIRST ──────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode",    default="full",
                    choices=["baseline","onlyL1","onlyIN","full"])
parser.add_argument("--seed",    type=int, default=None)
parser.add_argument("--force",   action="store_true")
parser.add_argument("--status",  action="store_true")
parser.add_argument("--skip_dl", action="store_true")
args = parser.parse_args()

# ── Set flags BEFORE config imports ───────────────────────────
import config as _cfg
MODE = args.mode

if   MODE == "baseline": _cfg.L1_NORMALISE=False; _cfg.INSTANCE_NORM=False
elif MODE == "onlyL1":   _cfg.L1_NORMALISE=True;  _cfg.INSTANCE_NORM=False
elif MODE == "onlyIN":   _cfg.L1_NORMALISE=False;  _cfg.INSTANCE_NORM=True
elif MODE == "full":     _cfg.L1_NORMALISE=True;  _cfg.INSTANCE_NORM=True

_cfg.OUTPUT_ROOT = Path(f"./results/{MODE}")
_cfg.LOG_DIR     = Path(f"./logs/{MODE}")
_cfg.CKPT_DIR    = _cfg.OUTPUT_ROOT / "checkpoints"

for _d in [_cfg.OUTPUT_ROOT, _cfg.LOG_DIR, _cfg.CKPT_DIR,
           _cfg.OUTPUT_ROOT/"figures", _cfg.OUTPUT_ROOT/"maps",
           _cfg.OUTPUT_ROOT/"metrics"]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Imports ───────────────────────────────────────────────────
from config import (RANDOM_SEEDS, TRAIN_IMAGES, VAL_IMAGES,
                    TEST_IMAGES, THRESHOLD_IMAGES,
                    DATA_ROOT, IMAGE_FILES)
from src.logger    import get_logger
from src.checkpoint import (is_done, mark_done, save_sklearn_model,
                             load_sklearn_model, print_progress)
from src.data_loader  import compute_train_norm_stats, load_preprocessed
from src.band_selector import select_good_bands
from src.sampler       import build_train_set, build_val_set
from src.evaluator     import (evaluate_all_models_all_images,
                                calibrate_all_thresholds,
                                save_results, compile_multiseed_results)
from src.visualizer    import (band_score_plot, metrics_heatmap,
                                metrics_bar_chart, summary_table_plot,
                                training_curves, visualize_all_images_gt,
                                comprehensive_prediction_maps,
                                prediction_maps_row, pr_roc_curves,
                                confusion_matrix_grid)
from models.classical_models import get_classical_models

LOG = get_logger("hosd", _cfg.LOG_DIR)
NORM_CACHE = _cfg.OUTPUT_ROOT / "metrics" / "norm_stats.npz"


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _predict_sklearn(model, batch_size=8192):
    def _fn(X):
        probs = []
        for i in range(0, len(X), batch_size):
            probs.append(model.predict_proba(X[i:i+batch_size])[:,1])
        return np.concatenate(probs)
    return _fn


def _predict_dl(model, device, batch_size=65536):
    import torch, torch.amp as amp
    def _fn(X):
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                Xb = torch.from_numpy(X[i:i+batch_size]).float().to(
                    device, non_blocking=True)
                with amp.autocast(device_type=device.type,
                                  enabled=(device.type=="cuda")):
                    logits = model(Xb)
                probs.append(
                    torch.softmax(logits,1)[:,1].float().cpu().numpy())
        return np.concatenate(probs)
    return _fn


def _save_thr(thr, seed):
    p = _cfg.OUTPUT_ROOT/"metrics"/f"thresholds_{seed}.json"
    with open(p,"w") as f: json.dump(thr, f, indent=2)


def _load_thr(seed):
    p = _cfg.OUTPUT_ROOT/"metrics"/f"thresholds_{seed}.json"
    return json.load(open(p)) if p.exists() else {}


def _aggregate_csv(df, model_names):
    agg = []
    cols = ["OA","F1_oil","IoU","Precision","Recall",
            "AUC_ROC","AUC_PR","Kappa","MCC"]
    for mname in model_names:
        sub = df[df["model"]==mname]
        if sub.empty: continue
        a = {"model": mname}
        for col in cols:
            if col in sub.columns:
                a[f"{col}_mean"] = float(sub[col].mean())
                a[f"{col}_std"]  = float(sub[col].std())
        agg.append(a)
    return agg


# ══════════════════════════════════════════════════════════════
# STAGES
# ══════════════════════════════════════════════════════════════

def stage_band_selection(force=False):
    if not force and is_done("band_selection"):
        LOG.info("[SKIP] Band selection done.")
        return select_good_bands(logger=LOG)
    LOG.info("="*60)
    LOG.info("STAGE 0: Band Selection (L1-scored, mode-independent)")
    LOG.info("="*60)
    good_bands, scores = select_good_bands(logger=LOG, force_recompute=True)
    band_score_plot(scores, good_bands, _cfg.COMPOSITE_SCORE_THR)
    mark_done("band_selection", {"n_good_bands": int(len(good_bands))})
    return good_bands, scores


def stage_build_dataset(good_bands, mean, std, seed, force=False):
    cache = _cfg.OUTPUT_ROOT/"metrics"/f"dataset_seed{seed}.npz"
    if not force and cache.exists():
        LOG.info(f"[SKIP] Dataset seed={seed}.")
        d = np.load(cache)
        return (d["X_train"].astype(np.float32), d["y_train"],
                d["X_val"].astype(np.float32),   d["y_val"])
    LOG.info(f"Building dataset seed={seed} | "
             f"L1={_cfg.L1_NORMALISE} IN={_cfg.INSTANCE_NORM}")
    X_tr, y_tr = build_train_set(good_bands, mean, std, seed=seed, logger=LOG)
    X_vl, y_vl = build_val_set(good_bands, mean, std, seed=seed, logger=LOG)
    LOG.info(f"  Train:{len(X_tr):,} Val:{len(X_vl):,}")
    np.savez(cache, X_train=X_tr, y_train=y_tr, X_val=X_vl, y_val=y_vl)
    return X_tr, y_tr, X_vl, y_vl


def stage_classical(X_tr, y_tr, good_bands, mean, std,
                    seed, skip_done=True, force=False):
    LOG.info("="*60)
    LOG.info(f"STAGE 2: Classical (seed={seed})")
    LOG.info("="*60)

    eval_key  = f"classical_eval_seed{seed}"
    csv_path  = _cfg.OUTPUT_ROOT/"metrics"/f"classical_seed{seed}.csv"
    models    = get_classical_models()
    clf_names = list(models.keys())
    pfns      = {}
    any_new   = False

    for mname, clf in models.items():
        key  = f"classical_{mname}_seed{seed}"
        ckpt = load_sklearn_model(mname, seed)
        if skip_done and not force and is_done(key) and ckpt:
            LOG.info(f"  [SKIP] {mname}")
            pfns[mname] = _predict_sklearn(ckpt)
        else:
            LOG.info(f"  Fitting {mname} ...")
            t0 = time.time()
            clf.fit(X_tr, y_tr)
            LOG.info(f"  {mname} done in {time.time()-t0:.1f}s")
            save_sklearn_model(clf, mname, seed)
            pfns[mname] = _predict_sklearn(clf)
            mark_done(key); any_new = True

    if not force and not any_new and is_done(eval_key) and csv_path.exists():
        LOG.info(f"  [SKIP] Classical eval done")
        df  = pd.read_csv(csv_path)
        thr = {k: _load_thr(seed).get(k,0.5) for k in clf_names}
        return pfns, _aggregate_csv(df, clf_names), [df], thr

    tc = _load_thr(seed)
    if any_new or not all(m in tc for m in clf_names):
        thr = calibrate_all_thresholds(pfns, good_bands, mean, std,
              threshold_images=THRESHOLD_IMAGES, logger=LOG)
        tc.update(thr); _save_thr(tc, seed)
    else:
        thr = {k: tc[k] for k in clf_names if k in tc}

    df, agg_list, _, _ = evaluate_all_models_all_images(
        pfns, good_bands, mean, std, thresholds=thr, logger=LOG)
    agg = []
    for a in agg_list:
        mark_done(f"classical_{a['model']}_seed{seed}", a)
        agg.append(a)
    df.to_csv(csv_path, index=False)
    mark_done(eval_key)
    return pfns, agg, [df], thr


def stage_dl(X_tr, y_tr, X_vl, y_vl,
             good_bands, mean, std, seed,
             skip_done=True, force=False):
    try:
        import torch
        from src.trainer         import train_model, get_device
        from models.dl_baselines import get_dl_baselines
        from models.physspec_net import get_physspec_net
    except ImportError as e:
        LOG.warning(f"PyTorch unavailable: {e}")
        return {}, [], [], {}

    LOG.info("="*60)
    LOG.info(f"STAGE 3: DL Models (seed={seed})")
    LOG.info("="*60)

    eval_key  = f"dl_eval_seed{seed}"
    csv_path  = _cfg.OUTPUT_ROOT/"metrics"/f"dl_seed{seed}.csv"
    device    = get_device()
    in_bands  = X_tr.shape[1]

    all_models = get_dl_baselines(in_bands)
    all_models["PhysSpecNet"] = get_physspec_net(in_bands)
    dl_names = list(all_models.keys())

    for mname, m in all_models.items():
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        LOG.info(f"  {mname}: {n:,} params")

    pfns = {}; histories = {}; any_new = False

    for mname, model in all_models.items():
        key = f"dl_{mname}_seed{seed}"
        # SpectralFormer → CPU
        edev = torch.device("cpu") \
               if "SpectralFormer" in mname else device

        if skip_done and not force and is_done(key):
            LOG.info(f"  [SKIP] {mname}")
            from src.checkpoint import load_best_torch_checkpoint
            ckpt = load_best_torch_checkpoint(mname, seed)
            if ckpt:
                model.load_state_dict(ckpt["model_state"])
                model.to(edev).eval()
                pfns[mname] = _predict_dl(model, edev)
        else:
            trained, hist = train_model(
                mname, model, X_tr, y_tr, X_vl, y_vl,
                seed=seed, logger=LOG, resume=True)
            histories[mname] = hist
            pfns[mname] = _predict_dl(trained, edev)
            mark_done(key); any_new = True

    if histories:
        training_curves(histories, seed)

    if not force and not any_new and is_done(eval_key) and csv_path.exists():
        LOG.info(f"  [SKIP] DL eval done")
        df  = pd.read_csv(csv_path)
        thr = {k: _load_thr(seed).get(k,0.5) for k in dl_names}
        return pfns, _aggregate_csv(df, dl_names), [df], histories

    tc = _load_thr(seed)
    if any_new or not all(m in tc for m in dl_names):
        thr = calibrate_all_thresholds(pfns, good_bands, mean, std,
              threshold_images=THRESHOLD_IMAGES, logger=LOG)
        tc.update(thr); _save_thr(tc, seed)
    else:
        thr = {k: tc[k] for k in dl_names if k in tc}

    df, agg_list, _, _ = evaluate_all_models_all_images(
        pfns, good_bands, mean, std, thresholds=thr, logger=LOG)
    df.to_csv(csv_path, index=False)
    mark_done(eval_key)
    return pfns, list(agg_list), [df], histories


def stage_ensemble(all_pfns, all_agg, good_bands, mean, std, seed):
    """Soft voting ensemble — only for full mode."""
    LOG.info("="*60)
    LOG.info(f"STAGE 4: Soft Voting Ensemble (seed={seed})")
    LOG.info("="*60)

    from sklearn.metrics import (f1_score as sk_f1,
        average_precision_score, cohen_kappa_score,
        matthews_corrcoef, accuracy_score, confusion_matrix)

    # Top 4 by F1
    f1s  = {a["model"]: a.get("F1_oil_mean",0) for a in all_agg
            if a["model"] in all_pfns}
    top4 = sorted(f1s, key=f1s.get, reverse=True)[:4]
    LOG.info(f"  Top 4: {top4}")
    efns = {m: all_pfns[m] for m in top4}

    # Calibrate on GM07
    px, lb, H, W = load_preprocessed("GM07", good_bands, mean, std)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(px), min(20000, len(px)), replace=False)
    Xc, yc = px[idx], lb[idx]; del px, lb; gc.collect()

    probs_c = np.mean([fn(Xc) for fn in efns.values()], axis=0)
    best_thr, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 80):
        f1 = sk_f1(yc, (probs_c>=t).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, float(t)
    LOG.info(f"  Threshold={best_thr:.3f} GM07_F1={best_f1:.4f}")
    del Xc, yc; gc.collect()

    rows = []
    for img in TEST_IMAGES:
        px, lb, H, W = load_preprocessed(img, good_bands, mean, std)
        probs = np.mean([fn(px) for fn in efns.values()], axis=0)
        pred  = (probs >= best_thr).astype(np.uint8)
        tn, fp, fn_c, tp = confusion_matrix(lb, pred, labels=[0,1]).ravel()
        rows.append({
            "image":     img, "model": "SoftEnsemble",
            "OA":        float(accuracy_score(lb, pred)),
            "F1_oil":    float(2*tp/(2*tp+fp+fn_c+1e-8)),
            "IoU":       float(tp/(tp+fp+fn_c+1e-8)),
            "Precision": float(tp/(tp+fp+1e-8)),
            "Recall":    float(tp/(tp+fn_c+1e-8)),
            "Specificity": float(tn/(tn+fp+1e-8)),
            "AUC_PR":    float(average_precision_score(lb, probs)),
            "AUC_ROC":   float(__import__('sklearn.metrics',
                               fromlist=['roc_auc_score']).roc_auc_score(lb, probs)),
            "Kappa":     float(cohen_kappa_score(lb, pred)),
            "MCC":       float(matthews_corrcoef(lb, pred)),
        })
        LOG.info(f"  [{img}] F1={rows[-1]['F1_oil']:.4f} "
                 f"AUC-PR={rows[-1]['AUC_PR']:.4f}")
        del px, lb; gc.collect()

    df = pd.DataFrame(rows)
    LOG.info(f"  SUMMARY SoftEnsemble: "
             f"F1={df['F1_oil'].mean():.4f}+-{df['F1_oil'].std():.4f} "
             f"AUC-PR={df['AUC_PR'].mean():.4f}")
    df.to_csv(_cfg.OUTPUT_ROOT/"metrics"/f"ensemble_seed{seed}.csv", index=False)

    agg = {"model": "SoftEnsemble"}
    for col in ["OA","F1_oil","IoU","Precision","Recall",
                "Specificity","AUC_PR","AUC_ROC","Kappa","MCC"]:
        agg[f"{col}_mean"] = float(df[col].mean())
        agg[f"{col}_std"]  = float(df[col].std())
    return df, agg


def stage_visualize(all_pfns, all_thr, all_agg, per_image_df,
                    good_bands, mean, std, band_scores, seed,
                    clf_names=None, dl_names=None, force=False):
    viz_key = f"visualize_seed{seed}"
    if not force and is_done(viz_key):
        LOG.info(f"  [SKIP] Viz done"); return

    LOG.info("="*60); LOG.info(f"STAGE 5: Visualisation (seed={seed})")
    LOG.info("="*60)

    if per_image_df is not None and not per_image_df.empty:
        for metric in ["F1_oil","AUC_PR","IoU","Kappa"]:
            if metric in per_image_df.columns:
                metrics_heatmap(per_image_df, metric, seed)

    if all_agg:
        metrics_bar_chart(all_agg, seed)
        df_agg = pd.DataFrame(all_agg)
        if "F1_oil_mean" in df_agg.columns:
            df_agg = df_agg.sort_values("F1_oil_mean", ascending=False)
        summary_table_plot(df_agg.to_dict('records'))

    LOG.info("  Building prediction maps ...")
    _, _, prob_cache, pred_cache = evaluate_all_models_all_images(
        all_pfns, good_bands, mean, std,
        thresholds=all_thr, image_list=TEST_IMAGES, logger=LOG)

    from src.data_loader import load_mat
    gt_c={}; rgb_c={}; H_c={}; W_c={}

    for img in TEST_IMAGES:
        try:
            img_raw, gt = load_mat(img)
            H, W = gt.shape
            H_c[img]=H; W_c[img]=W; gt_c[img]=gt.flatten()
            def _n(ch):
                lo,hi=np.percentile(ch,2),np.percentile(ch,98)
                return np.clip((ch-lo)/(hi-lo+1e-8),0,1)
            rgb = np.stack([_n(img_raw[:,:,29].astype(np.float32)),
                            _n(img_raw[:,:,19].astype(np.float32)),
                            _n(img_raw[:,:,9].astype(np.float32))], axis=-1)
            rgb_c[img]=rgb; del img_raw
            prediction_maps_row(img, gt, prob_cache[img],
                                pred_cache[img], H, W, rgb)
            pr_roc_curves(img, gt.flatten(), prob_cache[img])
            confusion_matrix_grid(img, gt.flatten(), pred_cache[img])
        except Exception as e:
            LOG.warning(f"  Viz error {img}: {e}")

    if clf_names and dl_names:
        try:
            phys  = [m for m in pred_cache.get(TEST_IMAGES[0],{})
                     if "PhysSpec" in m]
            dl_no = [m for m in dl_names if "PhysSpec" not in m]
            comprehensive_prediction_maps(
                img_names=TEST_IMAGES,
                prob_cache=prob_cache, pred_cache=pred_cache,
                gt_cache=gt_c, H_cache=H_c, W_cache=W_c,
                rgb_cache=rgb_c, classical_models=clf_names,
                dl_models=dl_no+phys, logger=LOG)
        except Exception as e:
            LOG.warning(f"  Comp maps error: {e}")

    mark_done(viz_key)
    LOG.info("  Viz done.")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    if args.status:
        print_progress(); return

    LOG.info("="*60)
    LOG.info(f"   HOSD Pipeline v10 — Mode: {MODE.upper()}")
    LOG.info("="*60)
    LOG.info(f"L1={_cfg.L1_NORMALISE} | IN={_cfg.INSTANCE_NORM}")
    LOG.info(f"Train:{TRAIN_IMAGES}")
    LOG.info(f"Val:{VAL_IMAGES} | Thr:{THRESHOLD_IMAGES}")
    LOG.info(f"Test:{TEST_IMAGES}")
    LOG.info(f"Output:{_cfg.OUTPUT_ROOT}")

    missing = [n for n,f in IMAGE_FILES.items()
               if not (DATA_ROOT/f).exists()]
    if missing:
        LOG.error(f"Missing: {missing[:3]}"); sys.exit(1)

    seeds   = [args.seed] if args.seed else RANDOM_SEEDS
    force   = args.force
    t_start = time.time()

    # GT visualization
    if not is_done("gt_visualization") or force:
        try:
            visualize_all_images_gt(logger=LOG)
            mark_done("gt_visualization")
        except Exception as e:
            LOG.warning(f"GT viz error: {e}")

    # Band selection
    good_bands, band_scores = stage_band_selection(force=force)
    LOG.info(f"Good bands: {len(good_bands)}/224")

    # Norm stats
    if force and NORM_CACHE.exists():
        NORM_CACHE.unlink()
    if not NORM_CACHE.exists():
        mean, std = compute_train_norm_stats(good_bands, logger=LOG)
        np.savez(NORM_CACHE, mean=mean, std=std)
    else:
        d = np.load(NORM_CACHE)
        mean = d["mean"].astype(np.float32)
        std  = d["std"].astype(np.float32)

    all_agg_global = []

    for seed in seeds:
        LOG.info(f"\n{'='*60}")
        LOG.info(f"SEED {seed} | Mode: {MODE}")
        LOG.info(f"{'='*60}")

        X_tr, y_tr, X_vl, y_vl = stage_build_dataset(
            good_bands, mean, std, seed, force=force)

        all_pfns = {}; all_thr = _load_thr(seed)
        all_agg  = []; all_pi  = []

        # Classical
        cp, ca, cpi, ct = stage_classical(
            X_tr, y_tr, good_bands, mean, std, seed,
            skip_done=(not force), force=force)
        all_pfns.update(cp); all_thr.update(ct)
        all_agg += ca; all_pi += cpi
        clf_names = list(cp.keys())

        # DL
        dl_names = []
        if not args.skip_dl:
            dp, da, dpi, dh = stage_dl(
                X_tr, y_tr, X_vl, y_vl,
                good_bands, mean, std, seed,
                skip_done=(not force), force=force)
            all_pfns.update(dp)
            all_thr.update({k:v for k,v in _load_thr(seed).items()
                            if k not in all_thr})
            all_agg += da; all_pi += dpi
            dl_names = list(dp.keys())

        # Ensemble — only in full mode
        if MODE == "full" and len(all_pfns) >= 4 and not args.skip_dl:
            _, ens_agg = stage_ensemble(
                all_pfns, all_agg, good_bands, mean, std, seed)
            all_agg.append(ens_agg)

        if all_pi:
            combined = pd.concat(all_pi, ignore_index=True)
            save_results(combined, all_agg, seed)
            all_agg_global += all_agg
            if all_pfns:
                stage_visualize(all_pfns, all_thr, all_agg, combined,
                                good_bands, mean, std, band_scores, seed,
                                clf_names=clf_names, dl_names=dl_names,
                                force=force)

    if len(seeds) > 1:
        ms = compile_multiseed_results(seeds)
        if not ms.empty:
            ms.to_csv(_cfg.OUTPUT_ROOT/"metrics"/"multiseed.csv",
                      index=False)

    mark_done("pipeline_complete")
    LOG.info(f"\nDone in {(time.time()-t_start)/60:.1f} min")
    LOG.info(f"Results → {_cfg.OUTPUT_ROOT}")
    print_progress()


if __name__ == "__main__":
    main()
