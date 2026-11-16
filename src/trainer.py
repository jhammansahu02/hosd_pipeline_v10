"""
DL Trainer v10
==============
- Warmup (5 epochs) + ReduceLROnPlateau
- FocalLoss with smoothing
- SpectralFormer forced to CPU (Quadro P1000 CC 6.1)
- Resume from last checkpoint
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.amp as amp
from torch.utils.data import DataLoader, TensorDataset

from config import (BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
                    PATIENCE, MIXUP_ALPHA, WARMUP_EPOCHS)
from models.losses import FocalLossWithSmoothing, mixup_data, mixup_criterion
from src.checkpoint import save_torch_checkpoint, load_last_torch_checkpoint


def get_device():
    if torch.cuda.is_available():
        dev   = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled   = True
        print(f"[GPU] {props.name}  VRAM: {props.total_memory/1e9:.1f} GB")
    else:
        dev = torch.device("cpu")
        print(f"[CPU] {torch.get_num_threads()} threads")
    return dev


def _make_loaders(X, y, Xv, yv, device, batch_size=None):
    if batch_size is None:
        batch_size = BATCH_SIZE
    pin = device.type == "cuda"
    return (
        DataLoader(TensorDataset(torch.from_numpy(X).float(),
                                 torch.from_numpy(y).long()),
                   batch_size=batch_size, shuffle=True,
                   num_workers=0, pin_memory=pin),
        DataLoader(TensorDataset(torch.from_numpy(Xv).float(),
                                 torch.from_numpy(yv).long()),
                   batch_size=batch_size * 4, shuffle=False,
                   num_workers=0, pin_memory=pin),
    )


def _train_epoch(model, loader, opt, crit, device, scaler):
    model.train()
    tl = tc = n = 0
    for Xb, yb in loader:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if MIXUP_ALPHA > 0:
            Xb, ya, yb2, lam = mixup_data(Xb, yb, MIXUP_ALPHA, device)
        with amp.autocast(device_type=device.type,
                          enabled=(device.type == "cuda")):
            logits = model(Xb)
            loss   = (mixup_criterion(crit, logits, ya, yb2, lam)
                      if MIXUP_ALPHA > 0 else crit(logits, yb))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update()
        tl += loss.item() * len(yb)
        tc += (logits.argmax(1) == yb).sum().item()
        n  += len(yb)
    return tl / n, tc / n


@torch.no_grad()
def _val_epoch(model, loader, crit, device):
    model.eval()
    vl = vc = n = 0
    for Xb, yb in loader:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with amp.autocast(device_type=device.type,
                          enabled=(device.type == "cuda")):
            logits = model(Xb)
            loss   = crit(logits, yb)
        vl += loss.item() * len(yb)
        vc += (logits.argmax(1) == yb).sum().item()
        n  += len(yb)
    return vl / n, vc / n


def train_model(model_name, model, X_train, y_train, X_val, y_val,
                seed, logger=None, resume=True):
    torch.manual_seed(seed); np.random.seed(seed)

    # SpectralFormer → CPU (Quadro P1000 CC 6.1 limitation)
    device = get_device()
    # if "SpectralFormer" in model_name:
    #     device = torch.device("cpu")
    #     if logger:
    #         logger.info("  [SpectralFormer] Forced to CPU (CC 6.1)")

    _batch = BATCH_SIZE  # 8192 ← T4 16GB handle karega
    
    model  = model.to(device)
    crit   = FocalLossWithSmoothing()
    opt    = torch.optim.AdamW(model.parameters(),
                               lr=LR, weight_decay=WEIGHT_DECAY)

    def warmup_fn(epoch):
        return (epoch + 1) / WARMUP_EPOCHS if epoch < WARMUP_EPOCHS else 1.0

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_fn)
    reduce_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    scaler = amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    tr_dl, vl_dl = _make_loaders(X_train, y_train, X_val, y_val, device, batch_size=_batch)

    start_epoch   = 0
    best_val_loss = float("inf")
    patience_ctr  = 0
    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    if resume:
        ckpt = load_last_torch_checkpoint(model_name, seed)
        if ckpt:
            model.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["optim_state"])
            if "scaler_state" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch   = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            history       = ckpt.get("history", history)
            if logger:
                logger.info(f"  Resumed {model_name} from epoch {start_epoch}")

    if logger:
        n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Training {model_name} | seed={seed} | "
            f"params={n_par:,} | device={device} | "
            f"AMP {'fp16' if device.type=='cuda' else 'off'} | "
            f"epochs={start_epoch}->{EPOCHS} | warmup={WARMUP_EPOCHS}")

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = _train_epoch(model, tr_dl, opt, crit, device, scaler)
        vl_loss, vl_acc = _val_epoch(model, vl_dl, crit, device)

        if epoch < WARMUP_EPOCHS:
            warmup_sched.step(); phase = "warmup"
        else:
            reduce_sched.step(vl_loss); phase = "train"

        cur_lr = opt.param_groups[0]['lr']
        for k, v in zip(["train_loss","val_loss","train_acc","val_acc"],
                        [tr_loss, vl_loss, tr_acc, vl_acc]):
            history[k].append(v)

        is_best = vl_loss < best_val_loss
        if is_best:
            best_val_loss = vl_loss; patience_ctr = 0
        else:
            patience_ctr += 1

        save_torch_checkpoint({
            "model_state":   model.state_dict(),
            "optim_state":   opt.state_dict(),
            "scaler_state":  scaler.state_dict(),
            "epoch":         epoch,
            "best_val_loss": best_val_loss,
            "history":       history,
        }, model_name, seed, epoch, is_best)

        if logger and (epoch % 10 == 0 or is_best):
            logger.info(
                f"  [{model_name}] Epoch {epoch:03d}/{EPOCHS} "
                f"| TrLoss={tr_loss:.4f} TrAcc={tr_acc*100:.1f}% "
                f"| VlLoss={vl_loss:.4f} VlAcc={vl_acc*100:.1f}% "
                f"| LR={cur_lr:.2e} [{phase}] "
                f"| {'<<BEST>>' if is_best else f'pat={patience_ctr}/{PATIENCE}'} "
                f"| {time.time()-t0:.1f}s")

        if patience_ctr >= PATIENCE:
            if logger: logger.info(f"  Early stop epoch {epoch}")
            break

    # Load best for inference
    from src.checkpoint import load_best_torch_checkpoint
    ckpt = load_best_torch_checkpoint(model_name, seed)
    if ckpt:
        model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, history
