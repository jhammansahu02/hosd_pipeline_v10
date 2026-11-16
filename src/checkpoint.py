"""
Checkpoint system — Mode-specific paths
"""
import json, pickle, shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


def _get_state_file() -> Path:
    import config as _cfg
    return _cfg.OUTPUT_ROOT / "checkpoints" / "pipeline_state.json"


def _load_state() -> Dict:
    f = _get_state_file()
    if f.exists():
        with open(f) as fp:
            return json.load(fp)
    return {"completed": [], "metrics": {},
            "started": str(datetime.now())}


def _save_state(state: Dict) -> None:
    f = _get_state_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    with open(tmp, "w") as fp:
        json.dump(state, fp, indent=2)
    shutil.move(str(tmp), str(f))


def is_done(stage: str) -> bool:
    return stage in _load_state()["completed"]


def mark_done(stage: str, metrics: Optional[Dict] = None) -> None:
    state = _load_state()
    if stage not in state["completed"]:
        state["completed"].append(stage)
    if metrics:
        state["metrics"][stage] = metrics
    state["last_updated"] = str(datetime.now())
    _save_state(state)


def save_metrics(stage: str, metrics: Dict) -> None:
    state = _load_state()
    state["metrics"][stage] = metrics
    _save_state(state)


def get_metrics(stage: str) -> Optional[Dict]:
    return _load_state()["metrics"].get(stage)


def save_sklearn_model(model: Any, name: str, seed: int) -> Path:
    import config as _cfg
    path = _cfg.CKPT_DIR / f"{name}_seed{seed}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_sklearn_model(name: str, seed: int) -> Optional[Any]:
    import config as _cfg
    path = _cfg.CKPT_DIR / f"{name}_seed{seed}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_torch_checkpoint(state_dict: Dict, name: str, seed: int,
                          epoch: int, is_best: bool = False) -> Path:
    import config as _cfg, torch
    ckpt_dir = _cfg.CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{name}_seed{seed}_epoch{epoch}.pt"
    torch.save(state_dict, path)
    shutil.copy(str(path),
                str(ckpt_dir / f"{name}_seed{seed}_last.pt"))
    if is_best:
        shutil.copy(str(path),
                    str(ckpt_dir / f"{name}_seed{seed}_best.pt"))
    return path


def load_best_torch_checkpoint(name: str, seed: int) -> Optional[Dict]:
    import config as _cfg, torch
    path = _cfg.CKPT_DIR / f"{name}_seed{seed}_best.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def load_last_torch_checkpoint(name: str, seed: int) -> Optional[Dict]:
    import config as _cfg, torch
    path = _cfg.CKPT_DIR / f"{name}_seed{seed}_last.pt"
    if not path.exists():
        return load_best_torch_checkpoint(name, seed)
    return torch.load(path, map_location="cpu")


def print_progress() -> None:
    state = _load_state()
    print("\n" + "=" * 60)
    print("PIPELINE PROGRESS")
    print("=" * 60)
    for s in state["completed"]:
        print(f"  ✓  {s}")
    print("=" * 60 + "\n")
