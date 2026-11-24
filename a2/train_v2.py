"""Longer, larger-patch training runs ("v2") for the final pipeline.

Same U-Net, loss (BCE + soft Dice), optimiser (AdamW) and cosine schedule as a2/train.py. Changes:
* fast_conv: the exact MPS-friendly layers of a2/fastconv.py (about 3x faster training);
* 96^3 patches x 2 for 40,000 iterations, about 13x the voxels seen by the 64^3 runs;
* stronger training-only augmentation (a2/sampling_v2.py);
* validation on a fixed, evenly spread subset of `val_cases` validation cases during training, and
  on all validation cases once at the end (no checkpoint is selected: the final weights are kept);
* a checkpoint (weights, optimiser, schedule, history) every `eval_every` iterations, from which an
  interrupted run resumes exactly, because patch i of the stream depends only on (seed, i).

Run from the assignment folder (both runs, one after the other):
    python -m a2.train_v2 uniform96 tumour96
Final weights are written to models/{name}.pt in the v1 format and load into the plain UNet3D.
"""

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .fastconv import make_fast
from .model import UNet3D, count_parameters
from .sampling_v2 import PatchDatasetV2
from .train import TrainConfig, dice_bce_loss, pick_device, seed_everything, state_hash, validate


@dataclass(frozen=True)
class TrainConfigV2(TrainConfig):
    fast_conv: bool = True
    val_cases: int = 20        # validation cases checked during training; all are evaluated at the end
    p_spatial: float = 0.25    # probability of rotation + scaling
    p_gamma: float = 0.3       # per-channel probability of a gamma change
    p_noise: float = 0.15      # probability of Gaussian noise
    num_workers: int = 10


def val_subset(val_ids, n):
    """n validation cases spread evenly over the fixed list (all of them if n <= 0 or n >= len)."""
    if n <= 0 or n >= len(val_ids):
        return list(val_ids)
    return [val_ids[i] for i in np.linspace(0, len(val_ids) - 1, n).round().astype(int)]


def _paths(out_dir, ckpt_dir, name):
    return Path(out_dir) / f"{name}.pt", Path(out_dir) / f"{name}_history.json", Path(ckpt_dir) / f"{name}.partial.pt"


def _cpu(obj):
    """Tensors to CPU and NumPy scalars to Python numbers, so the checkpoint loads with weights_only=True."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu(v) for v in obj)
    return obj


def load_run_v2(cfg, out_dir="models", device=None):
    """Finished run with exactly this config as a plain UNet3D, plus its history; otherwise None."""
    ckpt, hist, _ = _paths(out_dir, ".", cfg.name)
    if not ckpt.exists():
        return None
    saved = torch.load(ckpt, map_location="cpu", weights_only=True)
    if saved["config"] != asdict(cfg):
        return None
    model = UNet3D(features=cfg.features)
    model.load_state_dict(saved["state_dict"])
    return model.to(device or pick_device()), json.loads(hist.read_text())


def train_v2(cfg, cache_dir, train_ids, val_ids, out_dir="models", ckpt_dir="cache/checkpoints", device=None,
             log=print):
    device = device or pick_device()
    found = load_run_v2(cfg, out_dir, device)
    if found is not None:
        log(f"[{cfg.name}] reusing finished run from {out_dir}")
        return found
    ckpt, hist_path, partial = _paths(out_dir, ckpt_dir, cfg.name)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    partial.parent.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    model = UNet3D(features=cfg.features)
    init_hash = state_hash(model)  # on the standard layers, so it is comparable with the v1 runs
    if cfg.fast_conv:
        make_fast(model)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # math.cos, not np.cos: the factor must be a Python float so optimiser/scheduler state is safely loadable
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.5 * (1 + math.cos(math.pi * it / cfg.iterations)))
    history = {"iteration": [], "train_loss": [], "val_loss": [], "val_dice": [], "lr": [],
               "patch_has_tumour": [], "elapsed_s": []}
    start_it, elapsed_before = 0, 0.0
    if partial.exists():
        state = torch.load(partial, map_location="cpu", weights_only=True)
        if state["config"] == asdict(cfg):
            model.load_state_dict(state["state_dict"])
            opt.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"])
            start_it, history, elapsed_before = state["iteration"], state["history"], state["elapsed_s"]
            log(f"[{cfg.name}] resuming from iteration {start_it}")

    ds = PatchDatasetV2(cache_dir, train_ids, cfg.patch, cfg.iterations * cfg.batch_size, cfg.sampler, cfg.fg_prob,
                        seed=cfg.seed, train_augment=True, p_spatial=cfg.p_spatial, p_gamma=cfg.p_gamma,
                        p_noise=cfg.p_noise)
    items = Subset(ds, range(start_it * cfg.batch_size, len(ds)))  # resume exactly where the stream stopped
    loader = DataLoader(items, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                        persistent_workers=cfg.num_workers > 0, prefetch_factor=4 if cfg.num_workers else None)
    vsub = val_subset(val_ids, cfg.val_cases)
    log(f"[{cfg.name}] {count_parameters(model):,} parameters, init hash {init_hash}, device {device}, "
        f"fast conv {cfg.fast_conv}, patch {cfg.patch}^3 x {cfg.batch_size}, {cfg.iterations:,} iterations, "
        f"validation on {len(vsub)} cases every {cfg.eval_every:,}")

    window_loss, window_has = [], []
    t0 = t_prog = time.time()
    it_prog = start_it
    model.train()
    for it, (x, y) in enumerate(loader, start=start_it + 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = dice_bce_loss(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        window_loss.append(loss.item())
        window_has.extend(y.flatten(1).amax(1).cpu().numpy().tolist())

        if it % 500 == 0 and it % cfg.eval_every:
            rate = (time.time() - t_prog) / max(it - it_prog, 1)
            log(f"[{cfg.name}] it {it:6d}  train {np.mean(window_loss[-500:]):.4f}  {rate:.3f} s/it  "
                f"ETA {(cfg.iterations - it) * rate / 3600:.1f} h")
            t_prog, it_prog = time.time(), it
        if it % cfg.eval_every == 0 or it == cfg.iterations:
            val_loss, val_dice = validate(model, cache_dir, vsub, cfg, device)
            model.train()
            elapsed = elapsed_before + time.time() - t0
            history["iteration"].append(it)
            history["train_loss"].append(float(np.mean(window_loss)))
            history["val_loss"].append(val_loss)
            history["val_dice"].append(val_dice)
            history["lr"].append(float(sched.get_last_lr()[0]))
            history["patch_has_tumour"].append(float(np.mean(window_has)))
            history["elapsed_s"].append(elapsed)
            log(f"[{cfg.name}] it {it:6d}  train {history['train_loss'][-1]:.4f}  val({len(vsub)}) loss {val_loss:.4f}  "
                f"Dice {val_dice:.4f}  tumour patches {history['patch_has_tumour'][-1]:.2f}  {elapsed / 3600:.2f} h")
            window_loss, window_has = [], []
            tmp = partial.with_suffix(".tmp")
            torch.save(_cpu({"config": asdict(cfg), "state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                             "scheduler": sched.state_dict(), "iteration": it, "history": history,
                             "elapsed_s": elapsed}), tmp)
            tmp.replace(partial)  # atomic: an interrupted save never corrupts the last checkpoint
            t_prog, it_prog = time.time(), it

    full_loss, full_dice = validate(model, cache_dir, val_ids, cfg, device)
    elapsed = elapsed_before + time.time() - t0
    history.update({"full_val_loss": full_loss, "full_val_dice": full_dice, "full_val_cases": len(val_ids),
                    "val_subset": vsub, "runtime_min": elapsed / 60, "init_hash": init_hash, "device": str(device)})
    log(f"[{cfg.name}] done: all {len(val_ids)} validation cases, loss {full_loss:.4f}, Dice {full_dice:.4f}, "
        f"{elapsed / 3600:.2f} h")
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"config": asdict(cfg), "state_dict": state_dict}, ckpt)
    hist_path.write_text(json.dumps(history, indent=1))
    partial.unlink(missing_ok=True)
    plain = UNet3D(features=cfg.features)
    plain.load_state_dict(state_dict)
    return plain.to(device), history


COMMON_V2 = dict(patch=96, batch_size=2, iterations=40_000, lr=1e-3, weight_decay=1e-4, seed=2440,
                 features=(16, 32, 64, 128, 256), eval_every=5_000, val_overlap=0.5)

EXPERIMENTS_V2 = {
    "uniform96": TrainConfigV2(name="uniform96", sampler="uniform", fg_prob=0.0, **COMMON_V2),
    "tumour96": TrainConfigV2(name="tumour96", sampler="tumour", fg_prob=0.5, **COMMON_V2),
}


if __name__ == "__main__":
    import pandas as pd

    from .experiments import CACHE_DIR, SPLIT_FILE

    split = pd.read_csv(SPLIT_FILE)
    train_ids = split.loc[split.split == "train", "case_id"].tolist()
    val_ids = split.loc[split.split == "validation", "case_id"].tolist()
    for name in sys.argv[1:]:
        train_v2(EXPERIMENTS_V2[name], CACHE_DIR, train_ids, val_ids, log=lambda s: print(s, flush=True))
