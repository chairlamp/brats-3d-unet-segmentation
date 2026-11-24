"""Patch-based training with periodic full-volume validation.

Run from the assignment folder to train the experiments defined in a2/experiments.py:
    python -m a2.train uniform tumour
The notebook calls the same `train` function, which reuses a saved run when its config matches.
"""

import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import load_cached
from .inference import sliding_window_probs
from .model import UNet3D, count_parameters
from .sampling import PatchDataset


@dataclass(frozen=True)
class TrainConfig:
    name: str
    sampler: str               # "uniform" or "tumour"
    fg_prob: float = 0.0       # tumour sampler only: probability a patch is centred on tumour
    patch: int = 96
    batch_size: int = 2
    iterations: int = 4000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 2440
    features: tuple = (16, 32, 64, 128, 256)
    eval_every: int = 500
    val_overlap: float = 0.5
    amp: bool = False


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def state_hash(model):
    """Short hash of the weights, used to show that both samplers start from identical initialisation."""
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def soft_dice(probs, target, eps=1.0):
    """Soft Dice over the whole batch at once, so patches without tumour do not produce 0/0."""
    inter = (probs * target).sum()
    return (2 * inter + eps) / (probs.sum() + target.sum() + eps)


def dice_bce_loss(logits, target):
    """Binary cross-entropy (per-voxel calibration) + (1 - soft Dice) (overlap, insensitive to class imbalance)."""
    return F.binary_cross_entropy_with_logits(logits, target) + 1 - soft_dice(torch.sigmoid(logits), target)


def full_volume_loss(probs, target):
    """The training loss evaluated on blended full-volume probabilities instead of logits."""
    p = torch.from_numpy(probs).clamp(1e-6, 1 - 1e-6)
    t = torch.from_numpy(target.astype(np.float32))
    return float(F.binary_cross_entropy(p, t) + 1 - soft_dice(p, t))


def validate(model, cache_dir, val_ids, cfg, device):
    """Sliding-window inference on every validation case; mean full-volume loss and Dice at threshold 0.5."""
    losses, dices = [], []
    for cid in val_ids:
        img, lbl = load_cached(cache_dir, cid)
        probs = sliding_window_probs(model, np.asarray(img, dtype=np.float32), cfg.patch, cfg.val_overlap, device)
        ref = np.asarray(lbl) > 0
        pred = probs >= 0.5
        losses.append(full_volume_loss(probs, ref))
        denom = pred.sum() + ref.sum()
        dices.append(1.0 if denom == 0 else 2 * np.count_nonzero(pred & ref) / denom)
    return float(np.mean(losses)), float(np.mean(dices))


def _paths(out_dir, name):
    out_dir = Path(out_dir)
    return out_dir / f"{name}.pt", out_dir / f"{name}_history.json"


def load_run(cfg, out_dir, device):
    """Saved model and history if a run with exactly this config exists, else None."""
    ckpt, hist = _paths(out_dir, cfg.name)
    if not ckpt.exists():
        return None
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    if saved["config"] != asdict(cfg):
        return None
    model = UNet3D(features=cfg.features)
    model.load_state_dict(saved["state_dict"])
    return model.to(device), json.loads(hist.read_text())


def train(cfg, cache_dir, train_ids, val_ids, out_dir="models", device=None, num_workers=6, log=print):
    device = device or pick_device()
    found = load_run(cfg, out_dir, device)
    if found is not None:
        log(f"[{cfg.name}] reusing saved run from {out_dir}")
        return found
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    model = UNet3D(features=cfg.features)
    init_hash = state_hash(model)
    model.to(device)
    ds = PatchDataset(cache_dir, train_ids, cfg.patch, cfg.iterations * cfg.batch_size, cfg.sampler,
                      cfg.fg_prob, seed=cfg.seed, train_augment=True)
    # Item order is already random (each index has its own seeded RNG), so no shuffling here.
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers,
                        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers else None)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.5 * (1 + np.cos(np.pi * it / cfg.iterations)))
    log(f"[{cfg.name}] {count_parameters(model):,} parameters, init hash {init_hash}, device {device}")

    history = {"iteration": [], "train_loss": [], "val_loss": [], "val_dice": [], "lr": [],
               "patch_has_tumour": [], "elapsed_s": []}
    window_loss, window_has = [], []
    t0 = time.time()
    model.train()
    for it, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp):
            logits = model(x)
        loss = dice_bce_loss(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        window_loss.append(loss.item())
        window_has.extend(y.flatten(1).amax(1).cpu().numpy().tolist())

        if it % cfg.eval_every == 0 or it == cfg.iterations:
            val_loss, val_dice = validate(model, cache_dir, val_ids, cfg, device)
            model.train()
            history["iteration"].append(it)
            history["train_loss"].append(float(np.mean(window_loss)))
            history["val_loss"].append(val_loss)
            history["val_dice"].append(val_dice)
            history["lr"].append(sched.get_last_lr()[0])
            history["patch_has_tumour"].append(float(np.mean(window_has)))
            history["elapsed_s"].append(time.time() - t0)
            log(f"[{cfg.name}] it {it:5d}  train {history['train_loss'][-1]:.4f}  val loss {val_loss:.4f}  "
                f"val Dice {val_dice:.4f}  tumour patches {history['patch_has_tumour'][-1]:.2f}  "
                f"{(time.time() - t0) / 60:.1f} min")
            window_loss, window_has = [], []
            Path(out_dir, f"{cfg.name}_history.partial.json").write_text(json.dumps(history))

    history["runtime_min"] = (time.time() - t0) / 60
    history["init_hash"] = init_hash
    history["device"] = str(device)
    ckpt, hist = _paths(out_dir, cfg.name)
    torch.save({"config": asdict(cfg), "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, ckpt)
    hist.write_text(json.dumps(history, indent=1))
    return model, history


if __name__ == "__main__":
    import pandas as pd

    from .experiments import CACHE_DIR, EXPERIMENTS, SPLIT_FILE

    split = pd.read_csv(SPLIT_FILE)
    train_ids = split.loc[split.split == "train", "case_id"].tolist()
    val_ids = split.loc[split.split == "validation", "case_id"].tolist()
    for name in sys.argv[1:]:
        train(EXPERIMENTS[name], CACHE_DIR, train_ids, val_ids, log=lambda s: print(s, flush=True))
