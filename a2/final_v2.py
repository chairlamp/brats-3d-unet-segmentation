"""Inference and validation-only selection for the v2 models (96^3 patches, a2/train_v2.py).

Component maps (sliding window with 96^3 windows, 50 % overlap, Gaussian blending, as in v1):
    U = uniform96, T = tumour96; Uf, Tf = the same model on the left-right mirrored volume,
    with the map mirrored back. A candidate averages some of these maps before thresholding.

Candidates, from simplest (level 1) to most complex (level 4):
    1  uniform96                     [U]             1  tumour96          [T]
    2  uniform96 + flip              [U, Uf]         2  tumour96 + flip   [T, Tf]
    3  uniform96 + tumour96          [U, T]
    4  uniform96 + tumour96 + flip   [U, Uf, T, Tf]
Thresholds: 0.5 (the v1 default), 0.4 and 0.6.

Selection rule, fixed before any v2 validation result was computed, and applied to the
validation cases only: among all candidates and thresholds (no postprocessing), find the
highest mean validation Dice. Every option within MARGIN = 0.002 of it counts as equivalent,
and among those the simplest wins: lowest level, then threshold 0.5 before 0.4 or 0.6, then the
higher mean Dice. The connected-component rule is then chosen for the selected pipeline with
the v1 criterion (lowest median validation HD95 without lowering mean Dice by more than 0.001).
"""

import numpy as np
import pandas as pd
import torch

from . import evaluate as ev
from .data import load_cached
from .metrics import case_metrics
from .model import UNet3D

PATCH, OVERLAP = 96, 0.5
CANDIDATES = {  # name: (simplicity level, component maps averaged)
    "uniform96": (1, ["U"]),
    "tumour96": (1, ["T"]),
    "uniform96 + flip": (2, ["U", "Uf"]),
    "tumour96 + flip": (2, ["T", "Tf"]),
    "uniform96 + tumour96": (3, ["U", "T"]),
    "uniform96 + tumour96 + flip": (4, ["U", "Uf", "T", "Tf"]),
}
THRESHOLDS = (0.5, 0.4, 0.6)
MARGIN = 0.002
COMPONENT_SIZES_ML = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)  # the same candidates as for v1


def load_model(path, device):
    """A finished v2 run (models/{name}.pt) as a plain UNet3D in eval mode."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = UNet3D(features=tuple(saved["config"]["features"]))
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()


def component_dirs(models, cache_dir, case_ids, device, out_root, tags=("U", "Uf", "T", "Tf")):
    """Compute (or reuse) the cached component maps; returns {tag: cache folder}.

    models: {"U": uniform96, "T": tumour96}. A tag ending in "f" is the flipped prediction.
    """
    dirs = {}
    for tag in tags:
        model, flip = models[tag[0]], tag.endswith("f")
        ev.predict_probs(model, cache_dir, case_ids, PATCH, OVERLAP, device, out_dir=out_root, flip=flip, keep=False)
        dirs[tag] = ev.map_dir(out_root, model, PATCH, OVERLAP, flip=flip)
    return dirs


def average(dirs, tags, case_id):
    """Mean of the component maps `tags` for one case (float32)."""
    return np.mean([np.load(dirs[t] / f"{case_id}.npy").astype(np.float32) for t in tags], axis=0)


def averaged_maps(dirs, tags, case_ids):
    return {c: average(dirs, tags, c) for c in case_ids}


def sweep(dirs, cache_dir, case_ids, meta, candidates=CANDIDATES, thresholds=THRESHOLDS):
    """Per-case metrics (no postprocessing) for every candidate and threshold."""
    rows = []
    for cid in case_ids:
        _, lbl = load_cached(cache_dir, cid)
        ref, spacing = np.asarray(lbl) > 0, meta[cid]["spacing"]
        maps = {t: np.load(d / f"{cid}.npy").astype(np.float32) for t, d in dirs.items()}
        for name, (_, tags) in candidates.items():
            p = np.mean([maps[t] for t in tags], axis=0)
            for th in thresholds:
                rows.append({"case_id": cid, "candidate": name, "threshold": th,
                             **case_metrics(p >= th, ref, spacing)})
    return pd.DataFrame(rows)


def select(sweep_df, candidates=CANDIDATES, thresholds=THRESHOLDS, margin=MARGIN):
    """Apply the selection rule of the module docstring. Returns (candidate, threshold, ranked table)."""
    table = sweep_df.groupby(["candidate", "threshold"]).agg(
        mean_dice=("dice", "mean"), median_dice=("dice", "median"), sensitivity=("sensitivity", "mean"),
        fp_ml=("fp_ml", "mean"), median_hd95=("hd95_mm", "median"), mean_hd95=("hd95_mm", "mean"),
        empty=("empty_pred", "sum")).reset_index()
    table["level"] = table.candidate.map(lambda c: candidates[c][0])
    table["threshold_rank"] = table.threshold.map(lambda t: 0 if t == 0.5 else 1)
    best = table.mean_dice.max()
    table["equivalent"] = table.mean_dice >= best - margin
    ranked = table.sort_values(["equivalent", "level", "threshold_rank", "mean_dice"],
                               ascending=[False, True, True, False]).reset_index(drop=True)
    chosen = ranked.iloc[0]
    return chosen.candidate, float(chosen.threshold), ranked.drop(columns="threshold_rank")


def choose_component_rule(probs, cache_dir, meta, threshold):
    """v1 criterion (notebook Section 5.2) on validation maps. Returns (min_component_ml or None, table)."""
    rows = []
    for T in (None,) + COMPONENT_SIZES_ML:
        r = ev.evaluate(probs, cache_dir, meta, threshold=threshold, min_component_ml=T)
        rows.append({"T_ml": "none" if T is None else str(T), "dice mean": r.dice.mean(),
                     "dice median": r.dice.median(), "hd95_mm mean": r.hd95_mm.mean(),
                     "hd95_mm median": r.hd95_mm.median()})
    table = pd.DataFrame(rows).set_index("T_ml")
    base, rest = table.iloc[0], table.iloc[1:]
    ok = rest[(rest["dice mean"] >= base["dice mean"] - 0.001) & (rest["hd95_mm median"] < base["hd95_mm median"])]
    return (float(ok["hd95_mm median"].idxmin()) if len(ok) else None), table
