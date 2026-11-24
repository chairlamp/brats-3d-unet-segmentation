"""Full-volume prediction, postprocessing and per-case evaluation.

Predictions are made on the brain-cropped grid. Everything outside the brain bounding box
is background in both the reference (checked during preprocessing) and the prediction
(never predicted), so metrics on the cropped grid equal metrics on the full 240x240x155 grid.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .data import load_cached
from .inference import sliding_window_probs
from .metrics import case_metrics, voxel_volume_ml
from .train import state_hash

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)  # face, edge and corner neighbours all connect


def map_dir(out_dir, model, patch, overlap, zero_channel=None, blend="gaussian", flip=False):
    """Cache folder for one model and one set of inference settings: out_dir/<key>.

    The key is a hash of the weights and of every setting that changes the maps, so a cached
    map is only reused by an identical run. Changing the model, patch, overlap, blending, flip
    or zeroed channel therefore recomputes instead of silently loading old predictions.
    """
    settings = json.dumps({"patch": patch, "overlap": overlap, "zero_channel": zero_channel, "blend": blend,
                           "flip": flip}, sort_keys=True)
    return Path(out_dir) / hashlib.sha256((state_hash(model) + settings).encode()).hexdigest()[:10]


def predict_probs(model, cache_dir, case_ids, patch, overlap, device, out_dir=None, zero_channel=None,
                  blend="gaussian", flip=False, keep=True):
    """Sliding-window probability maps for each case, cached as float16 .npy in map_dir(out_dir, ...).

    zero_channel: index of an MRI channel to replace with zeros at test time (bonus stress test).
    flip: predict the left-right mirrored volume and mirror the map back (test-time augmentation).
    Maps are float16-rounded whether computed or loaded, so a rerun reproduces the metrics exactly.
    keep=False only fills the cache and returns no maps (saves memory).
    Returns ({case_id: probs}, {case_id: inference seconds}). Timings are saved next to the
    cached maps, so a rerun that loads them still reports the original inference time.
    """
    out_dir = map_dir(out_dir, model, patch, overlap, zero_channel, blend, flip) if out_dir else None
    timing_path = out_dir / "timing.json" if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    saved_seconds = json.loads(timing_path.read_text()) if timing_path and timing_path.exists() else {}
    probs, seconds = {}, {}
    for cid in case_ids:
        path = out_dir / f"{cid}.npy" if out_dir else None
        if path is not None and path.exists():
            seconds[cid] = saved_seconds.get(cid, float("nan"))
            if keep:
                probs[cid] = np.load(path).astype(np.float32)
            continue
        img, _ = load_cached(cache_dir, cid)
        img = np.asarray(img, dtype=np.float32)
        if zero_channel is not None:
            img = img.copy()
            img[zero_channel] = 0
        if flip:
            img = np.ascontiguousarray(img[:, ::-1])  # axis D is left-right (see sampling.augment)
        t = time.perf_counter()
        p = sliding_window_probs(model, img, patch, overlap, device, blend=blend)
        seconds[cid] = time.perf_counter() - t
        p16 = np.ascontiguousarray(p[::-1] if flip else p).astype(np.float16)
        if path is not None:
            np.save(path, p16)
            saved_seconds[cid] = seconds[cid]
            timing_path.write_text(json.dumps(saved_seconds))
        if keep:
            probs[cid] = p16.astype(np.float32)
    return probs, seconds


def remove_small_components(mask, min_ml, spacing):
    """Delete 26-connected components smaller than min_ml millilitres."""
    lab, n = ndimage.label(mask, structure=CONNECTIVITY_26)
    if n == 0:
        return mask
    sizes_ml = np.bincount(lab.ravel())[1:] * voxel_volume_ml(spacing)
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes_ml >= min_ml
    return keep[lab]


def component_sizes_ml(mask, spacing):
    lab, n = ndimage.label(mask, structure=CONNECTIVITY_26)
    return np.sort(np.bincount(lab.ravel())[1:] * voxel_volume_ml(spacing))[::-1] if n else np.array([])


def evaluate(probs, cache_dir, meta, threshold=0.5, min_component_ml=None):
    """Per-case metrics table. meta maps case_id -> preprocessing metadata (for spacing)."""
    rows = []
    for cid, p in probs.items():
        _, lbl = load_cached(cache_dir, cid)
        ref = np.asarray(lbl) > 0
        spacing = meta[cid]["spacing"]
        pred = p >= threshold
        if min_component_ml is not None:
            pred = remove_small_components(pred, min_component_ml, spacing)
        rows.append({"case_id": cid, **case_metrics(pred, ref, spacing)})
    return pd.DataFrame(rows).set_index("case_id")


def median_hd95_empty_worst(df):
    """Median HD95 with empty predictions counted as infinitely far (the worst case).

    HD95 is undefined (NaN) for an empty prediction; dropping those cases would report the
    median over only the cases the model still found, which flatters a model that misses many.
    """
    return float(np.median(df["hd95_mm"].fillna(np.inf)))


def summarise(df, cols=("dice", "iou", "sensitivity", "hd95_mm")):
    """Mean, SD and median per metric (NaN HD95 from empty predictions excluded) and empty-prediction count."""
    out = df[list(cols)].agg(["mean", "std", "median"]).T
    out["n"] = df[list(cols)].notna().sum()
    out.attrs["empty_predictions"] = int(df["empty_pred"].sum())
    return out
