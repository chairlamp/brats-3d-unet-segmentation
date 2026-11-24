"""Overlap, boundary and volume metrics for binary 3D masks.

All functions take boolean arrays on the same voxel grid. Distances and volumes use the
voxel spacing in millimetres, so results are in physical units rather than voxel counts.
"""

import numpy as np
from scipy import ndimage


def confusion_counts(pred, ref):
    pred, ref = pred.astype(bool), ref.astype(bool)
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))
    return tp, fp, fn


def dice(pred, ref):
    tp, fp, fn = confusion_counts(pred, ref)
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else 2 * tp / denom


def iou(pred, ref):
    tp, fp, fn = confusion_counts(pred, ref)
    denom = tp + fp + fn
    return 1.0 if denom == 0 else tp / denom


def sensitivity(pred, ref):
    tp, _, fn = confusion_counts(pred, ref)
    return float("nan") if tp + fn == 0 else tp / (tp + fn)


def voxel_volume_ml(spacing):
    """Volume of one voxel in millilitres (1 mL = 1000 mm^3)."""
    return float(np.prod(spacing)) / 1000.0


def volume_ml(mask, spacing):
    return np.count_nonzero(mask) * voxel_volume_ml(spacing)


def _surface(mask):
    """Boundary voxels: foreground voxels with at least one 6-connected background neighbour."""
    eroded = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1), border_value=0)
    return mask & ~eroded


def hd95(pred, ref, spacing):
    """Symmetric 95th-percentile Hausdorff distance in mm.

    Computes the 95th percentile of surface-to-surface distances in each direction
    (prediction -> reference and reference -> prediction) and returns the larger one,
    the same convention as MONAI's compute_hausdorff_distance(percentile=95).
    Returns NaN when either mask is empty, because a boundary distance is undefined.
    """
    pred, ref = pred.astype(bool), ref.astype(bool)
    if not pred.any() or not ref.any():
        return float("nan")
    # Crop to the joint bounding box (plus one voxel) so the distance transforms stay small.
    coords = np.argwhere(pred | ref)
    lo = np.maximum(coords.min(0) - 1, 0)
    hi = np.minimum(coords.max(0) + 2, pred.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    ps, rs = _surface(pred[box]), _surface(ref[box])
    dist_to_ref = ndimage.distance_transform_edt(~rs, sampling=spacing)
    dist_to_pred = ndimage.distance_transform_edt(~ps, sampling=spacing)
    return float(max(np.percentile(dist_to_ref[ps], 95), np.percentile(dist_to_pred[rs], 95)))


def case_metrics(pred, ref, spacing):
    """All per-case numbers used in the report, as one dict."""
    tp, fp, fn = confusion_counts(pred, ref)
    vv = voxel_volume_ml(spacing)
    ref_ml, pred_ml = (tp + fn) * vv, (tp + fp) * vv
    return {
        "dice": dice(pred, ref),
        "iou": iou(pred, ref),
        "sensitivity": sensitivity(pred, ref),
        "hd95_mm": hd95(pred, ref, spacing),
        "ref_ml": ref_ml,
        "pred_ml": pred_ml,
        "fp_ml": fp * vv,
        "fn_ml": fn * vv,
        "abs_vol_err_ml": abs(pred_ml - ref_ml),
        "pct_vol_err": 100.0 * (pred_ml - ref_ml) / ref_ml if ref_ml > 0 else float("nan"),
        "empty_pred": tp + fp == 0,
    }
