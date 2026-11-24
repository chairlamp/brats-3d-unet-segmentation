"""Loading, geometry checks, preprocessing and caching for MSD Task01 BrainTumour.

Array convention used everywhere: images are channel-first [C, D, H, W] where D, H, W are
the NIfTI voxel axes i, j, k in file order. They are not reordered, so a prediction on the
same grid can be written back with the original affine unchanged.
"""

import json
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

CHANNELS = ["FLAIR", "T1w", "T1gd", "T2w"]  # order of the 4th NIfTI axis, from dataset.json
LABEL_NAMES = {0: "background", 1: "edema", 2: "non-enhancing tumour", 3: "enhancing tumour"}
TARGET_SPACING = (1.0, 1.0, 1.0)  # mm; the common grid every case must be on before training


def load_split(split_file, data_root):
    df = pd.read_csv(split_file)
    df["image_path"] = [Path(data_root) / p for p in df["image_relpath"]]
    df["label_path"] = [Path(data_root) / p for p in df["label_relpath"]]
    return df


def header_geometry(image_path, label_path):
    """Geometry read from the NIfTI headers only (no voxel data is decompressed)."""
    im, lb = nib.load(image_path), nib.load(label_path)
    return {
        "image_shape": tuple(int(s) for s in im.shape),
        "label_shape": tuple(int(s) for s in lb.shape),
        "spacing": tuple(float(z) for z in im.header.get_zooms()[:3]),
        "label_spacing": tuple(float(z) for z in lb.header.get_zooms()[:3]),
        "axcodes": "".join(nib.aff2axcodes(im.affine)),
        "label_axcodes": "".join(nib.aff2axcodes(lb.affine)),
        "affines_match": bool(np.allclose(im.affine, lb.affine)),
        "image_dtype": str(im.header.get_data_dtype()),
        "affine": im.affine,
    }


def load_case(image_path, label_path):
    """Image as float32 [4, D, H, W], label as uint8 [D, H, W], plus affine and spacing."""
    im, lb = nib.load(image_path), nib.load(label_path)
    image = np.moveaxis(np.asarray(im.dataobj, dtype=np.float32), -1, 0)
    label = np.asarray(lb.dataobj).astype(np.uint8)
    return image, label, im.affine, tuple(float(z) for z in im.header.get_zooms()[:3])


def check_geometry(image, label, image_affine, label_affine):
    """The four channels share one 4D header by construction; the label must match that grid."""
    problems = []
    if image.shape[1:] != label.shape:
        problems.append(f"shape {image.shape[1:]} vs label {label.shape}")
    if not np.allclose(image_affine, label_affine):
        problems.append("image and label affines differ")
    return problems


def resample(image, label, spacing, target=TARGET_SPACING):
    """Resample to `target` spacing: linear (order 1) for MRI, nearest neighbour (order 0) for labels.

    Linear interpolation is fine for intensities but would create fractional, meaningless
    label values (e.g. 1.5 between edema and non-enhancing tumour); order 0 copies the
    nearest original label so the output contains only values that were in the input.
    """
    zoom = [s / t for s, t in zip(spacing, target)]
    image = np.stack([ndimage.zoom(ch, zoom, order=1) for ch in image])
    label = ndimage.zoom(label, zoom, order=0)
    return image, label


def brain_mask(image):
    """Brain = voxels that are non-zero in any channel. Skull-stripped BraTS volumes are 0 outside."""
    return (image != 0).any(axis=0)


def bounding_box(mask):
    coords = np.argwhere(mask)
    return coords.min(0), coords.max(0) + 1


def normalize_nonzero(image):
    """Z-score each channel using only that channel's non-zero (brain) voxels; zeros stay exactly 0.

    Including the zero background (about 83% of the volume) would pull the mean towards 0
    and inflate the standard deviation by an amount that depends on head size, not tissue.
    The mask is per channel because the channels' skull-stripped regions differ slightly
    (a one-voxel rim at the brain surface, and in some cases a smaller field of view in one
    channel). With a shared mask, a voxel that is 0 in one channel would become
    -mean/SD (about -3), an artificial value; here it stays 0, the channel's brain mean.
    """
    out = np.zeros_like(image, dtype=np.float32)
    stats = []
    for c in range(image.shape[0]):
        m = image[c] != 0
        vals = image[c][m]
        mean, std = float(vals.mean()), float(vals.std())
        out[c][m] = (vals - mean) / max(std, 1e-8)
        stats.append((mean, std))
    return out, stats


def preprocess_case(case_id, image_path, label_path, cache_dir, n_fg_coords=20000):
    """Load one case, verify geometry, normalise, crop to the brain and write it to the cache.

    Writes {case_id}_img.npy (float16 [4, d, h, w]), {case_id}_lbl.npy (uint8, original
    multi-class labels) and {case_id}_fg.npy (up to n_fg_coords random tumour voxel
    coordinates, used by the tumour-aware sampler). Returns a metadata dict.
    """
    cache_dir = Path(cache_dir)
    im, lb = nib.load(image_path), nib.load(label_path)
    image, label, affine, spacing = load_case(image_path, label_path)
    problems = check_geometry(image, label, im.affine, lb.affine)
    if problems:
        raise ValueError(f"{case_id}: {problems}")
    resampled = not np.allclose(spacing, TARGET_SPACING)
    if resampled:
        image, label = resample(image, label, spacing)

    mask = brain_mask(image)
    channel_masks = image != 0
    norm, raw_stats = normalize_nonzero(image)
    lo, hi = bounding_box(mask)
    tumour_outside_box = int(np.count_nonzero(label > 0)) - int(
        np.count_nonzero(label[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] > 0))
    box = (slice(None),) + tuple(slice(a, b) for a, b in zip(lo, hi))
    img_c, lbl_c = norm[box], label[box[1:]]

    fg = np.argwhere(lbl_c > 0).astype(np.int16)
    if len(fg) > n_fg_coords:
        rng = np.random.default_rng(zlib.crc32(case_id.encode()))
        fg = fg[rng.choice(len(fg), n_fg_coords, replace=False)]

    np.save(cache_dir / f"{case_id}_img.npy", img_c.astype(np.float16))
    np.save(cache_dir / f"{case_id}_lbl.npy", lbl_c)
    np.save(cache_dir / f"{case_id}_fg.npy", fg)

    # Statistics of the stored (float16) values over each channel's own brain voxels
    stored = img_c.astype(np.float16).astype(np.float32)
    cm = channel_masks[box]
    norm_mean = [float(stored[c][cm[c]].mean()) for c in range(4)]
    norm_std = [float(stored[c][cm[c]].std()) for c in range(4)]
    return {
        "case_id": case_id,
        "orig_shape": [int(s) for s in label.shape],
        "spacing": list(TARGET_SPACING if resampled else spacing),
        "resampled": resampled,
        "affine": np.asarray(affine).tolist(),
        "bbox_lo": [int(v) for v in lo],
        "bbox_hi": [int(v) for v in hi],
        "crop_shape": [int(s) for s in lbl_c.shape],
        "brain_voxels": int(mask.sum()),
        "channel_zero_in_brain": [int((mask & ~channel_masks[c]).sum()) for c in range(4)],
        "tumour_voxels": int(np.count_nonzero(label > 0)),
        "tumour_outside_box": tumour_outside_box,
        "label_values": sorted(int(v) for v in np.unique(label)),
        "raw_mean": [s[0] for s in raw_stats],
        "raw_std": [s[1] for s in raw_stats],
        "norm_mean": norm_mean,
        "norm_std": norm_std,
    }


def preprocess_all(split_df, cache_dir, workers=8):
    """Preprocess every case in parallel, skipping cases already in the cache. Returns a metadata DataFrame."""
    from concurrent.futures import ProcessPoolExecutor

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    todo = [r for r in split_df.itertuples() if r.case_id not in index]
    if todo:
        with ProcessPoolExecutor(workers) as pool:
            futures = [pool.submit(preprocess_case, r.case_id, r.image_path, r.label_path, cache_dir) for r in todo]
            for f in futures:
                meta = f.result()
                index[meta["case_id"]] = meta
        index_path.write_text(json.dumps(index))
    return pd.DataFrame([index[c] for c in split_df["case_id"]])


def scan_fingerprint(image_path, zoom=0.1):
    """Coarse, intensity-scale-invariant signature of a scan's tissue contrast.

    FLAIR and T2w are each z-scored within the brain (background left at 0, as in the
    preprocessing), downsampled ~10x (24 x 24 x 16 voxels) and z-scored again. Normalising
    within the brain first matters: on raw intensities the brain-versus-background contrast
    dominates and any two brains of similar shape correlate at ~0.99. Two cases built from
    the same acquisition give ~1 even if intensities were rescaled or their labels differ.
    """
    img = nib.load(image_path)
    parts = []
    for c in (0, 3):
        x = np.array(img.dataobj[..., c], dtype=np.float32)  # writable copy
        m = x != 0
        x[m] = (x[m] - x[m].mean()) / x[m].std()
        v = ndimage.zoom(x, zoom, order=1).ravel()
        parts.append((v - v.mean()) / v.std())
    return np.concatenate(parts)  # dot product / length = mean of the per-channel correlations


def near_duplicate_pairs(split_df, threshold=0.99, workers=8):
    """Pairs of cases whose scan fingerprints correlate above `threshold`, with the correlation."""
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers) as pool:
        fps = np.stack(list(pool.map(scan_fingerprint, split_df.image_path, chunksize=8)))
    corr = fps @ fps.T / fps.shape[1]
    np.fill_diagonal(corr, 0)
    ids = split_df.case_id.to_numpy()
    i, j = np.where(np.triu(corr) > threshold)
    pairs = pd.DataFrame({"case_a": ids[i], "case_b": ids[j], "r": corr[i, j]})
    next_best = np.sort(corr[np.triu_indices(len(ids), 1)])[-len(pairs) - 1]
    return pairs, float(next_best)


def load_cached(cache_dir, case_id, mmap=True):
    mode = "r" if mmap else None
    img = np.load(Path(cache_dir) / f"{case_id}_img.npy", mmap_mode=mode)
    lbl = np.load(Path(cache_dir) / f"{case_id}_lbl.npy", mmap_mode=mode)
    return img, lbl


def uncrop(cropped, meta, fill=0):
    """Paste a cropped [d, h, w] array back into the original grid using the stored bounding box."""
    full = np.full(meta["orig_shape"], fill, dtype=cropped.dtype)
    lo, hi = meta["bbox_lo"], meta["bbox_hi"]
    full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = cropped
    return full
