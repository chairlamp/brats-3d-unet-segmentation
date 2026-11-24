"""Patch sampling from cached, brain-cropped volumes, plus training-only augmentation."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def augment(x, y, rng):
    """Paired, training-only augmentation of an image patch [C, D, H, W] and mask [D, H, W].

    1. Left-right flip (axis D = NIfTI i, the left-right axis) with probability 0.5. The
       same flip is applied to image and mask so they stay aligned. Brains are roughly
       mirror-symmetric and tumours occur in either hemisphere, so a flipped case is a
       plausible case. Anterior-posterior or superior-inferior flips are not, and are not used.
    2. Per-channel intensity scale in [0.9, 1.1] and shift in [-0.1, 0.1] (in units of the
       per-case standard deviation), applied to each channel's non-zero voxels only, so the
       background stays 0 (the same convention as the normalisation).
       This mimics scanner-to-scanner contrast variation that z-scoring does not remove.
    The label is never interpolated, so its values stay binary.
    """
    if rng.random() < 0.5:
        x, y = x[:, ::-1], y[::-1]
    scale = rng.uniform(0.9, 1.1, size=(x.shape[0], 1, 1, 1)).astype(np.float32)
    shift = rng.uniform(-0.1, 0.1, size=(x.shape[0], 1, 1, 1)).astype(np.float32)
    x = np.where(x != 0, x * scale + shift, 0).astype(np.float32)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


class PatchDataset(Dataset):
    """Random 3D patches with binary whole-tumour masks.

    Item idx is generated from its own RNG seeded with (seed, idx), so the stream of
    patches is identical regardless of the number of DataLoader workers. Both samplers
    consume the RNG in the same order, so with the same seed they visit the same case
    sequence and draw the same augmentations; only the patch location rule differs:

    sampler="uniform": the patch start is uniform over all valid positions in the
        brain-cropped volume.
    sampler="tumour":  with probability fg_prob the patch is centred on a random tumour
        voxel (clipped to stay inside the volume); otherwise it is placed uniformly.

    Returns image float32 [C, D, H, W] and mask float32 [1, D, H, W].
    """

    def __init__(self, cache_dir, case_ids, patch, n_samples, sampler="uniform", fg_prob=0.5,
                 seed=0, train_augment=False):
        assert sampler in ("uniform", "tumour")
        self.cache_dir = Path(cache_dir)
        self.case_ids = list(case_ids)
        self.patch = np.array((patch,) * 3 if np.isscalar(patch) else patch)
        self.n_samples = n_samples
        self.sampler = sampler
        self.fg_prob = fg_prob
        self.seed = seed
        self.train_augment = train_augment
        self._cache = {}

    def __len__(self):
        return self.n_samples

    def _arrays(self, case_id):
        # Memory-mapped, so each worker only reads the bytes of the patches it cuts.
        if case_id not in self._cache:
            d = self.cache_dir
            self._cache[case_id] = (np.load(d / f"{case_id}_img.npy", mmap_mode="r"),
                                    np.load(d / f"{case_id}_lbl.npy", mmap_mode="r"),
                                    np.load(d / f"{case_id}_fg.npy"))
        return self._cache[case_id]

    def location(self, idx):
        """(case_id, patch start, centred_on_tumour) for item idx, without reading any voxels."""
        rng = np.random.default_rng([self.seed, idx])
        case_id = self.case_ids[rng.integers(len(self.case_ids))]
        _, lbl, fg = self._arrays(case_id)
        shape = np.array(lbl.shape)
        max_start = np.maximum(shape - self.patch, 0)
        uniform_start = np.array([rng.integers(0, m + 1) for m in max_start])
        u = rng.random()
        fg_voxel = fg[rng.integers(len(fg))] if len(fg) else None
        centred = self.sampler == "tumour" and u < self.fg_prob and fg_voxel is not None
        start = np.clip(fg_voxel - self.patch // 2, 0, max_start) if centred else uniform_start
        return case_id, start, centred, rng

    def __getitem__(self, idx):
        case_id, start, _, rng = self.location(idx)
        img, lbl, _ = self._arrays(case_id)
        sl = tuple(slice(s, s + p) for s, p in zip(start, self.patch))
        x = np.asarray(img[(slice(None),) + sl], dtype=np.float32)
        y = np.asarray(lbl[sl]) > 0
        pad = [(0, p - s) for p, s in zip(self.patch, y.shape)]
        if any(p[1] for p in pad):
            x = np.pad(x, [(0, 0)] + pad)
            y = np.pad(y, pad)
        if self.train_augment:
            x, y = augment(x, y, rng)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y[None].astype(np.float32))


def tumour_fraction(dataset, n=2000):
    """Fraction of the first n patches that contain at least one tumour voxel, plus the mean tumour fraction per patch."""
    has, frac = [], []
    for idx in range(min(n, len(dataset))):
        case_id, start, _, _ = dataset.location(idx)
        _, lbl, _ = dataset._arrays(case_id)
        y = np.asarray(lbl[tuple(slice(s, s + p) for s, p in zip(start, dataset.patch))]) > 0
        has.append(y.any())
        frac.append(y.sum() / np.prod(dataset.patch))
    return float(np.mean(has)), float(np.mean(frac))
