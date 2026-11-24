"""Patch dataset with stronger training-only augmentation, for the longer v2 training runs.

Patch locations come from PatchDataset.location, unchanged, so the uniform and tumour-aware v2 runs
visit the same case sequence. Every random number of the augmentation is drawn in a fixed order
whether or not the transform is applied, so both runs also receive identical augmentation draws.

Transforms (training only; image and mask always share the same geometry):
1. Rotation (up to +/-15 deg about each axis) and isotropic scaling (x0.9-1.1) about the patch
   centre, with probability 0.25. Image channels: linear interpolation; mask: nearest neighbour,
   so it stays binary. The patch is resampled from a larger region of the cropped volume, so no
   artificial empty corners appear unless the region leaves the brain crop (background is 0 anyway).
   Small rotations and scalings mimic residual head pose and head-size variation; they are mild
   because every case is already registered to a common atlas.
2. Left-right flip with probability 0.5 (as in v1).
3. Per-channel intensity scale x0.9-1.1 and shift +/-0.1 SD on brain voxels (as in v1).
4. Per-channel gamma (0.7-1.5) with probability 0.3, on brain voxels, keeping their mean and SD:
   changes contrast non-linearly, like differences between scanners and sequence parameters.
5. Gaussian noise (SD up to 0.1 in z-score units) on brain voxels with probability 0.15.
Background voxels (exactly 0 after brain z-scoring) stay 0 under 3-5.
"""

import itertools

import numpy as np
import torch
from scipy import ndimage

from .sampling import PatchDataset


def rotation_matrix(angles):
    """3x3 rotation: about axis 0, then 1, then 2 (angles in radians)."""
    a, b, c = angles
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def gamma_keep_stats(v, gamma):
    """Gamma transform of a 1D array of brain intensities, rescaled to keep its mean and SD."""
    if v.size < 2:
        return v
    mean, sd, lo, hi = v.mean(), v.std(), v.min(), v.max()
    u = ((v - lo) / max(hi - lo, 1e-6)) ** gamma * (hi - lo) + lo
    return (u - u.mean()) / max(u.std(), 1e-6) * sd + mean


class PatchDatasetV2(PatchDataset):
    def __init__(self, *args, p_spatial=0.25, max_rot_deg=15.0, scale_range=(0.9, 1.1), p_gamma=0.3,
                 gamma_range=(0.7, 1.5), p_noise=0.15, max_noise_sd=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_spatial, self.max_rot = p_spatial, np.deg2rad(max_rot_deg)
        self.scale_range, self.p_gamma, self.gamma_range = scale_range, p_gamma, gamma_range
        self.p_noise, self.max_noise_sd = p_noise, max_noise_sd

    def _crop(self, img, lbl, start):
        sl = tuple(slice(s, s + p) for s, p in zip(start, self.patch))
        x = np.asarray(img[(slice(None),) + sl], dtype=np.float32)
        y = np.asarray(lbl[sl]) > 0
        pad = [(0, p - s) for p, s in zip(self.patch, y.shape)]
        if any(p[1] for p in pad):
            x, y = np.pad(x, [(0, 0)] + pad), np.pad(y, pad)
        return x, y

    def _spatial(self, img, lbl, start, angles, zoom):
        """Patch with the same centre as the plain crop, rotated and scaled about that centre."""
        patch = self.patch.astype(float)
        m = rotation_matrix(angles) / zoom            # output offset -> input offset
        out_centre = (patch - 1) / 2
        centre = start + out_centre                   # same centre as the unaugmented patch
        corners = np.array(list(itertools.product(*[(0.0, p - 1) for p in patch]))) - out_centre
        reach = np.abs(corners @ m.T).max(0)          # half-extent of input that the patch samples
        lo = np.floor(centre - reach).astype(int) - 1
        hi = np.ceil(centre + reach).astype(int) + 2
        shape = np.array(lbl.shape)
        src_lo, src_hi = np.maximum(lo, 0), np.minimum(hi, shape)
        src = tuple(slice(a, b) for a, b in zip(src_lo, src_hi))
        dst = tuple(slice(a - l, b - l) for a, b, l in zip(src_lo, src_hi, lo))
        region_img = np.zeros((img.shape[0],) + tuple(hi - lo), np.float32)
        region_lbl = np.zeros(tuple(hi - lo), np.float32)
        region_img[(slice(None),) + dst] = img[(slice(None),) + src]
        region_lbl[dst] = np.asarray(lbl[src]) > 0
        offset = centre - lo - m @ out_centre         # input position = m @ output index + offset
        kw = dict(matrix=m, offset=offset, output_shape=tuple(int(p) for p in patch), mode="constant", cval=0.0)
        x = np.stack([ndimage.affine_transform(region_img[c], order=1, **kw) for c in range(img.shape[0])])
        y = ndimage.affine_transform(region_lbl, order=0, **kw) > 0.5
        return x.astype(np.float32), y

    def __getitem__(self, idx):
        case_id, start, _, rng = self.location(idx)
        img, lbl, _ = self._arrays(case_id)
        n_ch = img.shape[0]
        # All draws happen here, in a fixed order, whether or not each transform is used
        spatial = rng.random() < self.p_spatial
        angles = rng.uniform(-self.max_rot, self.max_rot, size=3)
        zoom = rng.uniform(*self.scale_range)
        flip = rng.random() < 0.5
        scale = rng.uniform(0.9, 1.1, size=(n_ch, 1, 1, 1)).astype(np.float32)
        shift = rng.uniform(-0.1, 0.1, size=(n_ch, 1, 1, 1)).astype(np.float32)
        gamma_on = rng.random(n_ch) < self.p_gamma
        gammas = rng.uniform(*self.gamma_range, size=n_ch)
        noise_on = rng.random() < self.p_noise
        noise_sd = rng.uniform(0.0, self.max_noise_sd)
        noise_seed = int(rng.integers(2 ** 31))

        if self.train_augment and spatial:
            x, y = self._spatial(img, lbl, start, angles, zoom)
        else:
            x, y = self._crop(img, lbl, start)
        if self.train_augment:
            if flip:
                x, y = x[:, ::-1], y[::-1]
            brain = x != 0
            x = np.where(brain, x * scale + shift, 0).astype(np.float32)
            for c in np.flatnonzero(gamma_on):
                x[c][brain[c]] = gamma_keep_stats(x[c][brain[c]], gammas[c])
            if noise_on:
                noise = np.random.default_rng(noise_seed).standard_normal(x.shape).astype(np.float32)
                x = np.where(brain, x + noise_sd * noise, 0).astype(np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(y)[None].astype(np.float32)))
