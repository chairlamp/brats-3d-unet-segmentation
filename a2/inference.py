"""Overlapping sliding-window inference over a whole preprocessed volume."""

import numpy as np
import torch


def window_starts(size, patch, stride):
    """Start indices along one axis. The last window is shifted back so it ends at the volume edge."""
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def gaussian_weight(patch, sigma_scale=0.125):
    """Separable 3D Gaussian centred on the patch, sigma = patch/8 per axis, max 1.

    Voxels near a patch border see less spatial context, so their predictions are
    down-weighted when overlapping windows are blended. The floor keeps every voxel's
    weight positive so a voxel covered by a single window is still normalised correctly.
    """
    axes = []
    for p in patch:
        x = np.arange(p) - (p - 1) / 2
        axes.append(np.exp(-0.5 * (x / (p * sigma_scale)) ** 2))
    w = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    w = w / w.max()
    return np.maximum(w, 1e-3).astype(np.float32)


@torch.no_grad()
def sliding_window_probs(model, image, patch, overlap, device, batch_size=4, blend="gaussian"):
    """Tumour probability for every voxel of a [C, D, H, W] volume.

    Windows of size `patch` are placed with stride round(patch * (1 - overlap)). Each
    window's sigmoid probabilities are multiplied by a blending weight and accumulated;
    the final probability is the weighted sum divided by the summed weights, so
    overlapping probabilities are combined before any threshold is applied.
    blend="gaussian" uses gaussian_weight; blend="mean" weights every voxel equally.
    """
    patch = (patch,) * 3 if np.isscalar(patch) else tuple(patch)
    C, *shape = image.shape
    # Pad any axis shorter than the patch, then crop the padding off the output.
    pads = [max(p - s, 0) for p, s in zip(patch, shape)]
    if any(pads):
        image = np.pad(image, [(0, 0)] + [(0, p) for p in pads])
    padded = image.shape[1:]
    strides = [max(1, int(round(p * (1 - overlap)))) for p in patch]
    grid = [window_starts(s, p, st) for s, p, st in zip(padded, patch, strides)]
    starts = [(a, b, c) for a in grid[0] for b in grid[1] for c in grid[2]]

    if blend == "gaussian":
        weight = torch.from_numpy(gaussian_weight(patch)).to(device)
    else:
        weight = torch.ones(patch, device=device)
    prob_sum = torch.zeros(padded, device=device)
    weight_sum = torch.zeros(padded, device=device)
    vol = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))

    model.eval()
    for i in range(0, len(starts), batch_size):
        chunk = starts[i:i + batch_size]
        batch = torch.stack([vol[:, a:a + patch[0], b:b + patch[1], c:c + patch[2]] for a, b, c in chunk])
        probs = torch.sigmoid(model(batch.to(device)).float())[:, 0]
        for (a, b, c), p in zip(chunk, probs):
            sl = (slice(a, a + patch[0]), slice(b, b + patch[1]), slice(c, c + patch[2]))
            prob_sum[sl] += p * weight
            weight_sum[sl] += weight
    out = (prob_sum / weight_sum).cpu().numpy()
    return out[:shape[0], :shape[1], :shape[2]]


def n_windows(shape, patch, overlap):
    patch = (patch,) * 3 if np.isscalar(patch) else tuple(patch)
    strides = [max(1, int(round(p * (1 - overlap)))) for p in patch]
    return int(np.prod([len(window_starts(max(s, p), p, st)) for s, p, st in zip(shape, patch, strides)]))
