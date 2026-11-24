"""Plotting helpers: anatomically oriented three-plane views and overlay colours.

Colours were checked with a colour-vision-deficiency validator: the two sampler colours
on a light background, and the three overlay colours (all pairs) on black MRI background.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

SAMPLER_COLOURS = {"uniform": "#2a78d6", "tumour": "#eb6834"}
ERROR_COLOURS = {"TP": "#199e70", "FP": "#d95926", "FN": "#3987e5"}
LABEL_COLOURS = {1: "#3987e5", 2: "#d95926", 3: "#199e70"}
INK, INK_2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

# Display convention for an array whose axes run towards R, A, S (see to_ras). Slices are
# shown with imshow(origin="lower"), so array index 0 is at the bottom-left of each panel.
#   plane     fixed axis   horizontal (left -> right)   vertical (bottom -> top)
#   axial     S (k)        R axis: L -> R               A axis: P -> A
#   coronal   A (j)        R axis: L -> R               S axis: I -> S
#   sagittal  R (i)        A axis: P -> A               S axis: I -> S
# Patient right is on the viewer's right (neurological convention).
PLANES = {"axial": 2, "coronal": 1, "sagittal": 0}
_EDGE_LABELS = {"axial": ("L", "R", "P", "A"), "coronal": ("L", "R", "I", "S"), "sagittal": ("P", "A", "I", "S")}


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
        "legend.frameon": False, "axes.titlecolor": INK, "text.color": INK,
    })


def to_ras(vol, affine):
    """Reorder and flip voxel axes so they run towards R, A, S, using only the affine.

    Works for [D, H, W] or channel-first [C, D, H, W]. Returns the array and the new
    spacing. For this dataset the affine is the identity (already RAS), so this is a
    no-op, but the plotting code does not rely on that.
    """
    tf = nib.orientations.ornt_transform(nib.orientations.io_orientation(affine),
                                         nib.orientations.axcodes2ornt(("R", "A", "S")))
    spacing = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(0))
    new_spacing = tuple(float(spacing[int(src)]) for src in np.argsort(tf[:, 0]))
    if vol.ndim == 4:
        return np.stack([nib.orientations.apply_orientation(v, tf) for v in vol]), new_spacing
    return nib.orientations.apply_orientation(vol, tf), new_spacing


def plane_slice(vol, plane, index):
    """2D slice of a RAS-ordered [D, H, W] array, transposed for imshow(origin='lower')."""
    return np.take(vol, index, axis=PLANES[plane]).T


def plane_aspect(spacing, plane):
    """imshow aspect = vertical voxel size / horizontal voxel size, so millimetres are square."""
    h, v = [a for a in range(3) if a != PLANES[plane]]
    return spacing[v] / spacing[h]


def show_plane(ax, vol, plane, index, spacing, overlay=None, cmap="gray", vmin=None, vmax=None,
               title=None, edge_labels=True, black_background=True):
    """One anatomically oriented slice. Exact zeros (outside the skull-stripped brain) are drawn
    black, so after z-scoring the background does not appear as mid-grey."""
    sl = plane_slice(vol, plane, index)
    cm = plt.get_cmap(cmap).copy()
    if black_background:
        sl = np.ma.masked_equal(sl, 0)
        cm.set_bad("black")
    ax.imshow(sl, cmap=cm, origin="lower", vmin=vmin, vmax=vmax,
              aspect=plane_aspect(spacing, plane), interpolation="nearest")
    if overlay is not None:
        ax.imshow(np.transpose(np.take(overlay, index, axis=PLANES[plane]), (1, 0, 2)), origin="lower",
                  aspect=plane_aspect(spacing, plane), interpolation="nearest")
    if edge_labels:
        left, right, bottom, top = _EDGE_LABELS[plane]
        kw = dict(color="white", fontsize=8, fontweight="bold", transform=ax.transAxes)
        ax.text(0.02, 0.5, left, ha="left", va="center", **kw)
        ax.text(0.98, 0.5, right, ha="right", va="center", **kw)
        ax.text(0.5, 0.02, bottom, ha="center", va="bottom", **kw)
        ax.text(0.5, 0.98, top, ha="center", va="top", **kw)
    if title:
        ax.set_title(title)
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def _rgba(masks_and_colours, shape, alpha):
    out = np.zeros(shape + (4,), dtype=np.float32)
    for mask, colour in masks_and_colours:
        out[mask] = to_rgba(colour, alpha)
    return out


def label_overlay(label, alpha=0.6):
    """RGBA volume colouring the original BraTS sub-region labels 1, 2, 3."""
    return _rgba([(label == k, c) for k, c in LABEL_COLOURS.items()], label.shape, alpha)


def error_overlay(pred, ref, alpha=0.75):
    """RGBA volume: true positive, false positive and false negative voxels in distinct colours."""
    pred, ref = pred.astype(bool), ref.astype(bool)
    return _rgba([(pred & ref, ERROR_COLOURS["TP"]), (pred & ~ref, ERROR_COLOURS["FP"]),
                  (~pred & ref, ERROR_COLOURS["FN"])], pred.shape, alpha)


def legend_patches(colours, names=None):
    names = names or {k: str(k) for k in colours}
    return [Patch(facecolor=c, label=names[k]) for k, c in colours.items()]


def display_range(vol, mask=None):
    """1st-99th percentile of brain voxels, for a consistent greyscale window."""
    vals = vol[mask] if mask is not None else vol[vol != 0]
    return np.percentile(vals, [1, 99])


def centre_of_mass(mask):
    return tuple(int(round(c)) for c in np.argwhere(mask).mean(0))
