"""The training runs used in the report. The notebook and `python -m a2.train` both read these,
so a run trained from the command line is reused by the notebook unchanged."""

from .train import TrainConfig

SPLIT_FILE = "assignment2_split.csv"
CACHE_DIR = "cache"

# Everything except the sampler is shared, so the comparison is controlled: same architecture,
# initialisation (same seed), loss, optimiser, learning-rate schedule, patch size, batch size,
# iteration budget, case sequence and augmentation draws.
COMMON = dict(patch=64, batch_size=4, iterations=5000, lr=1e-3, weight_decay=1e-4, seed=2440,
              features=(16, 32, 64, 128, 256), eval_every=500, val_overlap=0.5)

EXPERIMENTS = {
    "uniform": TrainConfig(name="uniform", sampler="uniform", fg_prob=0.0, **COMMON),
    "tumour": TrainConfig(name="tumour", sampler="tumour", fg_prob=0.5, **COMMON),
}
