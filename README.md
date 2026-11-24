# ENG2440 Assignment 2 — Whole-tumour segmentation in multimodal 3D MRI

Binary whole-tumour segmentation of four-channel brain MRI (FLAIR, T1w, contrast-enhanced T1, T2w) from the Medical Segmentation Decathlon Task01 BrainTumour dataset. A lightweight 3D U-Net is trained on 64³ patches with uniform and with tumour-aware patch sampling under otherwise identical conditions, reconstructed over whole volumes with overlapping sliding-window inference, and evaluated on a held-out test split with Dice, IoU, sensitivity, HD95 (mm) and tumour volume (mL).

Everything is in one executed notebook, `25011094_ENG2440_A2.ipynb`; the supporting code is in `a2/`.

Headline result on the 73-case test split (pipeline of notebook Sections 5–6: uniform sampling, 64³ patches, 50 % overlap, components < 0.5 mL removed): Dice 0.889 ± 0.065 (median 0.901), IoU 0.806, sensitivity 0.887, HD95 9.4 ± 10.7 mm (median 5.4 mm), median absolute volume error 6.6 mL, no empty predictions. 29 of the test cases share their scan with a case in train or validation (the dataset contains 116 near-duplicate pairs under different IDs; notebook Section 2.1). They score higher, but equally whether or not the twin was trained on, so this is case mix rather than memorisation (Section 6.5).

Beyond the brief (notebook Section 10, report Appendix C): retraining with 96³ patches for 8× more iterations and stronger augmentation, with the pipeline selected on validation only (tumour96 + left–right flip averaging, threshold 0.5), gives test Dice 0.911 ± 0.055 (median 0.927), better in 62 of 73 cases, HD95 8.0 mm (median 4.1 mm) and median absolute volume error 5.4 mL. The test split had already been evaluated once with the first pipeline; the v2 pipeline was frozen on validation and evaluated once.

## Requirements

- Python 3.12
- Apple silicon (MPS), a CUDA GPU, or CPU only. Training was run on an Apple M4 Max; on CPU it would be several times slower.
- About 35 GB of disk: 7.6 GB archive, 4.4 GB extracted training images and labels, 13 GB preprocessing cache, about 7.5 GB of cached probability maps, 90 MB of model weights.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
```

`requirements.lock.txt` pins every package used (PyTorch 2.15.0, NumPy 2.5.4, SciPy 1.18.1, nibabel 5.4.2, pandas 3.1.0, matplotlib 3.11.2 and their dependencies). `requirements.txt` lists only the direct dependencies.

## Data (not included)

1. Download `Task01_BrainTumour.tar` (7.6 GB) from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) or the MONAI mirror: `https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar`.
2. Extract it next to the notebook. Only `imagesTr/`, `labelsTr/` and `dataset.json` are used; the unlabelled `imagesTs/` is not needed:
   ```bash
   tar -xf Task01_BrainTumour.tar --exclude 'Task01_BrainTumour/imagesTs*'
   ```
3. `assignment2_split.csv` (from Moodle) is included unchanged.

```text
ENG2440_Assignment2/
  25011094_ENG2440_A2.ipynb
  assignment2_split.csv
  Task01_BrainTumour/
    imagesTr/        # 484 four-channel NIfTI volumes
    labelsTr/        # 484 label volumes
    dataset.json
  a2/                # supporting code
  cache/             # created by the notebook
  models/            # created by training
  predictions/       # created by the notebook
  results/           # per-case metric tables, created by the notebook
  report/figures/    # figures used in the report, created by the notebook
```

If the data are stored elsewhere, change only `DATA_ROOT` in the first code cell.

## Running

Exact run order used for the submitted outputs:

1. **Train both models** (optional as a separate step; the notebook trains them itself if `models/` is empty):
   ```bash
   python -m a2.train uniform tumour
   ```
   This trains the uniform-sampling run and then the tumour-aware run, about 2.5 hours (70 + 79 minutes) in total on an M4 Max, and writes `models/{uniform,tumour}.pt` and their training histories.
2. **Train the two v2 models** for notebook Section 10 (optional in the same way):
   ```bash
   python -m a2.train_v2 uniform96 tumour96
   ```
   96³ patches × 2 for 40,000 iterations each, about 9 hours in total on an M4 Max (271 + 266 minutes). It writes `models/{uniform96,tumour96}.pt` and their histories, plus a checkpoint every 5,000 iterations in `cache/checkpoints/`; rerunning the same command after an interruption resumes from the last checkpoint. Training uses the exact, faster convolution layers of `a2/fastconv.py`; the saved weights load into the plain `UNet3D`. For a long run on a laptop, prevent sleep (for example `caffeinate -is` on macOS) and keep it on mains power.
3. **Run the notebook** top to bottom with this folder as the working directory (kernel: `.venv`), or headlessly:
   ```bash
   python -c "import nbformat, nbclient; p = '25011094_ENG2440_A2.ipynb'; nb = nbformat.read(p, 4); nbclient.NotebookClient(nb, timeout=None).execute(); nbformat.write(nb, p)"
   ```

What happens on the first run:

- Section 2 runs the brief's setup check and verifies the split.
- Section 3 preprocesses all 484 cases once into `cache/` (about 1 minute with 8 worker processes).
- Section 4 reuses the trained models if their configuration in `a2/experiments.py` matches exactly; otherwise it trains them (about 70 minutes each).
- Sections 5–9 run sliding-window inference on the validation and test cases and cache the probability maps in `predictions/` (about 15 minutes on an M4 Max; the near-duplicate check in Section 2.1 adds about 1 minute in total).
- Section 10 reuses the v2 models in the same way (otherwise it trains them, about 4.5 hours each), selects the v2 pipeline on validation and evaluates it once on test (about 9 minutes).

The trained weights (`models/uniform.pt`, `models/tumour.pt`, `models/uniform96.pt`, `models/tumour96.pt`, about 22 MB each) are included, so steps 1 and 2 can be skipped and the notebook reproduces the reported results without retraining. Later runs reuse the cache, the models and the probability maps and finish in a few minutes. Each set of maps is stored under `predictions/<name>/<key>/`, where the key is a hash of the model weights and every inference setting (patch, overlap, blending, flip, zeroed channel). Changing a setting therefore recomputes the maps instead of silently reusing old ones. Delete `predictions/` to recompute everything, or `models/*.pt` to retrain.

## Where each part of the brief is answered

| Brief section | Notebook sections | Code |
|---|---|---|
| 1. Define the task and inspect the volumes | 1.1–1.3 | `a2/data.py`, `a2/viz.py` |
| 2. Case-level split | 2 | |
| 3. Preprocess MRI and labels safely | 3.1–3.4 | `a2/data.py`, `a2/sampling.py` (`augment`) |
| 4. Patch-based volumetric training | 4.1–4.5 | `a2/sampling.py`, `a2/model.py`, `a2/train.py`, `a2/experiments.py` |
| 5. Controlled sampling comparison | 5, 5.1, 5.2 | `a2/evaluate.py` |
| 6. Reconstruct and evaluate full volumes | 6.1–6.4 | `a2/inference.py`, `a2/metrics.py`, `a2/evaluate.py` |
| 7. Three-plane error analysis and limitations | 7.1–7.3 | `a2/viz.py` |
| Bonus: missing-modality stress test | 8 | |
| Bonus: sliding-window seam test | 9 | |
| Beyond the brief: improved v2 model | 10 | `a2/fastconv.py`, `a2/sampling_v2.py`, `a2/train_v2.py`, `a2/final_v2.py` |

## Reproducibility

- **Seed.** `SEED = 2440` for all four runs. It fixes the weight initialisation (all runs start from identical weights; the notebook checks the hash), the sequence of training cases, the patch positions and the augmentation draws. Every training item is generated from its own RNG seeded with `(seed, item index)`, so the stream does not depend on the number of DataLoader workers.
- **Split.** `assignment2_split.csv` is used unchanged (338 train, 73 validation, 73 test cases). The test split is used only in Sections 6–9, after the sampler, threshold, overlap and postprocessing rule were fixed on validation, and once more at the end of Section 10. The v2 pipeline was selected on validation with a rule fixed in `a2/final_v2.py` before any v2 result was computed; no v2 choice used the test split.
- **Hardware and runtime.** Apple M4 Max (36 GB), PyTorch MPS backend. Training: 69.7 min (uniform) and 78.5 min (tumour-aware) for 5,000 iterations each, including ten full validation passes; v2: 270.8 min (uniform96) and 266.1 min (tumour96) for 40,000 iterations each. Test inference: 1.0 s per case at 50 % overlap (1.1 s in an earlier run); v2 pipeline 1.05 s per case. First full notebook run after training: about 18 minutes (14 minutes when the v2 maps are already cached); later runs with cached predictions: a few minutes.
- **Determinism.** Apple's MPS backend is not bit-for-bit deterministic, so retraining can change results slightly. The outputs in the submitted notebook come from the saved runs in `models/`.

## Not included

The MSD archive and extracted volumes, `cache/` and `predictions/` are not included, following the brief: there are no MRI volumes, masks, patches or dataset copies. Everything else is included: the model weights, training histories, training log, per-case metric tables (`results/`), the report source and compiled PDF (`report/`) and the course split file.
