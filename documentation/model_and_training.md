# Model and training

[Documentation home](README.md) · [Top-level README](../README.md) · [v2 README](../v2/README.md)

Code: [`v2/src/btmri/model.py`](../v2/src/btmri/model.py), [`v2/src/btmri/train.py`](../v2/src/btmri/train.py),
[`v2/src/btmri/data.py`](../v2/src/btmri/data.py); notebook [`02_training_efficientnet_b0.ipynb`](../v2/notebooks/02_training_efficientnet_b0.ipynb).

## Pre-processing (identical in Python and in the browser)
1. Convert to greyscale (`PIL.Image.convert("L")`, ITU-R 601 luma).
2. Zero-pad to a square, keeping the aspect ratio.
3. Resize to 224×224 with bilinear filtering.
4. Replicate to three channels and normalise with ImageNet mean/std.

Pre-processed images are cached as one uint8 array (`cache/all_224.npy`, git-ignored).

## Architecture
| Item | Value |
|---|---|
| Backbone | EfficientNet-B0, `timm` model `efficientnet_b0.ra_in1k`, ImageNet-pretrained |
| Parameters | about 4.0 M |
| Head | global average pooling → dropout 0.2 → linear layer with 4 outputs |
| Final feature map | 7 × 7 × 1280 (used for Grad-CAM) |
| Export wrapper | `ExportWrapper` returns the logits and the four class activation maps (7 × 7) for the browser |

## Training set-up
| Hyper-parameter | Value |
|---|---|
| Optimiser | AdamW, learning rate 5e-4, weight decay 1e-4 |
| Schedule | One-cycle (`OneCycleLR`, `pct_start` 0.1) over all steps |
| Batch size / epochs | 32 / 12 |
| Loss | cross-entropy with inverse-frequency class weights `n_train / (4 · n_class)` |
| Class weights (v2 leakage-aware split) | glioma 0.818, meningioma 0.858, no tumour 2.042, pituitary 0.892 |
| Augmentation | `RandomResizedCrop` (scale 0.8–1.0, ratio 0.9–1.1), horizontal flip, `RandomAffine` (±10°, 5% translation), `ColorJitter` (brightness and contrast 0.2) |
| Checkpoint | best validation macro-F1 (leakage-aware: epoch 11; naive: epoch 8) |
| Seed | 42 for Python, NumPy and PyTorch, and for the batch order |
| Arguments | `python -m btmri.train --run clean|naive [--epochs 12] [--bs 32] [--lr 5e-4] [--seed 42]` (run from `v2/src`) |

**Naive comparison run** (`--run naive`): the same model and settings trained on the original Training folder with a
random, image-level 15% validation split per class (seed 42); evaluated on the same Testing images.

## Hardware and run times
Both v2 runs were trained on an 8-core Intel Xeon CPU without a GPU (PyTorch CPU build, 8 threads).
| Run | Train / val images | Median s per epoch | Total training time | Best epoch | Validation accuracy (best checkpoint) |
|---|---|---|---|---|---|
| Leakage-aware (`clean`) | 3,709 / 668 | 115 | 44 min | 11 | 98.7% |
| Naive | 4,760 / 840 | 136 | 61 min | 8 | 99.0% |

Several epochs took much longer (up to 1982 s), which inflates the totals; the median is the better guide to the compute needed.

Training curves: [`training_curves_clean.png`](../v2/results/figures/training_curves_clean.png),
[`training_curves_naive.png`](../v2/results/figures/training_curves_naive.png); per-epoch logs in
`v2/results/runs/<run>/history.csv`.

## Determinism
Seeds are fixed, but CPU kernels and thread scheduling can still change the last digits of the weights between machines,
so a re-run should land close to, not exactly on, the reported numbers. Only one seed was trained per configuration; the
confidence intervals do not include training variance ([limitations](limitations.md)).
