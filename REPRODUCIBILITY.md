# Reproducibility guide

[Documentation home](documentation/README.md) · [Top-level README](README.md)

This guide reproduces the **v2** results from scratch. Every reported number is derived from the files in
`v2/results/`, so reproducing those files reproduces the README, the paper and the demo statistics.
For **v1**, check out the tag `v1.0` (`git checkout v1.0`) and follow its README; its saved outputs are in [`v1/`](v1/README.md).

## 1. Environment used for the reported results
| Item | Version |
|---|---|
| OS / hardware | Linux, 8-core Intel Xeon CPU, no GPU |
| Python | 3.13.5 |
| PyTorch / torchvision | 2.15.0+cpu / 0.29.2+cpu |
| timm | 1.0.31 |
| Captum | 0.9.0 |
| ONNX / ONNX Runtime | 1.23.2 / 1.31.1 |
| scikit-learn / scikit-image / SciPy | 1.9.1 / 0.26.0 / 1.18.1 |
| NumPy / pandas | 2.5.4 / 3.1.1 |
| ImageHash / Pillow | 4.3.2 / 13.0.0 |
| matplotlib / kagglehub | 3.11.2 / 1.0.3 |
| Node.js (demo logic test) | 24.22.1 |
| Tectonic (paper) | 0.15.0 |
| ONNX Runtime Web (demo) | 1.20.1 (loaded from jsDelivr) |

`v2/requirements.txt` gives minimum versions; for the closest match, install the versions above.

## 2. Set-up
```bash
git clone https://github.com/Rishabh-bgp/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification/v2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
The dataset (Kaggle *Brain Tumor MRI Dataset*, version 2) is downloaded automatically by `kagglehub` on first use; it is
public, so no API key is needed. It is not stored in the repository. Pre-processed arrays are cached in `cache/` at the
repository root (git-ignored).

## 3. Pipeline (run from `v2/`)
| Step | Command | Main outputs |
|---|---|---|
| 1. Duplicate audit and split | `python scripts/build_splits.py` | `results/review/hashes_and_groups.csv`, `dedup_summary.json`, split lists |
| 2. Train both models | `cd src && python -m btmri.train --run clean && python -m btmri.train --run naive && cd ..` | `results/runs/{clean,naive}/` |
| 3. Predictions and calibration | `python scripts/make_predictions.py` | `results/predictions/` |
| 4. Metrics, tables, figures | `python scripts/make_report.py` | `results/metrics.json`, `results/figures/`, `../docs/stats.json` |
| 5. Augmented-file audit | `python scripts/audit_augmented.py` | `results/review/augmented_*` |
| 6. Duplicate figures | `python scripts/make_dedup_figures.py` | `results/figures/dedup_*.png` |
| 7. ONNX export | `python scripts/export_onnx.py` | `models/`, `../docs/model/`, `results/onnx_verification.json` |
| 8. Explainability | `python scripts/run_xai.py [threads]` then `python scripts/plot_xai_summary.py && python scripts/xai_summary_md.py` | `results/xai/`, `results/figures/xai_*.png` |
| 9. Demo presets | `python scripts/make_presets.py && python scripts/check_samples.py` | `../docs/presets.json`, `../docs/samples/` |
| 10. Paper numbers and v1/v2 table | `python scripts/make_paper_numbers.py && python scripts/make_v1_v2.py` | `paper/numbers.tex`, `paper/tables/`, `results/v1_vs_v2.*` |
| 11. Paper | `cd paper && tectonic brain_tumor_mri_white_paper.tex && cd ..` | `paper/brain_tumor_mri_white_paper.pdf` |
| 12. README, notebooks, documentation | `python scripts/fill_readme.py && python scripts/build_notebooks.py && python scripts/build_documentation.py` | `README.md`, `notebooks/`, `../documentation/`, `../README.md` |
| 13. Tests (from the repository root) | `cd .. && pytest -q tests && node tests/test_logic.js` | — |

Training took a median of about 2 minutes per epoch (12 epochs per run) on the CPU above; see
[model and training](documentation/model_and_training.md#hardware-and-run-times). To regenerate only the reported numbers
from the committed checkpoints and logits, skip steps 1–2 (the split files and runs are committed) and start at step 3.

## 4. Expected results
With the committed checkpoint, steps 3 onward reproduce the committed files exactly, for example:
- leakage-aware accuracy 91.9% (1,470/1,600), macro-F1 0.919, temperature 1.909, threshold τ 0.69;
- naive accuracy 96.1%;
- 1,223 Training images removed, 3,709 / 668 train / validation images.

Retraining (step 2) on different hardware may change the last digits of the weights, and therefore the metrics slightly:
seeds are fixed (42), but CPU kernels are not bit-for-bit identical across machines and library versions. Only one seed
was trained, so expect differences of the order of the v1/v2 difference (about 1–2 points of accuracy) when retraining
with a different seed.

## 5. Checks
- `python scripts/check_samples.py`: the eight demo presets reproduce the prediction file in ONNX Runtime (each probability
  within 0.01, same uncertain flag).
- `python tests/check_samples_browser.py http://localhost:8000/` (from the root, with `cd docs && python3 -m http.server 8000`
  running and Playwright installed): the same check in a real browser.
- `results/onnx_verification.json`: PyTorch vs ONNX agreement.

## 6. Notebooks
The five notebooks in `v2/notebooks/` call the same scripts and modules. On Colab, the first cell clones the repository
and changes into `v2/`. Notebooks 2 and 4 reuse the committed checkpoint and XAI results unless `FORCE_TRAIN = True` /
`FORCE_XAI = True`. The committed notebooks were executed with `jupyter nbconvert --to notebook --execute --inplace`.
