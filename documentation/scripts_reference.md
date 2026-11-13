# Script and API reference

[Documentation home](README.md) · [Top-level README](../README.md) · [v2 README](../v2/README.md)

All paths are relative to `v2/`. Run scripts from `v2/` with the virtual environment active. The order below is the
pipeline order; see the [reproducibility guide](../REPRODUCIBILITY.md) for the full sequence.

## Scripts
| Script | Arguments | What it does | Writes |
|---|---|---|---|
| [`scripts/build_splits.py`](../v2/scripts/build_splits.py) | — | Downloads the dataset (kagglehub), hashes every image, finds candidate pairs, builds duplicate groups and the leakage-aware split. | `results/review/hashes_and_groups.csv`, `duplicate_pairs.csv`, `candidate_pairs_hamming_le10.csv.gz`, `{{train,val,test,excluded}}_files.csv`, `dedup_summary.json` |
| [`src/btmri/train.py`](../v2/src/btmri/train.py) | `--run clean|naive` (default `clean`), `--epochs 12`, `--bs 32`, `--lr 5e-4`, `--seed 42` | Trains EfficientNet-B0. Run as `cd src && python -m btmri.train --run clean`. | `results/runs/<run>/best.pt`, `history.csv`, `val_logits.csv`, `test_logits.csv` |
| [`scripts/make_predictions.py`](../v2/scripts/make_predictions.py) | — | Fits the temperature and threshold τ on validation logits and writes the per-image test predictions (the single source of truth). | `results/predictions/{{clean,naive}}_test_predictions.csv`, `calibration.json` |
| [`scripts/make_report.py`](../v2/scripts/make_report.py) | — | Computes every metric (with bootstrap CIs), the leakage-by-class table and the looser-rule sensitivity check; draws evaluation figures. | `results/metrics.json`, `results/review/test_predictions.csv`, `results/figures/{{confusion_matrix,roc_curves,reliability,training_curves_*}}.png`, `paper/tables/*.tex`, `docs/stats.json` |
| [`scripts/audit_augmented.py`](../v2/scripts/audit_augmented.py) | — | Matches every `*-aug-*` file against non-augmented images under 8 flips/rotations and compares accuracy. | `results/review/augmented_meningioma_matches.csv`, `augmented_summary.json` |
| [`scripts/make_dedup_figures.py`](../v2/scripts/make_dedup_figures.py) | — | Draws example duplicate pairs and per-class leakage counts. | `results/figures/dedup_examples.png`, `dedup_per_class.png` |
| [`scripts/export_onnx.py`](../v2/scripts/export_onnx.py) | — | Exports the model with logits + CAM outputs to ONNX and verifies it against PyTorch. | `models/brain_tumor_effnetb0.onnx` and `.pt`, `docs/model/brain_tumor_effnetb0.onnx`, `results/onnx_verification.json` |
| [`scripts/run_xai.py`](../v2/scripts/run_xai.py) | optional positional: number of torch threads (default 8) | Runs the XAI study (faithfulness, foreground share, agreement, sanity check) and draws the XAI figures. | `results/xai/*.csv`, `results/figures/xai_*.png` |
| [`scripts/run_lime.py`](../v2/scripts/run_lime.py) | optional positional: number of torch threads (default 8) | Runs LIME on the same 100 images and superpixels as KernelSHAP, recomputes the other five maps, and writes LIME faithfulness, sanity check, 6 × 6 agreement, target-class check and seed check. | `results/xai/lime_*.csv`, `method_agreement_spearman_with_lime.csv`, `target_class_sensitivity.csv`, `lime_summary.json`, `results/figures/xai_lime_examples.png` |
| [`scripts/time_lime_browser.py`](../v2/scripts/time_lime_browser.py) | URL of the served demo | Times the 200 forward passes LIME would need in headless Chrome with ONNX Runtime Web (one WASM thread). | `results/xai/lime_browser_timing.json` |
| [`scripts/plot_xai_summary.py`](../v2/scripts/plot_xai_summary.py) | — | Redraws the XAI summary figures from the saved CSVs. | `results/figures/xai_{{sanity_check,faithfulness,agreement}}.png` |
| [`scripts/xai_summary_md.py`](../v2/scripts/xai_summary_md.py) | — | Writes the XAI summary text used by the README. | `results/xai/summary.md` |
| [`scripts/make_presets.py`](../v2/scripts/make_presets.py) | — | Selects the eight demo presets (no candidate pair at Hamming ≤ 10, no `-aug-` files, seed 42), copies the files byte-for-byte and precomputes Grad-CAM++, IG, KernelSHAP and LIME maps. | `docs/presets.json`, `docs/samples/*`, `results/review/preset_pool.json` |
| [`scripts/check_samples.py`](../v2/scripts/check_samples.py) | — | Runs the shipped ONNX model on each preset and checks it against the prediction file (each probability within 0.01, same uncertain flag); exits non-zero on failure. | console report |
| [`scripts/make_paper_numbers.py`](../v2/scripts/make_paper_numbers.py) | — | Writes every number used in the paper as a LaTeX macro, plus the XAI table. | `paper/numbers.tex`, `paper/tables/xai.tex` |
| [`scripts/make_v1_v2.py`](../v2/scripts/make_v1_v2.py) | — | Builds the v1 vs v2 table from `../v1/results/` and the current results; updates the note in the demo page. | `results/v1_vs_v2.md`, `v1_vs_v2.json`, `paper/tables/v1v2.tex`, `docs/index.html` (between `<!--V1V2-->` markers) |
| [`scripts/fill_readme.py`](../v2/scripts/fill_readme.py) | — | Fills `v2/README.md` from `scripts/README_template.md` and the result files. | `README.md` |
| [`scripts/build_notebooks.py`](../v2/scripts/build_notebooks.py) | — | Generates the five notebooks (executed afterwards with `jupyter nbconvert --execute`). | `notebooks/*.ipynb` |
| [`scripts/build_documentation.py`](../v2/scripts/build_documentation.py) | — | Generates this documentation set from the result files. | `../documentation/*.md` |

## Library modules (`v2/src/btmri/`)
| Module | Main functions and classes |
|---|---|
| [`data.py`](../v2/src/btmri/data.py) | `seed_everything(seed)`, `download_dataset()`, `list_images(root)`, `load_gray_square(path, size)` (greyscale, zero-pad, bilinear resize → uint8), `build_cache(recs, size, cache_file)` |
| [`dedup.py`](../v2/src/btmri/dedup.py) | `file_md5(path)`, `phash64(path)`, `hamming_matrix(a, b)`, `duplicate_groups(hashes, md5s, threshold)` |
| [`splits.py`](../v2/src/btmri/splits.py) | `hash_table(root)`, `candidate_pairs(df, max_hamming=10)` (adds Hamming, correlation and the strict/linked flags), `assign_groups(df, pairs, col)`, `make_splits(df, pairs, seed=42)` |
| [`model.py`](../v2/src/btmri/model.py) | `build_model(pretrained=True, num_classes=4)`, `to_input(gray_uint8)` (→ normalised 3-channel float), `ExportWrapper` (logits + CAM) |
| [`train.py`](../v2/src/btmri/train.py) | `load_split_table(run, seed)`, `predict(model, X)`, `main()` (command-line entry point) |
| [`calibrate.py`](../v2/src/btmri/calibrate.py) | `fit_temperature(logits, y)`, `softmax(logits, t)`, `norm_entropy(p)`, `ece(p, y, bins=15)`, `choose_threshold(p_val, y_val, target_acc=0.99)` |
| [`xai.py`](../v2/src/btmri/xai.py) | `GreyModel`, `gradcam_family(gm, x, target, plus)`, `integrated_gradients`, `occlusion`, `superpixels`, `kernel_shap`, `explain(gm, x, target, method)`, `cascading_randomisation`, `rank_corr`, `blur`, `deletion_insertion`, `brain_mask`, `attention_summary` |

Rule constants are defined at the top of [`splits.py`](../v2/src/btmri/splits.py) (strict, v1 and v2 thresholds).

## Tests (repository root)
| Command | What it checks |
|---|---|
| `pytest -q tests` | `test_core.py`: Hamming matrix vs. Python popcount, union-find grouping, temperature and threshold fitting, no duplicate group crossing train/val/test; `test_samples.py`: runs `check_samples.py` |
| `node tests/test_logic.js` | the browser logic in `docs/logic.js` (softmax, entropy, CAM, rationale) |
| `python tests/check_samples_browser.py [URL]` | opens the demo in headless Chrome (Playwright), runs every preset and checks the probabilities and flags; default URL `http://localhost:8000/` |
