<div align="center">

# Brain Tumour MRI Classification: Leakage-Aware, Calibrated and Explainable EfficientNet-B0

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x_CPU-EE4C2C?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-EfficientNet--B0-0F766E)
![Captum](https://img.shields.io/badge/XAI-Grad--CAM·IG·Occlusion·SHAP-14B8A6)
![ONNX Runtime Web](https://img.shields.io/badge/ONNX_Runtime-Web-005CED?logo=onnx&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-5_notebooks-F37626?logo=jupyter&logoColor=white)
[![Live demo](https://img.shields.io/badge/live_demo-NEURO%2F%2FSCAN-2DD4BF?logo=githubpages&logoColor=black)](https://rishabh-bgp.github.io/brain-tumor-mri-classification/)
[![White paper](https://img.shields.io/badge/white_paper-PDF-22D3EE)](paper/brain_tumor_mri_white_paper.pdf)
![License](https://img.shields.io/badge/license-MIT-blue)

**[▶ Open the live demo: NEURO//SCAN](https://rishabh-bgp.github.io/brain-tumor-mri-classification/)** · **[📄 Read the white paper (PDF)](paper/brain_tumor_mri_white_paper.pdf)**

</div>

> ⚠️ **Research and education project, not a medical device.** The model classifies single 2-D MRI slices from one
> public dataset. It has not been clinically validated, can be confidently wrong, and must not be used for diagnosis
> or treatment decisions.

**Computer vision in PyTorch** for four-way brain MRI classification (**glioma, meningioma, pituitary tumour, no tumour**)
on the Kaggle [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
The project puts evaluation hygiene first: a perceptual-hash audit finds **near-duplicate images shared by the official
Training and Testing folders**, the training data are de-leaked and split by duplicate group, an ImageNet-pretrained
**EfficientNet-B0** is fine-tuned on a CPU, calibrated with temperature scaling, explained with **five XAI methods**
that are sanity-checked and scored for faithfulness, and served as a static web app where the model runs **in your
browser**.

By **Rishabh Aryan**, M.Tech (Artificial Intelligence and Data Science), Department of Computer Science and Engineering,
Indian Institute of Information Technology Bhagalpur, Bihar, India.

---

## 🆕 v2 vs v1 (results side by side)

v1 is commit [`a591cfa`](https://github.com/Rishabh-bgp/brain-tumor-mri-classification/tree/v1.0) (tag `v1.0`); its metric
files are kept unchanged in [`results/v1/`](results/v1/). v2 (tag `v2.0`) applies the team review. Every value below is read
from the saved metric files of each version ([`scripts/make_v1_v2.py`](scripts/make_v1_v2.py)).

| Metric (leakage-aware model unless stated) | v1 (`a591cfa`, tag `v1.0`) | v2 (this version) |
|---|---|---|
| Duplicate link rule | MD5, or Ham ≤ 6 & r ≥ 0.95 | v1, or Ham ≤ 10 & r ≥ 0.98 |
| Linked pairs | 5,962 | 6,177 |
| Training-folder images removed | 1,198 | 1,223 |
| Train / validation images | 3,739 / 663 | 3,709 / 668 |
| Test accuracy (95% CI) | 93.4% (92.1%–94.5%) | 91.9% (90.5%–93.2%) |
| Macro-F1 | 0.933 | 0.919 |
| Macro ROC-AUC | 0.989 | 0.987 |
| ECE raw → calibrated | 0.054 → 0.036 | 0.070 → 0.052 |
| Temperature T / threshold τ | 1.84 / 0.67 | 1.91 / 0.69 |
| Uncertain rate | 3.8% | 3.5% |
| Accuracy confident / uncertain | 95.3% / 44.3% | 93.8% / 39.3% |
| Recall glioma | 84.0% | 84.3% |
| Recall meningioma | 89.8% | 84.0% |
| Recall notumor | 100.0% | 99.8% |
| Recall pituitary | 99.8% | 99.5% |
| One-per-group test accuracy | 93.1% | 91.9% |
| Naive model accuracy (same test) | 96.1% | 96.1% |
| Gap naive − leakage-aware | 2.8 pts | 4.3 pts |
| Meningiomas with strict dup.: naive / leakage-aware | 105/105 / 74/105 | 105/105 / 55/105 |
| Bootstrap resamples | 1,000 (200 for two subset rows) | 1,000 for every row |

**Why v2 is better than v1, in plain language**
- **Stricter duplicate rule.** A review found **19 Testing images** (18 no tumour, 1 glioma) whose copies, altered in
  contrast or sharpening, sat at pHash distance 8–10 and were still in our training/validation data. v2 also links pairs at
  Hamming ≤ 10 with thumbnail correlation ≥ 0.98: **+215 linked pairs** (71 across Training/Testing), and
  **1,223 Training images removed** instead of 1,198. The model was retrained with identical settings.
- **Honest uncertainty everywhere.** Every row of the subset table now uses **1,000 bootstrap resamples** (v1 used 200 for two rows).
- **Corrected claims and citations.** Glioma source (figshare only), the augmented-file conclusion (possible but not
  demonstrated), the Roberts et al. and Deepak/Swati descriptions, the abstract's accuracy claim (now cited), consistent
  rounding and reference style, and an uncited entry removed.
- **Stricter demo presets.** Only test images with no candidate pair at all at Hamming ≤ 10 and no augmented files.
- **Demo fixes.** Samples stay disabled until the model has loaded, non-images get a clear message, and in-browser
  occlusion now uses the evaluated setting (32-px patch, **stride 16**, 169 passes) instead of a stride-32 preview.
- **Robustness check.** A looser rule (Hamming 7–10, correlation ≥ 0.95) would link 84 more Testing images to our
  training data; all 84 are predicted correctly, and without them accuracy is 91.4% (n = 1,516), close to the 91.9% on all 1,600.

**Why some v2 scores are lower.** Accuracy goes from 93.4% to 91.9%. That is expected: v2 is the **more honest
estimate**, because the hidden near-copies found by the review are no longer in training (no Testing image now has a training or validation partner with correlation ≥ 0.98 within Hamming 10). Not all of the drop is due to the rule, though:
the training pool loses only 25 images (18 no tumour, 7 glioma, no meningioma) while the group-wise train/validation
assignment is re-drawn, yet meningioma recall moves from 89.8% to 84.0%. Part of the change is therefore run-to-run
variance of a single training seed, which the confidence intervals do not include.

---

## 🧠 Live demo: NEURO//SCAN

<p align="center"><img src="docs/screenshot.png" alt="NEURO//SCAN web app: MRI viewer with a Grad-CAM overlay, class probabilities and the Why this prediction panel" width="100%"></p>

A radiology-workstation-style page: drop an MRI slice (or click one of eight test-split samples labelled with their true
class), watch the scan line sweep while EfficientNet-B0 runs through ONNX Runtime Web (WebAssembly), and read:

- **animated, calibrated class probabilities** for glioma, meningioma, pituitary and no tumour;
- a **heat overlay with an opacity slider** and a method toggle: **Grad-CAM** (exact, computed from the ONNX feature map and classifier weights), **occlusion** (32-px patch, stride 16: 169 masked forward passes in the browser, the same setting as the paper), and for the samples precomputed **Grad-CAM++, Integrated Gradients and KernelSHAP**;
- a **"Why this prediction" panel** built from templates filled only with measured quantities: calibrated confidence, margin over the runner-up, an *uncertain — low confidence* flag, entropy, where the Grad-CAM mass falls, agreement between XAI methods (samples), and the class's real test precision/recall and most common confusion. It contains no medical reasoning.

Images never leave your device. Mobile layout: [screenshot](docs/screenshot_mobile.png).

## 📚 Notebooks (run on Google Colab)

| # | Notebook | What it does | Colab |
|---|---|---|---|
| 1 | [01_data_and_deduplication.ipynb](notebooks/01_data_and_deduplication.ipynb) | `kagglehub` download, inspection, MD5 + pHash + correlation duplicate audit, leakage-aware split | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rishabh-bgp/brain-tumor-mri-classification/blob/main/notebooks/01_data_and_deduplication.ipynb) |
| 2 | [02_training_efficientnet_b0.ipynb](notebooks/02_training_efficientnet_b0.ipynb) | EfficientNet-B0 fine-tuning, leakage-aware and naive runs, training curves | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rishabh-bgp/brain-tumor-mri-classification/blob/main/notebooks/02_training_efficientnet_b0.ipynb) |
| 3 | [03_evaluation_and_calibration.ipynb](notebooks/03_evaluation_and_calibration.ipynb) | Prediction file, temperature scaling, uncertainty threshold, metrics with bootstrap CIs, confusion matrix, ROC, leakage comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rishabh-bgp/brain-tumor-mri-classification/blob/main/notebooks/03_evaluation_and_calibration.ipynb) |
| 4 | [04_explainable_ai.ipynb](notebooks/04_explainable_ai.ipynb) | Grad-CAM, Grad-CAM++, Integrated Gradients, occlusion, KernelSHAP; randomisation sanity check; deletion/insertion AUC; per-class and error grids | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rishabh-bgp/brain-tumor-mri-classification/blob/main/notebooks/04_explainable_ai.ipynb) |
| 5 | [05_onnx_export_and_web_demo.ipynb](notebooks/05_onnx_export_and_web_demo.ipynb) | ONNX export (logits + CAM), PyTorch/ONNX agreement check | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rishabh-bgp/brain-tumor-mri-classification/blob/main/notebooks/05_onnx_export_and_web_demo.ipynb) |

Each notebook installs its dependencies on Colab, checks for a GPU, clones this repository for the `src/` package and
downloads the dataset with `kagglehub` (public dataset, no API key needed). The committed notebooks keep their executed
outputs. Notebooks 2 and 4 reuse the committed checkpoint and XAI results unless you set `FORCE_TRAIN` / `FORCE_XAI = True`.

## 🔍 Duplicate audit (the dataset's hidden leakage)

Version 2 of the dataset states that duplicates were removed and train/test overlap eliminated. At the **file** level that
is true: none of the 340 byte-identical files (7,013 unique MD5s among 7,200) crosses the folders. At the
**image** level it is not:

| Check | Result |
|---|---|
| Strict near-duplicate pairs (pHash Hamming ≤ 2 and 64×64 thumbnail correlation ≥ 0.98) | 3,484 pairs, **1,270 across Training/Testing** |
| Linked pairs (MD5 equal, or Hamming ≤ 6 and correlation ≥ 0.95, or Hamming ≤ 10 and correlation ≥ 0.98) | 6,177 pairs, 2,229 across |
| Testing images with a strict near-duplicate in Training | **441 / 1,600 (27.6%)**: no tumour 327, meningioma 105, glioma 9, pituitary 0 |
| Testing images sharing a linked group with a Training image | 630 / 1,600 (39.4%) |
| Images in groups carrying two different labels | 13 (the Testing folder itself has near-identical images with different labels, e.g. `Te-gl_68` / `Te-me_96` at Hamming 0, so a few test errors may be label noise) |

**Leakage-aware split:** every Training image linked to a Testing image (1,223, of which 854 are no-tumour) and every
label-conflict image is removed; the rest is split into **3,709 train / 668 validation** images *by duplicate group*.
The official 1,600 Testing images are kept intact, and no image linked to them under this rule is seen in training.
Thresholds were chosen by visual inspection. **v2 (after internal review):** the third clause (Hamming ≤ 10 with correlation
≥ 0.98) catches contrast- or sharpening-altered copies at Hamming 8–10 that v1 missed; it adds 215 linked pairs
(71 across Training/Testing) and raises the number of removed Training images from 1,198 to 1,223; the leakage-aware model was retrained on the rebuilt split.
pHash can still miss some near-copies. **Robustness check (no retraining):** a looser rule that also links Hamming 7–10 with
correlation ≥ 0.95 would link 84 more Testing images to our train/validation images; without them, test accuracy is
91.4% (n = 1,516) vs 91.9% on all 1,600. All hashes and candidate pairs are released for re-analysis.

<p align="center"><img src="results/figures/dedup_examples.png" alt="Strict near-duplicates shared by the Training and Testing folders" width="100%"></p>

## 📈 Results (from the saved prediction file)

EfficientNet-B0 (ImageNet-pretrained, 4.0 M parameters), 224 px greyscale, 12 epochs on an 8-core CPU, seed 42.
Every number below, the white paper's tables and the demo's statistics are generated by scripts from
`results/predictions/clean_test_predictions.csv`.

**Leakage-aware model, full Testing folder (n = 1,600):** accuracy **91.9%** (95% bootstrap CI 90.5%–93.2%; 1,000 image-level resamples, slightly optimistic because some test images are near-copies of each other),
macro-F1 **0.919**, macro one-vs-rest ROC-AUC **0.987**, ECE 0.070 → **0.052** after temperature scaling (T = 1.91).
One image per duplicate group (n = 1,342): accuracy 91.9%, macro-F1 0.912.

| Class | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Glioma | 0.968 | 0.843 | 0.901 | 0.968 |
| Meningioma | 0.941 | 0.840 | 0.888 | 0.983 |
| No tumour | 0.816 | 0.998 | 0.898 | 0.996 |
| Pituitary | 0.980 | 0.995 | 0.988 | 1.000 |
| **Macro** | | | **0.919** | **0.987** |

The dominant error is a tumour called *no tumour* (42 gliomas, 48 meningiomas): no-tumour recall is 99.8% but its
precision is 81.6%. Validation accuracy inside the training distribution is 98.7%, well above test accuracy, which
suggests a shift between the Training and Testing folders beyond the duplicates removed.

**Naive vs leakage-aware.** An identical model trained on the original Training folder (random image-level validation split) scores **96.1%** on the same Testing folder, 4.3 points higher. Part of the gap may come from more training data (4,760 vs 3,709 training images; 1,190 vs 454 no-tumour), but the per-class split shows memorisation: on the 105 Testing meningiomas that have a strict near-duplicate in Training the naive model is correct on **105/105**, the leakage-aware model on 55/105, while on the 295 meningiomas without one the two are close (288 vs 281 correct). 50 of the 68-image gap comes from these 105 meningiomas, which suggests that much of the gap reflects memorisation of near-copies; part of it also reflects the leakage-aware model losing all training examples of that sub-population, so its leak-free accuracy is unknown.

| Model | Full Testing | Testing with a strict dup. in Training (n = 441) | Testing with no linked dup. (n = 970) |
|---|---|---|---|
| Leakage-aware | 91.9% | 88.0% | 92.5% |
| Naive (original Training folder) | 96.1% | 99.5% | 94.0% |

**Uncertainty threshold.** A prediction is flagged *uncertain — low confidence* when its calibrated confidence is below
**τ = 0.69**, the smallest value in {0.50, 0.51, …, 0.99} at which retained validation predictions reach ≥ 99% accuracy
(validation coverage 98.7%). On Testing, 3.5% of predictions are flagged; accuracy is 93.8% on confident and
39.3% on flagged predictions. Normalised entropy is displayed but not thresholded.

<table>
  <tr>
    <td align="center" width="50%"><img src="results/figures/confusion_matrix.png" alt="Confusion matrix" width="100%"><br><sub>Confusion matrix (Testing)</sub></td>
    <td align="center" width="50%"><img src="results/figures/roc_curves.png" alt="ROC curves" width="100%"><br><sub>One-vs-rest ROC curves</sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="results/figures/reliability.png" alt="Reliability diagram" width="100%"><br><sub>Calibration before/after temperature scaling</sub></td>
    <td align="center" width="50%"><img src="results/figures/training_curves_clean.png" alt="Training curves" width="100%"><br><sub>Training curves (leakage-aware split)</sub></td>
  </tr>
</table>

## 🔬 Explainable AI

Five attribution methods for the predicted class: **Grad-CAM**, **Grad-CAM++**, **Integrated Gradients** (Captum, 32 steps,
black baseline), **occlusion** (32-px patch, stride 16) and **KernelSHAP** on ~64 SLIC superpixels (200 coalitions).
Evaluated on 100 test images (25 per class) with **deletion/insertion AUC** (Petsiuk et al., 2018) against a random
baseline, and on 12 images with the **cascading model-parameter randomisation test** (Adebayo et al., 2018).

| Method | Deletion AUC ↓ | Insertion AUC ↑ | Mass on head region | ρ after classifier randomised | ρ after full randomisation |
|---|---|---|---|---|---|
| Grad-CAM | 0.487 ± 0.355 | 0.841 ± 0.195 | 79.2% | 0.11 | -0.07 |
| Grad-CAM++ | 0.490 ± 0.353 | 0.842 ± 0.183 | 84.0% | 0.94 | 0.07 |
| Integrated Gradients | 0.283 ± 0.259 | 0.679 ± 0.252 | 97.8% | 0.94 | 0.84 |
| Occlusion | 0.466 ± 0.333 | 0.803 ± 0.226 | 82.7% | 0.14 | 0.03 |
| KernelSHAP (SLIC) | 0.400 ± 0.351 | 0.749 ± 0.254 | 76.5% | 0.24 | 0.21 |
| Random baseline | 0.283 ± 0.149 | 0.497 ± 0.366 | 57.9% | — | — |

**What the numbers say.**
- **Insertion** separates the methods from the random baseline (0.497): Grad-CAM 0.841, Grad-CAM++ 0.842, occlusion 0.803, KernelSHAP 0.749, Integrated Gradients 0.679.
- **Deletion** (black fill) does not: no method beats random (0.283); Integrated Gradients (0.283) comes closest. Scattered black pixels push slices out of distribution faster than deleting one compact region, so this metric favours pixel-scattered maps under a black baseline.
- **Sanity check:** randomising only the classifier drops the rank correlation to 0.11 (Grad-CAM), 0.14 (occlusion) and 0.24 (KernelSHAP): these maps depend on what the model learned. Grad-CAM++ stays at 0.94 until deeper blocks are randomised, and **Integrated Gradients fails** (0.84 even after the whole network is re-initialised), so its good deletion score is not evidence of faithfulness.
- **Agreement** between methods is moderate: Grad-CAM vs Grad-CAM++ 0.84, vs occlusion 0.38, vs Integrated Gradients 0.51, vs KernelSHAP 0.22.
- In the most confident errors (mostly gliomas called *no tumour*) Grad-CAM sometimes falls on the skull rim or away from the lesion and sometimes covers it while the prediction is still wrong: a plausible map does not make the decision correct.

The demo therefore uses **Grad-CAM** (passes the sanity check; with Grad-CAM++ the highest insertion AUC) by default, with **occlusion** as a model-agnostic cross-check.

<table>
  <tr>
    <td align="center" width="50%"><img src="results/figures/xai_faithfulness.png" alt="Deletion and insertion AUC" width="100%"><br><sub>Faithfulness: deletion (↓) and insertion (↑) AUC</sub></td>
    <td align="center" width="50%"><img src="results/figures/xai_sanity_check.png" alt="Randomisation sanity check" width="100%"><br><sub>Sanity check: cascading randomisation</sub></td>
  </tr>
</table>
<p align="center"><img src="results/figures/xai_grid_correct.png" alt="Explanations, two correct cases per class" width="100%"><br><sub>Correctly classified test slices, two per class</sub></p>
<p align="center"><img src="results/figures/xai_grid_misclassified.png" alt="Explanations of the most confident errors" width="100%"><br><sub>The most confident errors</sub></p>

## 📄 White paper

[`paper/brain_tumor_mri_white_paper.pdf`](paper/brain_tumor_mri_white_paper.pdf) (LaTeX source `paper/brain_tumor_mri_white_paper.tex`,
references `paper/references.bib`, auto-generated numbers `paper/numbers.tex` and tables `paper/tables/`). Sections: abstract,
introduction, related work, data and duplicate audit, method, results, explainability, in-browser demonstrator,
limitations, ethics and conclusion. All citations were checked against Crossref or arXiv.

## 🗂️ Data

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by **Masoud Nickparvar**
(Kaggle, version 2, **CC BY 4.0**), compiled from figshare (Cheng et al., 2015), SARTAJ and Br35H (gliomas from figshare only; the SARTAJ gliomas were replaced in curation): 7,200 slices,
Training 4 × 1,400, Testing 4 × 400. The dataset is **not** committed; it is downloaded at run time:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

print("Path to dataset files:", path)
```
The eight demo samples in `docs/samples/` are resized Testing images, two per class, drawn with a fixed seed. Preset rule: only use test images with no candidate pair at all at Hamming ≤ 10, whatever the correlation; augmented (`*-aug-*`) files are also excluded.
They are redistributed under CC BY 4.0 with attribution.

## 🧾 Review files (`results/review/`)

| File | Contents |
|---|---|
| `train_files.csv`, `val_files.csv`, `test_files.csv`, `excluded_files.csv` | The exact split used (relative path, class, label) |
| `hashes_and_groups.csv` | MD5, 64-bit pHash, duplicate group, subset and leakage flags for all 7,200 images |
| `duplicate_pairs.csv` | Every strict or linked pair with Hamming distance, thumbnail correlation and cross-split/label-conflict flags |
| `candidate_pairs_hamming_le10.csv.gz` | All pairs within Hamming 10, for re-analysis with other thresholds |
| `dedup_summary.json` | Audit counts |
| `test_predictions.csv` | Raw per-image test predictions: path, true, predicted, calibrated and raw class probabilities, confidence, entropy, uncertain flag, duplicate flags |

## 🚀 How to run

```bash
git clone https://github.com/Rishabh-bgp/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_splits.py                       # hash audit + split files
cd src && python -m btmri.train --run clean && python -m btmri.train --run naive && cd ..
python scripts/make_predictions.py                   # single source of truth
python scripts/make_report.py                        # metrics, tables, figures, docs/stats.json
python scripts/run_xai.py && python scripts/make_presets.py
python scripts/export_onnx.py                        # docs/model/*.onnx
python scripts/make_paper_numbers.py && cd paper && tectonic brain_tumor_mri_white_paper.tex
pytest -q tests && node tests/test_logic.js
```
Use the trained model directly:
```python
import torch, sys; sys.path.insert(0, "src")
from btmri.model import build_model, to_input
from btmri.data import load_gray_square
m = build_model(pretrained=False); m.load_state_dict(torch.load("models/brain_tumor_effnetb0.pt", map_location="cpu")); m.eval()
x = torch.from_numpy(load_gray_square("slice.jpg", 224))[None]
print(torch.softmax(m(to_input(x)) / 1.91, 1))   # glioma, meningioma, notumor, pituitary
```
Run the web demo locally: `cd docs && python -m http.server 8000`, then open http://localhost:8000.

## 📁 Repository layout
```
src/btmri/   data, dedup, splits, model, train, calibrate, xai
scripts/     build_splits, make_predictions, make_report, run_xai, make_presets, export_onnx, make_paper_numbers, figures
notebooks/   five executed, Colab-ready notebooks
results/     predictions/, review/, runs/ (histories, logits, naive checkpoint), xai/, figures/, metrics.json
models/      brain_tumor_effnetb0.pt (PyTorch) and brain_tumor_effnetb0.onnx
docs/        NEURO//SCAN web app on GitHub Pages (index.html, style.css, logic.js, app.js, model/, samples/, stats.json, presets.json)
paper/       white paper PDF, LaTeX source, bibliography, generated numbers and tables
tests/       pytest (hashing, calibration, split integrity) and a Node test for the demo logic
```

## ⚠️ Limitations
No patient identifiers (other slices of a patient may still cross the split) · duplicate thresholds set by inspection, and pHash can still miss some near-copies ·
no-tumour images come from a different source collection (possible source shortcut) · single 2-D slice, four labels ·
one training seed per configuration · image-level bootstrap CIs (slightly optimistic) · one validation set used for checkpoint, temperature and threshold · no lesion masks, so explanations cannot be checked for localisation.

## 🔗 More projects
- [Machine Learning Part 3](https://github.com/Rishabh-bgp/machine-learning-part-3): road damage detection with YOLO11 · [ROAD//SCAN](https://rishabh-bgp.github.io/machine-learning-part-3/)
- [Machine Learning Part 2](https://github.com/Rishabh-bgp/machine-learning-part-2): YOLO11 task tour & potato leaf disease classification
- [Machine Learning Part 1](https://github.com/Rishabh-bgp/machine-learning-part-1): heart-disease classification & insurance-cost regression
- [Responsible AI on UCI Adult](https://github.com/Rishabh-bgp/responsible-ai-uci-adult): fairness, explainability & governance audit · [FAIR//AUDIT](https://rishabh-bgp.github.io/responsible-ai-uci-adult/)
- [beed-ml-study](https://github.com/Rishabh-bgp/beed-ml-study) and [pan-cancer-classification](https://github.com/Rishabh-bgp/pan-cancer-classification): more leakage-aware biomedical ML
- [All repositories](https://github.com/Rishabh-bgp?tab=repositories) · [Profile](https://github.com/Rishabh-bgp)

## 📄 License
[MIT](LICENSE) © 2026 Rishabh Aryan. Dataset © its Kaggle author and source collections (CC BY 4.0); EfficientNet-B0
weights from timm (Apache-2.0); ONNX Runtime Web by Microsoft (MIT); Captum (BSD-3-Clause).
