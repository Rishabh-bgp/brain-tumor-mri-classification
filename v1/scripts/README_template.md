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

## 🧠 Live demo: NEURO//SCAN

<p align="center"><img src="docs/screenshot.png" alt="NEURO//SCAN web app: MRI viewer with a Grad-CAM overlay, class probabilities and the Why this prediction panel" width="100%"></p>

A radiology-workstation-style page: drop an MRI slice (or click one of eight test-split samples labelled with their true
class), watch the scan line sweep while EfficientNet-B0 runs through ONNX Runtime Web (WebAssembly), and read:

- **animated, calibrated class probabilities** for glioma, meningioma, pituitary and no tumour;
- a **heat overlay with an opacity slider** and a method toggle: **Grad-CAM** (exact, computed from the ONNX feature map and classifier weights), **occlusion** (49 masked forward passes in the browser), and for the samples precomputed **Grad-CAM++, Integrated Gradients and KernelSHAP**;
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
is true: none of the {DUP_FILES} byte-identical files ({UNIQ_MD5} unique MD5s among 7,200) crosses the folders. At the
**image** level it is not:

| Check | Result |
|---|---|
| Strict near-duplicate pairs (pHash Hamming ≤ 2 and 64×64 thumbnail correlation ≥ 0.98) | {STRICT_PAIRS} pairs, **{STRICT_CROSS} across Training/Testing** |
| Linked pairs (MD5 equal, or Hamming ≤ 6 and correlation ≥ 0.95) | {LINKED_PAIRS} pairs, {LINKED_CROSS} across |
| Testing images with a strict near-duplicate in Training | **{TEST_STRICT} / 1,600 ({TEST_STRICT_PCT})**: no tumour {SD_NO}, meningioma {SD_ME}, glioma {SD_GL}, pituitary {SD_PI} |
| Testing images linked to a Training image | {TEST_LINKED} / 1,600 ({TEST_LINKED_PCT}) |
| Images in groups carrying two different labels | {CONFLICT} |

**Leakage-aware split:** every Training image linked to a Testing image ({REMOVED}, of which {EXCL_NO} are no-tumour) and every
label-conflict image is removed; the rest is split into **{N_TRAIN} train / {N_VAL} validation** images *by duplicate group*.
The official 1,600 Testing images are kept intact, so no linked copy of any test image is seen in training.
Thresholds were chosen by visual inspection; all hashes and candidate pairs are released for re-analysis.

<p align="center"><img src="results/figures/dedup_examples.png" alt="Strict near-duplicates shared by the Training and Testing folders" width="100%"></p>

## 📈 Results (from the saved prediction file)

EfficientNet-B0 (ImageNet-pretrained, 4.0 M parameters), 224 px greyscale, 12 epochs on an 8-core CPU, seed 42.
Every number below, the white paper's tables and the demo's statistics are generated by scripts from
`results/predictions/clean_test_predictions.csv`.

**Leakage-aware model, full Testing folder (n = 1,600):** accuracy **{ACC}** (95% bootstrap CI {ACC_CI}),
macro-F1 **{F1}**, macro one-vs-rest ROC-AUC **{AUC}**, ECE {ECE_RAW} → **{ECE_CAL}** after temperature scaling (T = {TEMP}).
One image per duplicate group (n = {N_DEDUP}): accuracy {ACC_DEDUP}, macro-F1 {F1_DEDUP}.

| Class | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
{PERCLASS_ROWS}

The dominant error is a tumour called *no tumour* ({GLI_NO} gliomas, {MEN_NO} meningiomas): no-tumour recall is perfect but its
precision is {PREC_NO}. Validation accuracy inside the training distribution is {VAL_ACC}, well above test accuracy, which
suggests a shift between the Training and Testing folders beyond the duplicates removed.

**Naive vs leakage-aware.** {NAIVE_TEXT}

| Model | Full Testing | Testing with a strict dup. in Training (n = {N_DUP}) | Testing with no linked dup. (n = {N_NODUP}) |
|---|---|---|---|
| Leakage-aware | {ACC} | {ACC_C_DUP} | {ACC_C_NODUP} |
| Naive (original Training folder) | {ACC_N} | {ACC_N_DUP} | {ACC_N_NODUP} |

**Uncertainty threshold.** A prediction is flagged *uncertain — low confidence* when its calibrated confidence is below
**τ = {THR}**, the smallest value in {0.50, 0.51, …, 0.99} at which retained validation predictions reach ≥ 99% accuracy
(validation coverage {VAL_COV}). On Testing, {UNC_RATE} of predictions are flagged; accuracy is {ACC_CONF} on confident and
{ACC_UNC} on flagged predictions. Normalised entropy is displayed but not thresholded.

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

{XAI_TABLE}

{XAI_TEXT}

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
(Kaggle, version 2, **CC BY 4.0**), compiled from figshare (Cheng et al., 2015), SARTAJ and Br35H: 7,200 slices,
Training 4 × 1,400, Testing 4 × 400. The dataset is **not** committed; it is downloaded at run time:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

print("Path to dataset files:", path)
```
The eight demo samples in `docs/samples/` are resized Testing images (two per class, seeded) that belong to no strict or
linked duplicate pair anywhere in the dataset, redistributed under CC BY 4.0 with attribution.

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
print(torch.softmax(m(to_input(x)) / {TEMP}, 1))   # glioma, meningioma, notumor, pituitary
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
No patient identifiers (other slices of a patient may still cross the split) · duplicate thresholds set by inspection ·
no-tumour images come from a different source collection (possible source shortcut) · single 2-D slice, four labels ·
one training seed per configuration · no lesion masks, so explanations cannot be checked for localisation.

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
