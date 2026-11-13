# Overview and quick start

[Documentation home](README.md) · [Top-level README](../README.md) · [v2 README](../v2/README.md)

> ⚠️ Research and education project, **not a medical device**. Do not use it for diagnosis or treatment decisions.

## What the project does
The project classifies single 2-D brain MRI slices into four classes (glioma, meningioma, pituitary tumour, no tumour)
on the Kaggle *Brain Tumor MRI Dataset* (version 2) and puts evaluation hygiene first:

1. **Duplicate audit.** MD5, a 64-bit perceptual hash (pHash) and a 64×64 thumbnail correlation find exact and
   near-duplicate images, including 1,270 strict near-duplicate pairs that cross the official
   Training/Testing boundary ([details](dedup_and_splits.md)).
2. **Leakage-aware split.** Every Training image linked to a Testing image is removed (1,223
   images in v2), and the rest is split into train/validation by duplicate group.
3. **Model.** An ImageNet-pretrained EfficientNet-B0 fine-tuned on a CPU ([details](model_and_training.md)).
4. **Calibration and uncertainty.** Temperature scaling and a validation-chosen confidence threshold
   ([details](evaluation_and_calibration.md)).
5. **Explainability.** Five attribution methods with a randomisation sanity check and deletion/insertion metrics
   ([details](explainability.md)).
6. **Demo.** The model runs in the browser via ONNX Runtime Web ([live demo](https://rishabh-bgp.github.io/brain-tumor-mri-classification/), [details](web_demo.md)).

## Headline results (v2, leakage-aware model, official Testing folder, n = 1,600)
| Metric | Value |
|---|---|
| Accuracy (95% bootstrap CI) | 91.9% (90.5%–93.2%) |
| Macro-F1 | 0.919 |
| Macro one-vs-rest ROC-AUC | 0.987 |
| ECE raw → after temperature scaling | 0.070 → 0.052 (T = 1.91) |
| Uncertain rate (τ = 0.69) | 3.5% |
| Naive model trained on the original Training folder | 96.1% |

The naive model is 4.3 points more accurate, and much of the gap appears to
come from near-copies of Testing images in its training data: on the 105 Testing meningiomas with a strict near-duplicate
in Training it is right on 105, the leakage-aware model on 55.
v1 reported 93.4%; see the [changelog](changelog.md) for why v2 differs.

## Quick start

### Google Colab (no installation)
Open any notebook with its badge; the first cell clones the repository, enters `v2/` and installs the dependencies, and
a later cell downloads the dataset with `kagglehub` (public, no API key needed).

| Notebook | Colab |
|---|---|
| [`01_data_and_deduplication.ipynb`](../v2/notebooks/01_data_and_deduplication.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rishabh-bgp/brain-tumor-mri-classification/blob/main/v2/notebooks/01_data_and_deduplication.ipynb) |
| [`02_training_efficientnet_b0.ipynb`](../v2/notebooks/02_training_efficientnet_b0.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rishabh-bgp/brain-tumor-mri-classification/blob/main/v2/notebooks/02_training_efficientnet_b0.ipynb) |
| [`03_evaluation_and_calibration.ipynb`](../v2/notebooks/03_evaluation_and_calibration.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rishabh-bgp/brain-tumor-mri-classification/blob/main/v2/notebooks/03_evaluation_and_calibration.ipynb) |
| [`04_explainable_ai.ipynb`](../v2/notebooks/04_explainable_ai.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rishabh-bgp/brain-tumor-mri-classification/blob/main/v2/notebooks/04_explainable_ai.ipynb) |
| [`05_onnx_export_and_web_demo.ipynb`](../v2/notebooks/05_onnx_export_and_web_demo.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rishabh-bgp/brain-tumor-mri-classification/blob/main/v2/notebooks/05_onnx_export_and_web_demo.ipynb) |

Notebooks 2 and 4 reuse the committed checkpoint and XAI results unless you set `FORCE_TRAIN = True` /
`FORCE_XAI = True`. A GPU runtime makes training much faster but is not required.

### Local (tested on Linux with Python 3.13, CPU only)
```bash
git clone https://github.com/Rishabh-bgp/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification/v2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_splits.py            # downloads the dataset with kagglehub, audits duplicates, writes the split
cd src && python -m btmri.train --run clean && cd ..
python scripts/make_predictions.py && python scripts/make_report.py
```
The full pipeline, expected outputs and run times are in the [reproducibility guide](../REPRODUCIBILITY.md).

### Docker
No Docker image is provided: the project was built and tested on a machine without Docker, and an untested Dockerfile
would not be reliable documentation. The local instructions above should work inside a standard Python container, but this has not been tested.

### Use the trained model
```python
import sys, torch; sys.path.insert(0, "src")          # run from v2/
from btmri.model import build_model, to_input
from btmri.data import load_gray_square
m = build_model(pretrained=False)
m.load_state_dict(torch.load("models/brain_tumor_effnetb0.pt", map_location="cpu")); m.eval()
x = torch.from_numpy(load_gray_square("slice.jpg", 224))[None]
print(torch.softmax(m(to_input(x)) / 1.9089, 1))   # glioma, meningioma, notumor, pituitary
```
