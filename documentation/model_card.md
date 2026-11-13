# Model card: EfficientNet-B0 brain tumour MRI classifier (v2, leakage-aware)

[Documentation home](README.md) · [Top-level README](../README.md) · [v2 README](../v2/README.md)

> ⚠️ Research and education project, **not a medical device**. Do not use it for diagnosis or treatment decisions.

## Model details
| | |
|---|---|
| Developer | Rishabh Aryan (M.Tech in AI and Data Science, IIIT Bhagalpur) |
| Version | v2 (trained for tag `v2.0`; unchanged in v2.1.0) |
| Type | Image classifier, EfficientNet-B0 (`timm` `efficientnet_b0.ra_in1k`) fine-tuned from ImageNet |
| Input | One 2-D MRI slice, converted to greyscale, zero-padded to square, resized to 224×224 |
| Output | Probabilities for glioma, meningioma, no tumour, pituitary tumour (temperature T = 1.91), an "uncertain" flag (top probability < 0.69), Grad-CAM map |
| Files | [`v2/models/brain_tumor_effnetb0.pt`](../v2/models/brain_tumor_effnetb0.pt) (PyTorch), [`v2/models/brain_tumor_effnetb0.onnx`](../v2/models/brain_tumor_effnetb0.onnx) (ONNX, logits + CAM) |
| Licence | Repository licence MIT ([LICENSE](../LICENSE)); the training data is CC BY 4.0 |
| Paper | [`v2/paper/brain_tumor_mri_white_paper.pdf`](../v2/paper/brain_tumor_mri_white_paper.pdf) |

## Intended use
- Teaching and research on **evaluation hygiene** (duplicate leakage, calibration, explanation sanity checks) in medical image
  classification.
- A worked, reproducible example and an in-browser demo.

## Out-of-scope use
- **Any clinical use**: diagnosis, triage, screening, treatment planning or second opinions. The model has not been validated
  on clinical data, is not a medical device and has no regulatory approval.
- Images other than single 2-D brain MRI slices similar to the training data (other organs, CT, 3-D volumes, photos of screens).
- Tumour types outside the three classes (e.g. metastases, lymphoma), grading, or segmentation. The model always answers with
  one of the four classes.

## Training data
3,709 training and 668 validation images from the Training folder of the Kaggle
*Brain Tumor MRI Dataset* v2, after removing the 1,223 Training-folder images linked to the Testing folder ([data card](data_card.md), [split method](dedup_and_splits.md)).

## Evaluation (official Testing folder, n = 1,600)
| Metric | Value |
|---|---|
| Accuracy (95% CI) | 91.9% (90.5%–93.2%) |
| Macro-F1 / macro AUC | 0.919 / 0.987 |
| Recall: glioma / meningioma / no tumour / pituitary | 84.3% / 84.0% / 99.8% / 99.5% |
| ECE after temperature scaling | 0.052 |
| Uncertain rate; accuracy of confident / uncertain predictions | 3.5%; 93.8% / 39.3% |

More in [evaluation and calibration](evaluation_and_calibration.md).

## Known limitations and failure modes
- Gliomas and meningiomas are most often mistaken for **no tumour** (42 and 48 of 400 each); a "no tumour" answer
  is the least reliable (precision 0.816).
- The no-tumour images come from a different source than the tumour images, so the model may use source cues.
- One training seed, CPU training, no external test set; see [limitations](limitations.md).
- Heatmaps are not validated localisation; Integrated Gradients fails the randomisation sanity check ([explainability](explainability.md)).

## Ethical considerations
- **Patient data.** The dataset is public and de-identified, without patient IDs or demographics, so performance across age,
  sex, scanner or hospital cannot be assessed and fairness cannot be checked.
- **Risk of over-trust.** High accuracy on a benchmark with near-duplicates can be mistaken for clinical readiness; this project
  reports the leakage-aware number, shows the naive number only for comparison, and labels the demo as not for diagnosis.
- **Privacy.** The web demo runs entirely in the browser; uploaded images are not sent anywhere.
- **Attribution.** The dataset (CC BY 4.0) and its sources are credited in the [references](references.md).
