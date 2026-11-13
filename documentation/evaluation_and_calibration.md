# Evaluation and calibration

[Documentation home](README.md) · [Top-level README](../README.md) · [v2 README](../v2/README.md)

Code: [`v2/scripts/make_predictions.py`](../v2/scripts/make_predictions.py) (the single source of every prediction),
[`v2/src/btmri/calibrate.py`](../v2/src/btmri/calibrate.py), [`v2/scripts/make_report.py`](../v2/scripts/make_report.py);
notebook [`03_evaluation_and_calibration.ipynb`](../v2/notebooks/03_evaluation_and_calibration.ipynb).

## Metrics
- **Accuracy** and **macro-F1** (unweighted mean of per-class F1) on the official Testing folder (n = 1,600, 400 per class).
- **ROC-AUC**, one-vs-rest per class and its macro average, from the predicted probabilities.
- **Expected calibration error (ECE)** with 15 equal-width confidence bins.
- **95% confidence intervals**: percentile bootstrap over Testing images, 1,000 resamples, seed 42. They describe
  test-set sampling only, not the variation between training runs.

## v2 leakage-aware model
| Class | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Glioma | 0.968 | 0.843 | 0.901 | 0.968 |
| Meningioma | 0.941 | 0.840 | 0.888 | 0.983 |
| No tumour | 0.816 | 0.998 | 0.898 | 0.996 |
| Pituitary | 0.980 | 0.995 | 0.988 | 1.000 |
| **Macro** | | | **0.919** (0.905–0.931) | **0.987** |

Confusion matrix (rows = true, columns = predicted):

| | Glioma | Meningioma | No tumour | Pituitary |
|---|---|---|---|---|
| **Glioma** | 337 | 21 | 42 | 0 |
| **Meningioma** | 8 | 336 | 48 | 8 |
| **No tumour** | 1 | 0 | 399 | 0 |
| **Pituitary** | 2 | 0 | 0 | 398 |

Most errors are gliomas and meningiomas predicted as *no tumour* (42 and 48), which shows as
low no-tumour precision (0.816). Only 454 no-tumour
training images remain after removing duplicates; class weights compensate for the count but not for the reduced variety.

## Results on duplicate-defined subsets
| Subset | n | Leakage-aware model, accuracy (95% CI) | Naive model, accuracy (95% CI) |
|---|---|---|---|
| All Testing images | 1,600 | 91.9% (90.5%–93.2%) | 96.1% (95.1%–97.0%) |
| One per duplicate group | 1,342 | 91.9% (90.4%–93.2%) | 95.9% (94.8%–96.9%) |
| Has strict dup. in Training | 441 | 88.0% (85.0%–91.2%) | 99.5% (98.9%–100.0%) |
| No linked dup. in Training | 970 | 92.5% (90.8%–94.0%) | 94.0% (92.6%–95.5%) |

The 441 strict-duplicate images contain no pituitary tumours, so the macro-F1 of that
subset is 0.564 when pituitary counts as 0 and
0.752 over the three present classes (leakage-aware model).

Leakage by class (correct / n):

| Class | Subset | Naive | Leakage-aware |
|---|---|---|---|
| Glioma | strict dup. | 7/9 | 7/9 |
| Glioma | no strict dup. | 339/391 | 330/391 |
| Meningioma | strict dup. | 105/105 | 55/105 |
| Meningioma | no strict dup. | 288/295 | 281/295 |
| No tumour | strict dup. | 327/327 | 326/327 |
| No tumour | no strict dup. | 73/73 | 73/73 |
| Pituitary | no strict dup. | 399/400 | 398/400 |

## Temperature scaling
A single temperature T divides the logits before the softmax. It is fitted on the **validation** images by minimising the
negative log-likelihood (L-BFGS on log T); accuracy is unchanged because the arg-max does not move.

| | Leakage-aware | Naive |
|---|---|---|
| T | 1.909 | 1.735 |
| Validation ECE raw → calibrated | 0.011 → 0.008 | 0.011 → 0.016 |
| Testing ECE raw → calibrated | 0.070 → 0.052 | 0.033 → 0.022 |

Reliability diagram: [`reliability.png`](../v2/results/figures/reliability.png).

## The "uncertain" flag
A prediction is flagged **uncertain** when its calibrated top probability is below τ. τ is the smallest value in
0.50, 0.51, …, 0.99 for which the validation predictions at or above it are at least 99% accurate.

| | Leakage-aware | Naive |
|---|---|---|
| τ | 0.69 | 0.50 |
| Validation coverage / retained accuracy | 98.7% / 99.1% | 100.0% / 99.0% |
| Testing uncertain rate | 3.5% | 0.4% |
| Testing accuracy: confident / uncertain | 93.8% / 39.3% | 96.4% / 42.9% |

The 99% target is reached on validation but not on Testing (93.8%): the validation images are probably closer
to the training images than some Testing images are. The flag therefore separates easier from harder
cases but is no guarantee. The normalised entropy of the probabilities is reported alongside in the prediction files.

## Where the numbers live
`v2/results/metrics.json` (all metrics), `v2/results/predictions/calibration.json` (T and τ), per-image predictions in
`v2/results/predictions/*_test_predictions.csv` and `v2/results/review/test_predictions.csv`; see the
[results catalogue](results_catalogue.md).
