# Results catalogue

[Documentation home](README.md) · [Top-level README](../README.md) · [v2 README](../v2/README.md)

Every file in [`v2/results/`](../v2/results/) (52 files). The v1 results are in [`v1/results/`](../v1/results/)
with the same layout as at the `v1.0` tag.

## `figures`
| File | Size | Contents |
|---|---|---|
| [`figures/confusion_matrix.png`](../v2/results/figures/confusion_matrix.png) | 56.6 kB | Confusion matrix of the leakage-aware model on the Testing folder (counts and row percentages). |
| [`figures/dedup_examples.png`](../v2/results/figures/dedup_examples.png) | 593.2 kB | Example cross-split near-duplicate pairs with their Hamming distance and correlation. |
| [`figures/dedup_per_class.png`](../v2/results/figures/dedup_per_class.png) | 43.6 kB | Testing images with a strict / linked duplicate in Training, per class. |
| [`figures/reliability.png`](../v2/results/figures/reliability.png) | 55.4 kB | Reliability diagram of the leakage-aware model, uncalibrated vs temperature-scaled (plot uses 10 bins with n ≥ 5; ECE in the legend uses 15). |
| [`figures/roc_curves.png`](../v2/results/figures/roc_curves.png) | 54.2 kB | One-vs-rest ROC curves per class, leakage-aware model. |
| [`figures/training_curves_clean.png`](../v2/results/figures/training_curves_clean.png) | 65.9 kB | Loss and validation metrics per epoch, leakage-aware run. |
| [`figures/training_curves_naive.png`](../v2/results/figures/training_curves_naive.png) | 60.6 kB | Loss and validation metrics per epoch, naive run. |
| [`figures/xai_agreement.png`](../v2/results/figures/xai_agreement.png) | 65.3 kB | Spearman agreement matrix between attribution methods. |
| [`figures/xai_faithfulness.png`](../v2/results/figures/xai_faithfulness.png) | 62.7 kB | Deletion and insertion AUC per method vs random. |
| [`figures/xai_grid_correct.png`](../v2/results/figures/xai_grid_correct.png) | 1.8 MB | Attribution maps for correctly classified Testing images. |
| [`figures/xai_grid_misclassified.png`](../v2/results/figures/xai_grid_misclassified.png) | 1.4 MB | Attribution maps for the most confident errors. |
| [`figures/xai_lime_examples.png`](../v2/results/figures/xai_lime_examples.png) | 903.4 kB | KernelSHAP and LIME maps side by side for two Testing images per class from the evaluation sample. |
| [`figures/xai_sanity_check.png`](../v2/results/figures/xai_sanity_check.png) | 90.2 kB | Cascading-randomisation curves per method. |

## `(top level)`
| File | Size | Contents |
|---|---|---|
| [`metrics.json`](../v2/results/metrics.json) | 17.7 kB | All evaluation metrics: `clean` and `naive` × {full_test, dedup_test, test_with_strict_train_dup, test_without_linked_train_dup} (accuracy, macro-F1, per-class P/R/F1, confusion matrix, ECE, AUC, CIs, uncertain rates), `leakage_by_class`, `sensitivity_looser_rule`, calibration summary. |
| [`onnx_verification.json`](../v2/results/onnx_verification.json) | 0.3 kB | ONNX vs PyTorch: max logit difference, arg-max agreement, CAM difference, ONNX CAM vs autograd Grad-CAM, file size. |
| [`v1_vs_v2.json`](../v2/results/v1_vs_v2.json) | 1.8 kB | The v1 vs v2 comparison rows (metric, v1, v2). |
| [`v1_vs_v2.md`](../v2/results/v1_vs_v2.md) | 1.2 kB | The same table in Markdown (used in the READMEs and the changelog). |

## `predictions`
| File | Size | Contents |
|---|---|---|
| [`predictions/calibration.json`](../v2/results/predictions/calibration.json) | 1.0 kB | Temperature, validation ECE before/after, threshold τ with its rule, validation coverage and accuracy, for both runs. |
| [`predictions/clean_test_predictions.csv`](../v2/results/predictions/clean_test_predictions.csv) | 217.2 kB; 1,600 rows | Leakage-aware model: one row per Testing image with true class, prediction, calibrated and raw probabilities, confidence, normalised entropy, uncertain flag. **Single source of truth** for every metric. |
| [`predictions/naive_test_predictions.csv`](../v2/results/predictions/naive_test_predictions.csv) | 209.6 kB; 1,600 rows | The same for the naive model. |

## `review`
| File | Size | Contents |
|---|---|---|
| [`review/augmented_meningioma_matches.csv`](../v2/results/review/augmented_meningioma_matches.csv) | 22.6 kB; 203 rows | Every `*-aug-*` file with its closest non-augmented match, transform, Hamming distance, correlation and whether it is linked. |
| [`review/augmented_summary.json`](../v2/results/review/augmented_summary.json) | 0.7 kB | Counts and accuracies from the augmented-file audit. |
| [`review/candidate_pairs_hamming_le10.csv.gz`](../v2/results/review/candidate_pairs_hamming_le10.csv.gz) | 322.7 kB; 31,956 rows | Every image pair within Hamming 10: paths, Hamming, correlation, MD5 equality, cross-split, strict / linked_v1 / linked flags. |
| [`review/dedup_summary.json`](../v2/results/review/dedup_summary.json) | 1.6 kB | Counts from the duplicate audit and split (pairs, groups, removed images, subset sizes per class). |
| [`review/duplicate_pairs.csv`](../v2/results/review/duplicate_pairs.csv) | 663.2 kB; 6,177 rows | The linked pairs only. |
| [`review/excluded_files.csv`](../v2/results/review/excluded_files.csv) | 51.8 kB; 1,223 rows | Training-folder images removed from training (cross-split groups and label conflicts). |
| [`review/hashes_and_groups.csv`](../v2/results/review/hashes_and_groups.csv) | 1.1 MB; 7,200 rows | Every image: path, folder, class, MD5, pHash, linked and strict group ids, subset, and leakage flags. |
| [`review/preset_pool.json`](../v2/results/review/preset_pool.json) | 0.4 kB | Size of the demo preset pool per class and the chosen presets. |
| [`review/test_files.csv`](../v2/results/review/test_files.csv) | 67.1 kB; 1,600 rows | Testing images (the official folder). |
| [`review/test_predictions.csv`](../v2/results/review/test_predictions.csv) | 217.2 kB; 1,600 rows | Testing predictions of the leakage-aware model joined with the duplicate flags; used by the demo checks and review. |
| [`review/train_files.csv`](../v2/results/review/train_files.csv) | 161.4 kB; 3,709 rows | Training images of the leakage-aware split. |
| [`review/val_files.csv`](../v2/results/review/val_files.csv) | 29.1 kB; 668 rows | Validation images of the leakage-aware split. |

## `runs`
| File | Size | Contents |
|---|---|---|
| [`runs/clean/best.pt`](../v2/results/runs/clean/best.pt) | 16.3 MB | Leakage-aware checkpoint (PyTorch state dict); identical to `v2/models/brain_tumor_effnetb0.pt`. |
| [`runs/clean/history.csv`](../v2/results/runs/clean/history.csv) | 1.5 kB; 12 rows | Per-epoch train loss, validation loss/accuracy/macro-F1, learning rate and seconds. |
| [`runs/clean/test_logits.csv`](../v2/results/runs/clean/test_logits.csv) | 119.0 kB; 1,600 rows | Raw logits on the Testing images. |
| [`runs/clean/val_logits.csv`](../v2/results/runs/clean/val_logits.csv) | 50.8 kB; 668 rows | Raw logits on the validation images (used to fit T and τ). |
| [`runs/naive/best.pt`](../v2/results/runs/naive/best.pt) | 16.3 MB | Naive checkpoint. |
| [`runs/naive/history.csv`](../v2/results/runs/naive/history.csv) | 1.5 kB; 12 rows | Per-epoch log, naive run. |
| [`runs/naive/test_logits.csv`](../v2/results/runs/naive/test_logits.csv) | 119.1 kB; 1,600 rows | Raw logits on the Testing images, naive run. |
| [`runs/naive/val_logits.csv`](../v2/results/runs/naive/val_logits.csv) | 63.6 kB; 840 rows | Raw logits on the naive validation images. |

## `xai`
| File | Size | Contents |
|---|---|---|
| [`xai/faithfulness_per_image.csv`](../v2/results/xai/faithfulness_per_image.csv) | 70.7 kB; 600 rows | Per image and method: deletion AUC, insertion AUC, foreground share. |
| [`xai/faithfulness_summary.csv`](../v2/results/xai/faithfulness_summary.csv) | 0.9 kB; 8 rows | Mean and standard deviation of the above per method. |
| [`xai/lime_browser_timing.json`](../v2/results/xai/lime_browser_timing.json) | 0.3 kB | Seconds for 200 forward passes in headless Chrome (ONNX Runtime Web, one WASM thread), batches of 4, 10 and 50. |
| [`xai/lime_faithfulness_per_image.csv`](../v2/results/xai/lime_faithfulness_per_image.csv) | 11.9 kB; 100 rows | LIME per image: deletion AUC, insertion AUC, foreground share, constant-map flag (same images as `faithfulness_per_image.csv`). |
| [`xai/lime_sanity_cascading_randomisation.csv`](../v2/results/xai/lime_sanity_cascading_randomisation.csv) | 3.1 kB; 72 rows | LIME cascading model randomisation: per stage and image, Spearman with the trained-model map, constant-map flag. |
| [`xai/lime_stability.csv`](../v2/results/xai/lime_stability.csv) | 2.5 kB; 72 rows | LIME and KernelSHAP maps with seeds 0, 1, 2: per image, Spearman between seed pairs. |
| [`xai/lime_summary.json`](../v2/results/xai/lime_summary.json) | 1.7 kB | Summary of the LIME run: settings, faithfulness, sanity, agreement, target-class and seed-check means, run time. |
| [`xai/method_agreement_spearman.csv`](../v2/results/xai/method_agreement_spearman.csv) | 0.6 kB; 5 rows | Mean Spearman correlation between the five original methods (5 × 5). |
| [`xai/method_agreement_spearman_with_lime.csv`](../v2/results/xai/method_agreement_spearman_with_lime.csv) | 0.8 kB; 6 rows | Mean Spearman correlation between the six methods including LIME (6 × 6); the 5 × 5 part equals `method_agreement_spearman.csv`. |
| [`xai/sanity_cascading_randomisation.csv`](../v2/results/xai/sanity_cascading_randomisation.csv) | 17.8 kB; 360 rows | Per stage, method and image: Spearman correlation with the trained-model map, constant-map flag. |
| [`xai/summary.md`](../v2/results/xai/summary.md) | 1.8 kB | Text summary of the XAI results used in the README. |
| [`xai/target_class_sensitivity.csv`](../v2/results/xai/target_class_sensitivity.csv) | 9.1 kB; 216 rows | Per image, method and other class: Spearman between the map for the predicted class and the map for that other class. |

## Column reference for the main tables
- `review/test_predictions.csv`: `relpath`, `true`, `pred`, `p_glioma`, `p_meningioma`, `p_notumor`, `p_pituitary`, `raw_p_glioma`, `raw_p_meningioma`, `raw_p_notumor`, `raw_p_pituitary`, `confidence`, `norm_entropy`, `uncertain`, `test_has_train_dup_strict`, `test_has_train_dup`, `test_dedup_rep`. `p_*` are calibrated probabilities (logits / T), `raw_p_*` uncalibrated.
- `review/hashes_and_groups.csv`: `relpath`, `split`, `cls`, `label`, `md5`, `phash`, `group`, `group_strict`, `subset`, `leaks_into_test`, `label_conflict`, `test_has_train_dup`, `test_has_train_dup_strict`, `test_dedup_rep`.
- `review/candidate_pairs_hamming_le10.csv.gz`: `hamming`, `corr`, `md5_equal`, `path_i`, `path_j`, `strict_dup`, `linked_v1`, `linked`, `cross_split`, `label_conflict`.
