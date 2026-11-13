# Dataset and data card

[Documentation home](README.md) · [Top-level README](../README.md) · [v2 README](../v2/README.md)

## Source and licence
| Field | Value |
|---|---|
| Name | Brain Tumor MRI Dataset, **version 2** (13 February 2026) |
| Author | Masoud Nickparvar (Kaggle) |
| URL | https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |
| Licence | CC BY 4.0 (redistribution allowed with attribution) |
| Download | `kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")`; the dataset is **not** committed to this repository |
| Composition (per the dataset description) | meningioma and pituitary images from figshare (Cheng et al., 2015) and SARTAJ; glioma images from figshare only (the SARTAJ gliomas were replaced in curation); no-tumour images from Br35H |
| Modality | 2-D MRI slices (JPEG), mixed sequences (T1, T1 with contrast, T2, FLAIR) and orientations (axial, coronal, sagittal); most images 512×512 |
| Identifiers | none: no patient IDs, so a patient-level split is impossible |

Only the eight demo presets in [`docs/samples/`](../docs/samples/) are redistributed, unmodified and with attribution.

## Class counts
| Folder | Glioma | Meningioma | No tumour | Pituitary | Total |
|---|---|---|---|---|---|
| Official Training | 1,400 | 1,400 | 1,400 | 1,400 | 5,600 |
| Official Testing | 400 | 400 | 400 | 400 | 1,600 |

After the v2 leakage-aware split ([method](dedup_and_splits.md)):

| Subset | Glioma | Meningioma | No tumour | Pituitary | Total |
|---|---|---|---|---|---|
| train | 1,134 | 1,081 | 454 | 1,040 | 3,709 |
| val | 201 | 191 | 92 | 184 | 668 |
| test | 400 | 400 | 400 | 400 | 1,600 |
| excluded | 65 | 128 | 854 | 176 | 1,223 |

`excluded` = Training-folder images removed because they share a duplicate group with a Testing image
(6 of them are also in groups with conflicting labels). The no-tumour class loses 854 of its 1,400 Training images,
so the leakage-aware model trains on only 454 no-tumour images; inverse-frequency class weights
compensate ([training](model_and_training.md)).

## Known issues
1. **Near-duplicates across the official folders.** No file is byte-identical across Training and Testing
   (340 files are involved in exact duplication, all within a folder;
   7,013 unique MD5 values among 7,200 files). But 441
   of 1,600 Testing images (27.6%)
   have a strict near-duplicate in Training: no tumour 327, meningioma
   105, glioma 9, pituitary
   0. 630 share a linked group with a
   Training image under the v2 rule.
2. **Augmented files (`*-aug-*`).** The meningioma class contains 100 files named `Tr-aug-me_*` and
   103 named `Te-aug-me_*`, apparently augmented copies. Searching the 8 flips/90° rotations, only
   26 match another image under the linked rule (10 only after a flip or
   rotation), so most are altered in ways the hash does not detect. Among Testing meningiomas without a strict
   near-duplicate, the leakage-aware model is right on 100/102
   augmented and 181/193 other files; the naive model, which
   trained on all augmented Training files, on 99/102 and
   189/193. Leakage through augmented copies is possible but not
   demonstrated. Full list: [`augmented_meningioma_matches.csv`](../v2/results/review/augmented_meningioma_matches.csv).
3. **Label conflicts.** 13 images belong to duplicate groups that carry more than one
   class label (10 conflicting pairs): 6 Training images, all excluded from training, and
   7 Testing images, which stay in the test set. The Testing folder itself contains such pairs, e.g. `Te-gl_68.jpg` (glioma) and `Te-me_96.jpg` (meningioma) at
   Hamming distance 0 and correlation 0.9999, so a few test errors may be
   unavoidable label noise.
4. **Source confounding.** No-tumour images come from a different collection (Br35H) than the tumour classes, so a model
   can partly learn source characteristics (contrast, resolution, framing) instead of pathology.
5. **Version differences.** Version 2 rebalanced the Testing folder; results on earlier versions are not on the same
   test set.

## Files describing the data
[`hashes_and_groups.csv`](../v2/results/review/hashes_and_groups.csv) (every image: MD5, pHash, group, subset, flags),
[`duplicate_pairs.csv`](../v2/results/review/duplicate_pairs.csv),
[`candidate_pairs_hamming_le10.csv.gz`](../v2/results/review/candidate_pairs_hamming_le10.csv.gz) and the split lists;
see the [results catalogue](results_catalogue.md).
