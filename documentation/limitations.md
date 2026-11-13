# Limitations

[Documentation home](README.md) · [Top-level README](../README.md) · [v2 README](../v2/README.md)

These are the specific known weaknesses of this study and demo. Each one limits what the numbers can support.

## Study design
1. **One training seed.** Each configuration was trained once (seed 42). The bootstrap confidence intervals
   (90.5%–93.2% for accuracy) cover test-set sampling only, **not training variance**.
   The v1 → v2 change (93.4% → 91.9%, with 79 changed predictions)
   suggests that run-to-run variation is of the same order as the differences discussed.
2. **CPU training with a small budget.** 12 epochs on an 8-core CPU, batch size 32, no hyper-parameter search, no
   ensembling and no test-time augmentation. Higher accuracy may be possible; the study focuses on evaluation, not
   on the best score.
3. **No external or clinical validation.** All results come from one public Kaggle dataset. There is no test on another
   hospital, scanner or protocol, no prospective data and no comparison with radiologists.
4. **2-D slices, not patients.** The dataset has no patient identifiers, so a patient-level split is impossible. Different
   slices of the same patient that do not look alike can still be in both training and Testing; the duplicate audit only
   catches images that look nearly identical.

## Data
5. **Label noise.** 13 images sit in duplicate groups with conflicting labels, including pairs
   inside the official Testing folder (e.g. `Te-gl_68` labelled glioma and `Te-me_96` labelled meningioma are near-identical),
   so some test errors may be unavoidable.
6. **Near-duplicates remain possible.** pHash plus thumbnail correlation misses copies that were cropped, rotated or edited
   more heavily. A looser rule would link 84 more Testing images
   ([sensitivity check](dedup_and_splits.md#sensitivity-check-a-looser-rule-no-retraining)); most `-aug-` files match no
   other image ([data card](data_card.md#known-issues)).
7. **Source confounding and class imbalance after deduplication.** No-tumour images come from a different collection,
   and only 454 of them remain for training. The model over-predicts "no tumour"
   (precision 0.816).
8. **Coarse labels.** Four classes only; no grading, no other tumour types, no segmentation.

## Calibration and uncertainty
9. **The uncertain threshold does not transfer fully.** τ = 0.69 gives 99.1%
   accuracy on confident validation predictions but 93.8% on Testing. Calibration was fitted on
   in-distribution validation images and is not guaranteed on new data.

## Explainability
10. **Weak faithfulness.** On the deletion test no method beats a random map (Integrated Gradients only matches it); on
    insertion all methods do. The black-fill deletion test is itself hard to interpret for MRI.
11. **Failed sanity checks.** Integrated Gradients keeps a Spearman correlation of 0.84
    with its original map after all weights are randomised, and Grad-CAM++ stays at 0.94 after
    the classifier is randomised.
12. **Small XAI sample and no ground truth.** 100 images for faithfulness, 12 for the sanity check, and no tumour
    masks, so localisation accuracy was not measured. LIME was not run.

## Web demo
13. **Image recompression (fixed in v2.0.2).** Until v2.0.2 the preset images were resized and re-encoded, which changed the
    displayed result of `Te-gl_391`. Since v2.0.2 the presets are the original files, byte-for-byte, and an automated check
    confirms they reproduce the saved predictions within 0.01. Uploaded images are still decoded and resized by the browser's
    canvas, which can differ slightly from PIL; in our test of 12 uploads the largest probability difference to PyTorch was 0.0011.
14. **Precomputed maps for presets only.** Grad-CAM++, Integrated Gradients and KernelSHAP are shown only for the eight presets;
    uploads get Grad-CAM and occlusion.
15. **Any image gets an answer.** The model has no out-of-distribution detector; a non-MRI image still receives one of the four
    classes, possibly with high confidence.
