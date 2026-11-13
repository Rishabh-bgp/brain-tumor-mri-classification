# Explainability

[Documentation home](README.md) · [Top-level README](../README.md) · [v2 README](../v2/README.md)

Code: [`v2/src/btmri/xai.py`](../v2/src/btmri/xai.py), [`v2/scripts/run_xai.py`](../v2/scripts/run_xai.py),
[`v2/scripts/plot_xai_summary.py`](../v2/scripts/plot_xai_summary.py); notebook
[`04_explainable_ai.ipynb`](../v2/notebooks/04_explainable_ai.ipynb); results in [`v2/results/xai/`](../v2/results/xai/summary.md).

**Summary.** Heatmaps show which pixels a method associates with the prediction. They are *not* evidence that the model
uses tumour tissue, and in this study none of the six methods beats a random map on the deletion test. Treat them as a
debugging aid, not as a clinical explanation.

## Set-up
- Model: the v2 leakage-aware EfficientNet-B0, wrapped so that it takes the 1-channel 224×224 image in [0, 1] (`GreyModel`).
- Target: the **predicted** class.
- Images: 100 Testing images (25 per true class, `sample(25, random_state=42)`) for faithfulness and agreement;
  12 of them (3 per class) for the sanity check.
- Every map is clipped/absolute-valued as described below and min-max normalised to [0, 1].
- Run: `python scripts/run_xai.py [threads]` and, for LIME, `python scripts/run_lime.py [threads]` from `v2/` (default 8 torch threads).

## Methods
| Method | How it works | Settings in this project |
|---|---|---|
| **Grad-CAM** (Selvaraju et al., 2017) | Weights the last convolutional feature maps (7×7×1280) by the mean gradient of the class score, sums them and applies ReLU. | Last feature map, bilinear up-sampling to 224. For this network (global average pooling + linear head) the weights equal the classifier weights / 49, which is what the browser uses. |
| **Grad-CAM++** (Chattopadhay et al., 2018) | Like Grad-CAM, but weights each spatial position with higher-order gradient terms, which can highlight several regions. | Closed form for an exponential score. |
| **Integrated Gradients** (Sundararajan et al., 2017) | Averages gradients along the straight path from a baseline image to the input and multiplies by (input − baseline). | Captum, black baseline, 32 steps, absolute value. |
| **Occlusion** (Zeiler and Fergus, 2014) | Slides a black square over the image and records how much the class logit drops. | Captum, 32×32 window, stride 16, positive drops only. |
| **KernelSHAP on superpixels** (Lundberg and Lee, 2017) | Splits the image into superpixels, switches random subsets off (black) and fits a weighted linear model whose coefficients approximate Shapley values of the class logit. | SLIC with 64 segments and compactness 0.1, 200 samples, black baseline, positive values only. |
| **LIME** (Ribeiro et al., 2016) | Switches random subsets of superpixels off and fits a linear model weighted by how close each perturbed image is to the original. | Same SLIC superpixels, black baseline and 200 samples as KernelSHAP; exponential kernel (width 0.25) on cosine distance; ridge regression (α = 1) on the class logit; positive coefficients only. Added in v2.1.1 ([details](#lime-added-in-v211)). |


## Faithfulness: deletion and insertion (Petsiuk et al., 2018)
Pixels are ranked by the map (ties broken randomly with a fixed seed). **Deletion** sets the top pixels to black in 32 steps
and records the predicted-class probability: a faithful map should make it drop fast (low area under the curve).
**Insertion** starts from a Gaussian-blurred image (kernel 31, σ = 10) and restores the top pixels: the probability should
rise fast (high area). A random map is the reference. Values: mean (standard deviation) over 100 images.

| Method | Deletion AUC ↓ | Insertion AUC ↑ | Share of map inside the foreground |
|---|---|---|---|
| Grad-CAM | 0.487 (0.35) | 0.841 (0.19) | 79.2% |
| Grad-CAM++ | 0.490 (0.35) | 0.842 (0.18) | 84.0% |
| Integrated Gradients | 0.283 (0.26) | 0.679 (0.25) | 97.8% |
| Occlusion | 0.466 (0.33) | 0.803 (0.23) | 82.7% |
| KernelSHAP (SLIC superpixels) | 0.400 (0.35) | 0.749 (0.25) | 76.5% |
| LIME (SLIC superpixels) | 0.363 (0.36) | 0.777 (0.25) | 80.6% |
| Random baseline | 0.283 (0.15) | 0.497 (0.37) | 57.9% |

- **Deletion:** no method is better than random; Integrated Gradients is level with it (0.283
  vs 0.283) and the others are worse. Blacking out pixels creates out-of-distribution images,
  so this test is hard to interpret for MRI, but it gives no support to the maps.
- **Insertion:** every method beats random; Grad-CAM and Grad-CAM++ are best.
- **Foreground share:** the fraction of map mass on pixels brighter than 0.08 after a 7×7 box blur (a crude head mask,
  not a tumour mask). Higher than random for all methods, so the maps mostly avoid the black background.

Figure: [`xai_faithfulness.png`](../v2/results/figures/xai_faithfulness.png).

## Sanity check: cascading randomisation (Adebayo et al., 2018)
Layers are re-initialised from the classifier downwards and each new map is compared with the original by Spearman rank
correlation (maps average-pooled to 56×56). A method that depends on the learned weights should drop towards 0.

| Method | classifier only | + conv_head | + blocks 6 | + blocks 5–4 | + blocks 3–2 | all layers |
|---|---|---|---|---|---|---|
| Grad-CAM | 0.11 | 0.08 | 0.01 | 0.02 | 0.01 | -0.07 |
| Grad-CAM++ | 0.94 | 0.80 | 0.51 | 0.37 | 0.35 | 0.07 |
| Integrated Gradients | 0.94 | 0.94 | 0.93 | 0.91 | 0.89 | 0.84 |
| Occlusion | 0.14 | 0.04 | 0.23 | 0.07 | 0.06 | 0.03 |
| KernelSHAP (SLIC superpixels) | 0.24 | 0.02 | 0.05 | 0.17 | 0.17 | 0.21 |
| LIME (SLIC superpixels) | -0.03 | 0.04 | 0.06 | 0.16 | -0.02 | -0.02 |

- **Grad-CAM, occlusion, KernelSHAP and LIME** change almost completely once the classifier is randomised: they depend on the model.
- **Integrated Gradients** stays at 0.84 even with all layers random, so it mostly reflects the
  input image (edges and intensity), not the trained model: it fails the test.
- **KernelSHAP** keeps a small correlation (0.21) after full randomisation because its superpixels depend on the image.
- **Grad-CAM++** stays high (0.94) after the classifier is randomised and only falls when the convolutional layers are.
- 7 of 72 Grad-CAM rows produced a constant (all-zero after ReLU) map on the randomised model; their correlation is set to 0.

Figure: [`xai_sanity_check.png`](../v2/results/figures/xai_sanity_check.png).

## Agreement between methods
Mean Spearman correlation between maps on the same image (100 images):

| | Grad-CAM | Grad-CAM++ | Integrated Gradients | Occlusion | KernelSHAP (SLIC superpixels) | LIME (SLIC superpixels) |
|---|---|---|---|---|---|---|
| Grad-CAM | 1.00 | 0.84 | 0.51 | 0.38 | 0.22 | 0.31 |
| Grad-CAM++ | 0.84 | 1.00 | 0.58 | 0.31 | 0.22 | 0.30 |
| Integrated Gradients | 0.51 | 0.58 | 1.00 | 0.31 | 0.28 | 0.34 |
| Occlusion | 0.38 | 0.31 | 0.31 | 1.00 | 0.18 | 0.22 |
| KernelSHAP (SLIC superpixels) | 0.22 | 0.22 | 0.28 | 0.18 | 1.00 | 0.47 |
| LIME (SLIC superpixels) | 0.31 | 0.30 | 0.34 | 0.22 | 0.47 | 1.00 |

Only the two CAM methods agree closely; LIME agrees best with KernelSHAP, which shares its superpixels; the other perturbation and gradient methods highlight different pixels. Figure:
[`xai_agreement.png`](../v2/results/figures/xai_agreement.png).

## LIME (added in v2.1.1)
LIME was added in v2.1.1 alongside KernelSHAP, on the same 100 images (faithfulness, agreement) and 12
images (sanity checks), with the same SLIC superpixels, black baseline, number of samples and target. Code:
`lime()` in [`xai.py`](../v2/src/btmri/xai.py) and [`v2/scripts/run_lime.py`](../v2/scripts/run_lime.py); results in
separate files (`results/xai/lime_*`, `method_agreement_spearman_with_lime.csv`, `target_class_sensitivity.csv`), so the
original five-method files are unchanged. Recomputing the five original maps for the agreement matrix reproduced the saved
5 × 5 agreement exactly (maximum difference 0).

| | LIME | KernelSHAP |
|---|---|---|
| Deletion AUC ↓ (random 0.283) | 0.363 | 0.400 |
| Insertion AUC ↑ (random 0.497) | 0.777 | 0.749 |
| Share of map inside the foreground | 80.6% | 76.5% |
| ρ after classifier randomised / after full randomisation | -0.03 / -0.02 | 0.24 / 0.21 |
| ρ with the other-class maps (target-class check) | -0.15 | -0.20 |
| ρ between three random seeds (stability) | 0.70 | 0.42 |
| Agreement LIME ↔ KernelSHAP | 0.47 | |

What LIME found:
- **Faithfulness.** Insertion 0.777 is above random and above KernelSHAP, below Grad-CAM,
  Grad-CAM++ and occlusion. Deletion 0.363 is, like every method except Integrated
  Gradients (which only matches random), worse than the random map.
- **Sanity check.** LIME passes the model-parameter randomisation test clearly: its correlation with the trained-model
  map is already -0.03 after randomising only the classifier.
- **Agreement.** LIME agrees most with KernelSHAP (0.47), which uses the same superpixels,
  and only weakly with Grad-CAM (0.31) and occlusion (0.22).
- **Stability.** With 200 samples, LIME maps from different random seeds agree better than KernelSHAP maps
  (0.70 vs 0.42), but neither is fully stable.

Settings: SLIC (64 segments, compactness 0.1), 200 samples with each superpixel kept with probability 0.5 (Captum's
default sampler), weights exp(−d²/0.25²) with d the cosine distance between the binary superpixel vector and the all-on
vector (as in the reference `lime` image explainer), weighted ridge regression (α = 1) on the class logit, positive
coefficients kept. Example maps: [`xai_lime_examples.png`](../v2/results/figures/xai_lime_examples.png).

## Target-class check (all six methods)
On the 12 sanity images each map for the predicted class is compared with the maps for the other three classes
(mean Spearman). A map that explains the decision should change when asked about another class:

| Method | Grad-CAM | Grad-CAM++ | Integrated Gradients | Occlusion | KernelSHAP (SLIC superpixels) | LIME (SLIC superpixels) |
|---|---|---|---|---|---|---|
| ρ with other-class maps | -0.00 | 0.92 | 0.94 | -0.18 | -0.20 | -0.15 |

Grad-CAM++ and Integrated Gradients barely change with the target class, consistent with their weak sanity-check results.
The data (label) randomisation test of Adebayo et al. (2018) needs a model retrained on shuffled labels and was **not run**
(this project does not retrain).

## How to read the heatmaps
1. **Warm colours = higher attribution for the predicted class**, normalised per image; compare locations, not absolute values
   between images or methods.
2. **Grad-CAM is coarse** (7×7 grid up-sampled), so it can only point to a region about 32 pixels wide.
3. **A map on the lesion does not prove the model uses the lesion**, and a map elsewhere (skull, edges, background) is a
   warning sign worth inspecting, especially for the no-tumour class, whose images come from a different source.
4. **Check the prediction first.** A map explains the predicted class; for a wrong or uncertain prediction it explains the error.
5. **Compare methods.** When Grad-CAM and occlusion disagree, neither should be trusted on its own.

Example grids: [correct predictions](../v2/results/figures/xai_grid_correct.png),
[misclassified](../v2/results/figures/xai_grid_misclassified.png).
In the web demo, Grad-CAM is exact (computed from the exported class activation maps) and occlusion runs in the browser
with the same 32-pixel window and stride 16 ([web demo](web_demo.md)).
