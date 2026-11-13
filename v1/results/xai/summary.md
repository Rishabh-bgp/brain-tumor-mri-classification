**What the numbers say.**
- **Insertion** separates the methods from the random baseline (0.424): Grad-CAM 0.828, Grad-CAM++ 0.823, occlusion 0.790, KernelSHAP 0.764, Integrated Gradients 0.680.
- **Deletion** (black fill) does not: only Integrated Gradients (0.260) beats random (0.285). Scattered black pixels push slices out of distribution faster than deleting one compact region, so this metric favours pixel-scattered maps under a black baseline.
- **Sanity check:** randomising only the classifier drops the rank correlation to 0.26 (Grad-CAM), 0.15 (occlusion) and 0.17 (KernelSHAP): these maps depend on what the model learned. Grad-CAM++ stays at 0.91 until deeper blocks are randomised, and **Integrated Gradients fails** (0.84 even after the whole network is re-initialised), so its good deletion score is not evidence of faithfulness.
- **Agreement** between methods is moderate: Grad-CAM vs Grad-CAM++ 0.80, vs occlusion 0.43, vs Integrated Gradients 0.41, vs KernelSHAP 0.19.
- In several confident errors (gliomas called *no tumour*) the maps sit on the ventricles or outside the brain rather than on the visible lesion.

The demo therefore uses **Grad-CAM** (passes the sanity check, best insertion) by default, with **occlusion** as a model-agnostic cross-check.
