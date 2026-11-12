**What the numbers say.**
- **Insertion** separates the methods from the random baseline (0.497): Grad-CAM 0.841, Grad-CAM++ 0.842, occlusion 0.803, KernelSHAP 0.749, Integrated Gradients 0.679.
- **Deletion** (black fill) does not: no method beats random (0.283); Integrated Gradients (0.283) comes closest. Scattered black pixels push slices out of distribution faster than deleting one compact region, so this metric favours pixel-scattered maps under a black baseline.
- **Sanity check:** randomising only the classifier drops the rank correlation to 0.11 (Grad-CAM), 0.14 (occlusion) and 0.24 (KernelSHAP): these maps depend on what the model learned. Grad-CAM++ stays at 0.94 until deeper blocks are randomised, and **Integrated Gradients fails** (0.84 even after the whole network is re-initialised), so its good deletion score is not evidence of faithfulness.
- **Agreement** between methods is moderate: Grad-CAM vs Grad-CAM++ 0.84, vs occlusion 0.38, vs Integrated Gradients 0.51, vs KernelSHAP 0.22.
- In the most confident errors (mostly gliomas called *no tumour*) Grad-CAM sometimes falls on the skull rim or away from the lesion and sometimes covers it while the prediction is still wrong: a plausible map does not make the decision correct.

The demo therefore uses **Grad-CAM** (passes the sanity check; with Grad-CAM++ the highest insertion AUC) by default, with **occlusion** as a model-agnostic cross-check.
