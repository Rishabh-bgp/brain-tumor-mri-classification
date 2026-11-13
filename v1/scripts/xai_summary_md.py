"""Write results/xai/summary.md (README text) from the saved XAI CSVs."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
fa = pd.read_csv(ROOT / "results/xai/faithfulness_per_image.csv").groupby("method")[["deletion_auc", "insertion_auc"]].mean()
san = pd.read_csv(ROOT / "results/xai/sanity_cascading_randomisation.csv").groupby(["method", "stage"]).spearman.mean()
ag = pd.read_csv(ROOT / "results/xai/method_agreement_spearman.csv", index_col=0)
t = f"""**What the numbers say.**
- **Insertion** separates the methods from the random baseline ({fa.loc['random', 'insertion_auc']:.3f}): Grad-CAM {fa.loc['gradcam', 'insertion_auc']:.3f}, Grad-CAM++ {fa.loc['gradcampp', 'insertion_auc']:.3f}, occlusion {fa.loc['occlusion', 'insertion_auc']:.3f}, KernelSHAP {fa.loc['kernel_shap', 'insertion_auc']:.3f}, Integrated Gradients {fa.loc['integrated_gradients', 'insertion_auc']:.3f}.
- **Deletion** (black fill) does not: only Integrated Gradients ({fa.loc['integrated_gradients', 'deletion_auc']:.3f}) beats random ({fa.loc['random', 'deletion_auc']:.3f}). Scattered black pixels push slices out of distribution faster than deleting one compact region, so this metric favours pixel-scattered maps under a black baseline.
- **Sanity check:** randomising only the classifier drops the rank correlation to {san.loc[('gradcam', 'classifier')]:.2f} (Grad-CAM), {san.loc[('occlusion', 'classifier')]:.2f} (occlusion) and {san.loc[('kernel_shap', 'classifier')]:.2f} (KernelSHAP): these maps depend on what the model learned. Grad-CAM++ stays at {san.loc[('gradcampp', 'classifier')]:.2f} until deeper blocks are randomised, and **Integrated Gradients fails** ({san.loc[('integrated_gradients', 'conv_stem')]:.2f} even after the whole network is re-initialised), so its good deletion score is not evidence of faithfulness.
- **Agreement** between methods is moderate: Grad-CAM vs Grad-CAM++ {ag.loc['gradcam', 'gradcampp']:.2f}, vs occlusion {ag.loc['gradcam', 'occlusion']:.2f}, vs Integrated Gradients {ag.loc['gradcam', 'integrated_gradients']:.2f}, vs KernelSHAP {ag.loc['gradcam', 'kernel_shap']:.2f}.
- In several confident errors (gliomas called *no tumour*) the maps sit on the ventricles or outside the brain rather than on the visible lesion.

The demo therefore uses **Grad-CAM** (passes the sanity check, best insertion) by default, with **occlusion** as a model-agnostic cross-check."""
(ROOT / "results/xai/summary.md").write_text(t + "\n")
print(t)
