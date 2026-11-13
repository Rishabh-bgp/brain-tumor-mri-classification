"""Write results/xai/summary.md (README text) from the saved XAI CSVs."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
fa = pd.read_csv(ROOT / "results/xai/faithfulness_per_image.csv").groupby("method")[["deletion_auc", "insertion_auc"]].mean()
san = pd.read_csv(ROOT / "results/xai/sanity_cascading_randomisation.csv").groupby(["method", "stage"]).spearman.mean()
ag = pd.read_csv(ROOT / "results/xai/method_agreement_spearman.csv", index_col=0)
import json  # noqa: E402
LIME_LINE = ""
if (ROOT / "results/xai/lime_summary.json").exists():  # LIME, v2.1.1 (scripts/run_lime.py)
    ls = json.loads((ROOT / "results/xai/lime_summary.json").read_text())
    LIME_LINE = (f"- **LIME** (added in v2.1.1; same superpixels, baseline and 200 samples as KernelSHAP): insertion "
                 f"{ls['lime_insertion_auc_mean']:.3f}, deletion {ls['lime_deletion_auc_mean']:.3f} (also not better than random). It passes the "
                 f"randomisation test (ρ {ls['lime_sanity_by_stage']['classifier']:.2f} after the classifier, "
                 f"{ls['lime_sanity_by_stage']['conv_stem']:.2f} after full randomisation), agrees most with KernelSHAP "
                 f"(ρ {ls['agreement_lime_vs']['kernel_shap']:.2f}) and is more stable across random seeds than KernelSHAP "
                 f"(ρ {ls['seed_stability_mean_spearman']['lime']:.2f} vs {ls['seed_stability_mean_spearman']['kernel_shap']:.2f}).\n")
t = f"""**What the numbers say.**
- **Insertion** separates the methods from the random baseline ({fa.loc['random', 'insertion_auc']:.3f}): Grad-CAM {fa.loc['gradcam', 'insertion_auc']:.3f}, Grad-CAM++ {fa.loc['gradcampp', 'insertion_auc']:.3f}, occlusion {fa.loc['occlusion', 'insertion_auc']:.3f}, KernelSHAP {fa.loc['kernel_shap', 'insertion_auc']:.3f}, Integrated Gradients {fa.loc['integrated_gradients', 'insertion_auc']:.3f}.
- **Deletion** (black fill) does not: no method beats random ({fa.loc['random', 'deletion_auc']:.3f}); Integrated Gradients ({fa.loc['integrated_gradients', 'deletion_auc']:.3f}) comes closest. Scattered black pixels push slices out of distribution faster than deleting one compact region, so this metric favours pixel-scattered maps under a black baseline.
- **Sanity check:** randomising only the classifier drops the rank correlation to {san.loc[('gradcam', 'classifier')]:.2f} (Grad-CAM), {san.loc[('occlusion', 'classifier')]:.2f} (occlusion) and {san.loc[('kernel_shap', 'classifier')]:.2f} (KernelSHAP): these maps depend on what the model learned. Grad-CAM++ stays at {san.loc[('gradcampp', 'classifier')]:.2f} until deeper blocks are randomised, and **Integrated Gradients fails** ({san.loc[('integrated_gradients', 'conv_stem')]:.2f} even after the whole network is re-initialised), so its deletion score, which only matches random, is not evidence that it reflects the model.
- **Agreement** between methods is moderate: Grad-CAM vs Grad-CAM++ {ag.loc['gradcam', 'gradcampp']:.2f}, vs occlusion {ag.loc['gradcam', 'occlusion']:.2f}, vs Integrated Gradients {ag.loc['gradcam', 'integrated_gradients']:.2f}, vs KernelSHAP {ag.loc['gradcam', 'kernel_shap']:.2f}.
{LIME_LINE}- In the most confident errors (mostly gliomas called *no tumour*) Grad-CAM sometimes falls on the skull rim or away from the lesion and sometimes covers it while the prediction is still wrong: a plausible map does not make the decision correct.

The demo therefore uses **Grad-CAM** (passes the sanity check; with Grad-CAM++ the highest insertion AUC) by default, with **occlusion** as a model-agnostic cross-check."""
(ROOT / "results/xai/summary.md").write_text(t + "\n")
print(t)
