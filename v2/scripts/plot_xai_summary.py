"""Re-draw the XAI summary figures from the saved CSVs (no recomputation)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results/figures"
LAB = {"gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++", "integrated_gradients": "Integrated Gradients",
       "occlusion": "Occlusion", "kernel_shap": "KernelSHAP (superpixels)", "lime": "LIME (superpixels)", "random": "Random"}
METHODS = ["gradcam", "gradcampp", "integrated_gradients", "occlusion", "kernel_shap"]
XAI = ROOT / "results/xai"
san = pd.read_csv(XAI / "sanity_cascading_randomisation.csv")
fa = pd.read_csv(XAI / "faithfulness_per_image.csv")
AGF = XAI / "method_agreement_spearman.csv"
if (XAI / "lime_faithfulness_per_image.csv").exists():  # LIME, added in v2.1.1 by scripts/run_lime.py
    METHODS = METHODS + ["lime"]
    san = pd.concat([san, pd.read_csv(XAI / "lime_sanity_cascading_randomisation.csv")], ignore_index=True)
    fa = pd.concat([fa, pd.read_csv(XAI / "lime_faithfulness_per_image.csv")[fa.columns]], ignore_index=True)
    AGF = XAI / "method_agreement_spearman_with_lime.csv"
NM = len(METHODS)
stages = list(dict.fromkeys(san.stage))
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for m in METHODS:
    s = san[san.method == m].groupby("stage", sort=False).spearman.mean()
    ax.plot(range(len(s)), s.values, "o-", label=LAB[m])
ax.set_xticks(range(len(stages)), stages, rotation=30, ha="right")
ax.set_ylabel("Spearman vs. trained-model map"); ax.set_xlabel("Cumulative randomisation (top \u2192 bottom)")
ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-0.2, 1.0); ax.legend(fontsize=7, loc="upper right")
ax.set_title("Model-parameter randomisation test (12 test images)")
fig.savefig(FIG / "xai_sanity_check.png", dpi=150, bbox_inches="tight"); plt.close(fig)
order = METHODS + ["random"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for k, col in enumerate(("deletion_auc", "insertion_auc")):
    mu = [fa[fa.method == m][col].mean() for m in order]
    sd = [fa[fa.method == m][col].std() for m in order]
    ax[k].bar(range(len(order)), mu, yerr=sd, color=["#2bb3c0"] * NM + ["#888"], capsize=3)
    ax[k].set_xticks(range(len(order)), [LAB[m].replace(" (superpixels)", "") for m in order], rotation=35, ha="right", fontsize=8)
    ax[k].set_title("Deletion AUC (lower is better)" if k == 0 else "Insertion AUC (higher is better)")
fig.savefig(FIG / "xai_faithfulness.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("ok")
ag = pd.read_csv(AGF, index_col=0).loc[METHODS, METHODS].values
fig, ax = plt.subplots(figsize=(5, 4.2))
ax.imshow(ag, cmap="viridis", vmin=0, vmax=1)
for a in range(NM):
    for b in range(NM):
        ax.text(b, a, f"{ag[a, b]:.2f}", ha="center", va="center", fontsize=8 if NM <= 5 else 7, color="w" if ag[a, b] < 0.6 else "k")
lab = [LAB[m].replace(" (superpixels)", "") for m in METHODS]
ax.set_xticks(range(NM), lab, rotation=35, ha="right", fontsize=8); ax.set_yticks(range(NM), lab, fontsize=8)
ax.set_title("Mean Spearman agreement between methods (100 images)", fontsize=9)
fig.savefig(FIG / "xai_agreement.png", dpi=150, bbox_inches="tight"); plt.close(fig)
