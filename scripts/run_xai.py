"""Explainability study on the leakage-aware model: faithfulness, sanity checks, method agreement and
qualitative grids. (Demo presets are built by scripts/make_presets.py.)"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btmri import CLASS_LABELS, CLASSES  # noqa: E402
from btmri.data import download_dataset, seed_everything  # noqa: E402
from btmri.model import build_model  # noqa: E402
from btmri.xai import (METHOD_LABELS, METHODS, GreyModel, attention_summary, cascading_randomisation,  # noqa: E402
                       deletion_insertion, explain, rank_corr)

seed_everything(42)
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
OUT = ROOT / "results/xai"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "results/figures"
groups = pd.read_csv(ROOT / "results/review/hashes_and_groups.csv")
X = np.load(ROOT.parent / "cache/all_224.npy")
idx_of = {p: i for i, p in enumerate(groups.relpath)}
pred = pd.read_csv(ROOT / "results/predictions/clean_test_predictions.csv")
model = build_model(pretrained=False)
model.load_state_dict(torch.load(ROOT / "results/runs/clean/best.pt"))
model.eval()
gm = GreyModel(model).eval()


def tensor(relpath):
    return torch.from_numpy(X[idx_of[relpath]]).float().div(255)[None, None]


rng = np.random.default_rng(42)
sample = pd.concat([pred[pred.true == c].sample(25, random_state=42) for c in CLASSES]).reset_index(drop=True)

# 1) faithfulness + agreement + attention on 100 test images (target = predicted class)
t0 = time.time()
rows, maps_cache = [], {}
for i, r in sample.iterrows():
    x = tensor(r.relpath); tgt = CLASSES.index(r.pred)
    maps = {m: explain(gm, x, tgt, m) for m in METHODS + ["random"]}
    maps_cache[r.relpath] = maps
    for m, s in maps.items():
        d, ins = deletion_insertion(gm, x, tgt, s)
        att = attention_summary(s, x[0, 0].numpy())
        rows.append({"relpath": r.relpath, "true": r.true, "pred": r.pred, "method": m, "deletion_auc": d,
                     "insertion_auc": ins, "inside_foreground": att["inside_foreground"]})
    if i % 10 == 0:
        print("faith", i, round(time.time() - t0), flush=True)
faith = pd.DataFrame(rows); faith.to_csv(OUT / "faithfulness_per_image.csv", index=False)
summ = faith.groupby("method")[["deletion_auc", "insertion_auc", "inside_foreground"]].agg(["mean", "std"])
summ.to_csv(OUT / "faithfulness_summary.csv")
agree = np.zeros((len(METHODS), len(METHODS)))
for p, maps in maps_cache.items():
    for a, ma in enumerate(METHODS):
        for b, mb in enumerate(METHODS):
            agree[a, b] += rank_corr(maps[ma], maps[mb]) / len(maps_cache)
pd.DataFrame(agree, index=METHODS, columns=METHODS).to_csv(OUT / "method_agreement_spearman.csv")

# 2) sanity check: cascading model-parameter randomisation on 12 images
sub = sample.groupby("true").head(3).reset_index(drop=True)
xs = torch.cat([tensor(p) for p in sub.relpath]); tg = [CLASSES.index(c) for c in sub.pred]
stages = ["classifier", "conv_head", "blocks.6", "blocks.4", "blocks.2", "conv_stem"]
san = pd.DataFrame(cascading_randomisation(model, xs, tg, METHODS, stages_to_use=stages))
san.to_csv(OUT / "sanity_cascading_randomisation.csv", index=False)
print("sanity done", round(time.time() - t0), flush=True)

# figures: faithfulness bars, sanity curves, agreement heatmap
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
order = METHODS + ["random"]
for k, col in enumerate(("deletion_auc", "insertion_auc")):
    mu = [faith[faith.method == m][col].mean() for m in order]
    sd = [faith[faith.method == m][col].std() for m in order]
    ax[k].bar(range(len(order)), mu, yerr=sd, color=["#2bb3c0"] * 5 + ["#888"], capsize=3)
    ax[k].set_xticks(range(len(order)), [METHOD_LABELS[m] for m in order], rotation=35, ha="right", fontsize=8)
    ax[k].set_title(("Deletion AUC (lower is better)" if k == 0 else "Insertion AUC (higher is better)"))
fig.savefig(FIG / "xai_faithfulness.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for m in METHODS:
    s = san[san.method == m].groupby("stage", sort=False).spearman.mean()
    ax.plot(range(len(s)), s.values, "o-", label=METHOD_LABELS[m])
ax.set_xticks(range(len(stages)), stages, rotation=30); ax.set_ylabel("Spearman vs. trained-model map")
ax.set_xlabel("Cumulative randomisation (top \u2192 bottom)"); ax.axhline(0, color="k", lw=0.6); ax.legend(fontsize=7)
ax.set_title("Model-parameter randomisation test")
fig.savefig(FIG / "xai_sanity_check.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(4.8, 4))
ax.imshow(agree, cmap="viridis", vmin=0, vmax=1)
for a in range(len(METHODS)):
    for b in range(len(METHODS)):
        ax.text(b, a, f"{agree[a, b]:.2f}", ha="center", va="center", fontsize=8, color="w" if agree[a, b] < 0.6 else "k")
lab = [METHOD_LABELS[m].replace(" (superpixels)", "") for m in METHODS]
ax.set_xticks(range(len(METHODS)), lab, rotation=35, ha="right", fontsize=8); ax.set_yticks(range(len(METHODS)), lab, fontsize=8)
ax.set_title("Mean Spearman agreement between methods"); fig.savefig(FIG / "xai_agreement.png"); plt.close(fig)


# 3) qualitative grids: per class two confident correct cases, plus misclassified cases
def grid(rows_df, fname, title):
    n = len(rows_df)
    fig, axes = plt.subplots(n, len(METHODS) + 1, figsize=(2 * (len(METHODS) + 1), 2.05 * n))
    axes = np.atleast_2d(axes)
    for i, (_, r) in enumerate(rows_df.iterrows()):
        x = tensor(r.relpath); img = x[0, 0].numpy(); tgt = CLASSES.index(r.pred)
        axes[i, 0].imshow(img, cmap="gray"); axes[i, 0].set_title(f"true {CLASS_LABELS[r.true]}\npred {CLASS_LABELS[r.pred]} ({r.confidence:.2f})", fontsize=7)
        for j, m in enumerate(METHODS):
            s = maps_cache.get(r.relpath, {}).get(m)
            if s is None:
                s = explain(gm, x, tgt, m)
            axes[i, j + 1].imshow(img, cmap="gray"); axes[i, j + 1].imshow(s, cmap="jet", alpha=0.45)
            if i == 0:
                axes[i, j + 1].set_title(METHOD_LABELS[m], fontsize=8)
        for a in axes[i]:
            a.axis("off")
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(FIG / fname, dpi=110); plt.close(fig)


correct = pred[pred.true == pred.pred]
qual = pd.concat([correct[correct.true == c].sample(2, random_state=7) for c in CLASSES])
grid(qual, "xai_grid_correct.png", "Explanations for correctly classified Testing images (two per class)")
wrong = pred[pred.true != pred.pred].sort_values("confidence", ascending=False).head(6)
if len(wrong):
    grid(wrong, "xai_grid_misclassified.png", "Explanations for misclassified Testing images (most confident errors)")

print("done", round(time.time() - t0), flush=True)
print(summ)
print(san.groupby(["method", "stage"], sort=False).spearman.mean().unstack())
