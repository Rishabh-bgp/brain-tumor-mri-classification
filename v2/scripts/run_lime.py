"""LIME study on the leakage-aware model, added in v2.1.1 alongside (not replacing) KernelSHAP.

Uses exactly the evaluation set-up of scripts/run_xai.py: the same 100 Testing images (25 per true class,
sample(25, random_state=42)), the predicted class as target, the same SLIC superpixels and black baseline as KernelSHAP,
the same deletion/insertion metric and the same cascading model-parameter randomisation on the same 12 images.
The existing XAI files are not modified; LIME results go to new files:

  results/xai/lime_faithfulness_per_image.csv         deletion / insertion AUC and foreground share per image
  results/xai/lime_sanity_cascading_randomisation.csv model-parameter randomisation (same stages as run_xai.py)
  results/xai/method_agreement_spearman_with_lime.csv 6 x 6 mean Spearman agreement (the 5 x 5 block is recomputed and
                                                      checked against method_agreement_spearman.csv)
  results/xai/target_class_sensitivity.csv            map for the predicted class vs maps for the other three classes
  results/xai/lime_stability.csv                      LIME and KernelSHAP run with three random seeds
  results/xai/lime_summary.json                       the numbers above, aggregated
  results/figures/xai_lime_examples.png               KernelSHAP vs LIME on example images

Run from v2/:  python scripts/run_lime.py [threads]
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btmri import CLASS_LABELS, CLASSES  # noqa: E402
from btmri.data import seed_everything  # noqa: E402
from btmri.model import build_model  # noqa: E402
from btmri.xai import (METHOD_LABELS, METHODS, GreyModel, attention_summary, cascading_randomisation,  # noqa: E402
                       deletion_insertion, explain, kernel_shap, lime, rank_corr)

seed_everything(42)
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
OUT = ROOT / "results/xai"
FIG = ROOT / "results/figures"
groups = pd.read_csv(ROOT / "results/review/hashes_and_groups.csv")
X = np.load(ROOT.parent / "cache/all_224.npy")
idx_of = {p: i for i, p in enumerate(groups.relpath)}
pred = pd.read_csv(ROOT / "results/predictions/clean_test_predictions.csv")
model = build_model(pretrained=False)
model.load_state_dict(torch.load(ROOT / "results/runs/clean/best.pt"))
model.eval()
gm = GreyModel(model).eval()
ALL = METHODS + ["lime"]


def tensor(relpath):
    return torch.from_numpy(X[idx_of[relpath]]).float().div(255)[None, None]


sample = pd.concat([pred[pred.true == c].sample(25, random_state=42) for c in CLASSES]).reset_index(drop=True)
old = pd.read_csv(OUT / "faithfulness_per_image.csv")
assert list(old.relpath.unique()) == list(sample.relpath), "evaluation sample differs from run_xai.py"

# 1) faithfulness (LIME) + agreement (all six methods) on the same 100 images
t0 = time.time()
rows, agree, maps_cache = [], np.zeros((len(ALL), len(ALL))), {}
for i, r in sample.iterrows():
    x = tensor(r.relpath); tgt = CLASSES.index(r.pred)
    maps = {m: explain(gm, x, tgt, m) for m in ALL}
    maps_cache[r.relpath] = maps
    d, ins = deletion_insertion(gm, x, tgt, maps["lime"])
    att = attention_summary(maps["lime"], x[0, 0].numpy())
    rows.append({"relpath": r.relpath, "true": r.true, "pred": r.pred, "method": "lime", "deletion_auc": d,
                 "insertion_auc": ins, "inside_foreground": att["inside_foreground"],
                 "constant_map": bool(np.ptp(maps["lime"]) == 0)})
    for a, ma in enumerate(ALL):
        for b, mb in enumerate(ALL):
            agree[a, b] += rank_corr(maps[ma], maps[mb]) / len(sample)
    if i % 10 == 0:
        print("faith", i, round(time.time() - t0), flush=True)
faith = pd.DataFrame(rows)
faith.to_csv(OUT / "lime_faithfulness_per_image.csv", index=False)
ag = pd.DataFrame(agree, index=ALL, columns=ALL)
ag.to_csv(OUT / "method_agreement_spearman_with_lime.csv")
old_ag = pd.read_csv(OUT / "method_agreement_spearman.csv", index_col=0)
agree_check = float((ag.loc[METHODS, METHODS] - old_ag.loc[METHODS, METHODS]).abs().max().max())
print("max |recomputed - saved| 5x5 agreement:", agree_check, flush=True)

# 2) model-parameter randomisation (same 12 images and stages as run_xai.py)
sub = sample.groupby("true").head(3).reset_index(drop=True)
xs = torch.cat([tensor(p) for p in sub.relpath]); tg = [CLASSES.index(c) for c in sub.pred]
stages = ["classifier", "conv_head", "blocks.6", "blocks.4", "blocks.2", "conv_stem"]
san = pd.DataFrame(cascading_randomisation(model, xs, tg, ["lime"], stages_to_use=stages))
san.to_csv(OUT / "lime_sanity_cascading_randomisation.csv", index=False)
print("sanity done", round(time.time() - t0), flush=True)

# 3) target-class sensitivity (all six methods): a map that explains the decision should change when it is asked to
#    explain a different class. Spearman between the predicted-class map and each other-class map, same 12 images.
cs = []
for i in range(len(sub)):
    x = xs[i:i + 1]
    for m in ALL:
        ref = explain(gm, x, tg[i], m)
        for k in range(len(CLASSES)):
            if k != tg[i]:
                cs.append({"image": i, "method": m, "other_class": CLASSES[k], "spearman": rank_corr(ref, explain(gm, x, k, m))})
cs = pd.DataFrame(cs)
cs.to_csv(OUT / "target_class_sensitivity.csv", index=False)
print("class sensitivity done", round(time.time() - t0), flush=True)

# 4) stability over random seeds (LIME and KernelSHAP are sampling-based), same 12 images, seeds 0, 1, 2
st = []
for i in range(len(sub)):
    x = xs[i:i + 1]
    for name, fn in (("lime", lime), ("kernel_shap", kernel_shap)):
        ms = [fn(gm, x, tg[i], seed=s) for s in (0, 1, 2)]
        for a, b in combinations(range(3), 2):
            st.append({"image": i, "method": name, "seed_a": a, "seed_b": b, "spearman": rank_corr(ms[a], ms[b])})
st = pd.DataFrame(st)
st.to_csv(OUT / "lime_stability.csv", index=False)

summary = {
    "n_images_faithfulness": int(len(faith)), "n_images_sanity": int(len(sub)),
    "settings": {"superpixels": "SLIC, 64 segments, compactness 0.1 (same as KernelSHAP)", "n_samples": 200,
                 "baseline": "black (0)", "kernel": "exp(-d^2/0.25^2), d = cosine distance to the all-on vector",
                 "surrogate": "weighted ridge regression, alpha = 1", "target": "predicted class logit",
                 "map": "positive coefficients, min-max normalised"},
    "lime_deletion_auc_mean": float(faith.deletion_auc.mean()), "lime_deletion_auc_std": float(faith.deletion_auc.std()),
    "lime_insertion_auc_mean": float(faith.insertion_auc.mean()), "lime_insertion_auc_std": float(faith.insertion_auc.std()),
    "lime_inside_foreground_mean": float(faith.inside_foreground.mean()),
    "lime_constant_maps": int(faith.constant_map.sum()),
    "lime_sanity_by_stage": {s: float(san[san.stage == s].spearman.mean()) for s in stages},
    "agreement_lime_vs": {m: float(ag.loc["lime", m]) for m in METHODS},
    "agreement_5x5_recomputed_max_abs_diff": agree_check,
    "target_class_sensitivity_mean_spearman": {m: float(cs[cs.method == m].spearman.mean()) for m in ALL},
    "seed_stability_mean_spearman": {m: float(st[st.method == m].spearman.mean()) for m in ("lime", "kernel_shap")},
    "seconds": round(time.time() - t0),
}
(OUT / "lime_summary.json").write_text(json.dumps(summary, indent=2))

# figure: KernelSHAP vs LIME on two images per class from the evaluation sample
ex = sample.groupby("true").head(2)
fig, axes = plt.subplots(len(ex), 3, figsize=(6, 2.05 * len(ex)))
for i, (_, r) in enumerate(ex.iterrows()):
    img = tensor(r.relpath)[0, 0].numpy()
    axes[i, 0].imshow(img, cmap="gray")
    axes[i, 0].set_title(f"true {CLASS_LABELS[r.true]}\npred {CLASS_LABELS[r.pred]}", fontsize=7)
    for j, m in enumerate(("kernel_shap", "lime")):
        axes[i, j + 1].imshow(img, cmap="gray"); axes[i, j + 1].imshow(maps_cache[r.relpath][m], cmap="jet", alpha=0.45)
        if i == 0:
            axes[i, j + 1].set_title(METHOD_LABELS[m], fontsize=8)
    for a in axes[i]:
        a.axis("off")
fig.suptitle("KernelSHAP vs LIME (same superpixels, 200 samples)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "xai_lime_examples.png", dpi=110); plt.close(fig)
print(json.dumps(summary, indent=1))
