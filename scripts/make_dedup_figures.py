"""Figures for the duplicate audit: example cross-split pairs and per-class leakage counts."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btmri import CLASS_LABELS, CLASSES  # noqa: E402
from btmri.data import download_dataset, load_gray_square  # noqa: E402

root = download_dataset()
pairs = pd.read_csv(ROOT / "results/review/duplicate_pairs.csv")
d = json.loads((ROOT / "results/review/dedup_summary.json").read_text())
FIG = ROOT / "results/figures"
ex = pairs[pairs.strict_dup & pairs.cross_split & ~pairs.md5_equal].sample(6, random_state=3)
fig, axes = plt.subplots(2, 6, figsize=(13, 5.8))
for k, (_, r) in enumerate(ex.iterrows()):
    for i, side in enumerate(("path_i", "path_j")):
        axes[i, k].imshow(load_gray_square(str(root / r[side]), 160), cmap="gray")
        axes[i, k].set_title(r[side].split("/")[0] + "/" + r[side].split("/")[2], fontsize=7)
        axes[i, k].axis("off")
    axes[1, k].text(80, 175, f"Hamming {r.hamming}, r={r['corr']:.3f}", ha="center", va="top", fontsize=7)
fig.suptitle("Examples of strict near-duplicates shared by the Training and Testing folders", fontsize=10)
fig.tight_layout(h_pad=2.2); fig.savefig(FIG / "dedup_examples.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 3.3))
x = np.arange(4)
tr = [d["subset_by_class"]["train"][c] + d["subset_by_class"]["val"][c] for c in CLASSES]
ex_ = [d["subset_by_class"]["excluded"][c] for c in CLASSES]
ax.bar(x, tr, label="kept (train + validation)", color="#2bb3c0")
ax.bar(x, ex_, bottom=tr, label="removed: linked to a Testing image", color="#f97316")
for i in range(4):
    ax.text(i, tr[i] + ex_[i] + 15, f"{ex_[i]}", ha="center", fontsize=8)
ax.set_xticks(x, [CLASS_LABELS[c] for c in CLASSES]); ax.set_ylabel("Training-folder images")
ax.set_title("Training images removed by the leakage-aware split"); ax.legend(fontsize=8)
fig.savefig(FIG / "dedup_per_class.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("ok")
