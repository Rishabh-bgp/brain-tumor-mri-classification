"""Generate the Colab-ready notebooks in v2/notebooks/ (they are then executed with nbconvert)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REPO = "Rishabh-bgp/brain-tumor-mri-classification"


def badge(fname):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/v2/notebooks/{fname})")


SETUP = r'''# --- Setup: works locally (repo checkout) and on Google Colab ---
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.exists("brain-tumor-mri-classification"):
        subprocess.run(["git", "clone", "-q", "https://github.com/Rishabh-bgp/brain-tumor-mri-classification.git"], check=True)
    os.chdir("brain-tumor-mri-classification/v2")  # this notebook belongs to the v2 code in v2/
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "kagglehub", "imagehash", "captum",
                    "onnx", "onnxruntime", "scikit-image"], check=True)
elif os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")  # -> v2/
elif os.path.isdir("v2"):
    os.chdir("v2")  # started from the repository root
sys.path.insert(0, os.path.abspath("src"))
print("working directory:", os.getcwd())'''

GPU = r'''import torch
print("PyTorch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU (Colab: Runtime > Change runtime type > GPU for faster training)")'''

DL = r'''import kagglehub

# Download latest version
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

print("Path to dataset files:", path)'''


def nb(fname, title, intro, cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{badge(fname)}\n\n{intro}\n\n"
                                        "*Rishabh Aryan · M.Tech (AI & Data Science), IIIT Bhagalpur. "
                                        "Research and education code, not a medical device.*"),
               nbf.v4.new_code_cell(SETUP), nbf.v4.new_code_cell(GPU)]
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.metadata["accelerator"] = "GPU"
    n.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    nbf.write(n, ROOT / "notebooks" / fname)


nb("01_data_and_deduplication.ipynb", "01 · Data download and duplicate audit",
   "Downloads the Kaggle *Brain Tumor MRI Dataset* with `kagglehub`, inspects it, and audits exact and near-duplicate "
   "images (MD5 + 64-bit perceptual hash + thumbnail correlation) before building leakage-aware splits.",
   [("code", DL),
    ("code", r'''from pathlib import Path
import pandas as pd
from btmri.data import list_images
recs = pd.DataFrame(list_images(Path(path)))
print(recs.groupby(["split", "cls"]).size().unstack())'''),
    ("code", r'''import matplotlib.pyplot as plt
from PIL import Image
from btmri import CLASS_LABELS
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for j, c in enumerate(CLASS_LABELS):
    for i, p in enumerate(recs[(recs.split == "Training") & (recs.cls == c)].sample(2, random_state=0).path):
        im = Image.open(p); axes[i, j].imshow(im.convert("L"), cmap="gray"); axes[i, j].axis("off")
        axes[i, j].set_title(f"{CLASS_LABELS[c]} {im.size}", fontsize=9)
plt.tight_layout(); plt.show()
sizes = pd.Series([Image.open(p).size for p in recs.path[::10]]).value_counts().head(8)
print("most common image sizes (every 10th image):\n", sizes)'''),
    ("md", "## Duplicate audit\nTwo images are **linked** when their MD5 matches, or when their 64-bit pHash Hamming distance "
           "is ≤ 6 **and** the Pearson correlation of their 64×64 thumbnails is ≥ 0.95, **or** (added in v2 after review) when the "
           "distance is ≤ 10 and the correlation ≥ 0.98, which catches contrast- or sharpening-altered copies. A **strict** "
           "near-duplicate requires distance ≤ 2 and correlation ≥ 0.98. Connected components of linked images are *groups*."),
    ("code", r'''from btmri.splits import hash_table, candidate_pairs, make_splits
df = hash_table(Path(path))
pairs = candidate_pairs(df)
df = make_splits(df, pairs)
print("exact duplicate files (MD5):", df.md5.duplicated(keep=False).sum(), "| unique MD5:", df.md5.nunique())
print("strict near-duplicate pairs:", pairs.strict_dup.sum(), "of which cross-split:", (pairs.strict_dup & pairs.cross_split).sum())
print("linked pairs:", pairs.linked.sum(), "of which cross-split:", (pairs.linked & pairs.cross_split).sum())
print("extra pairs from the v2 clause (Hamming <= 10 and corr >= 0.98):", (pairs.linked & ~pairs.linked_v1).sum(),
      "of which cross-split:", (pairs.linked & ~pairs.linked_v1 & pairs.cross_split).sum())
te = df.split == "Testing"
print("Testing images with a strict near-duplicate in Training:", df.test_has_train_dup_strict.sum(), "/", te.sum())
print("Testing images with a linked image in Training:", df.test_has_train_dup.sum())
print(df[te].groupby("cls")[["test_has_train_dup_strict", "test_has_train_dup"]].sum())'''),
    ("code", r'''import numpy as np
from btmri.data import load_gray_square
ex = pairs[pairs.strict_dup & pairs.cross_split & ~pairs.md5_equal].sample(6, random_state=1)
fig, axes = plt.subplots(2, 6, figsize=(15, 5.4))
for k, (_, r) in enumerate(ex.iterrows()):
    for i, side in enumerate(("path_i", "path_j")):
        axes[i, k].imshow(load_gray_square(str(Path(path) / r[side]), 160), cmap="gray"); axes[i, k].axis("off")
        axes[i, k].set_title(r[side].replace("/", "\n", 1), fontsize=7)
fig.suptitle("Cross-split near-duplicate pairs (top: one split, bottom: the other)"); plt.tight_layout(); plt.show()'''),
    ("code", r'''print(df.subset.value_counts())
print(df[df.subset != ""].groupby(["subset", "cls"]).size().unstack())
print("label-conflict images:", df.label_conflict.sum())'''),
    ("md", "The committed split files in `results/review/` were produced by `scripts/build_splits.py`, which runs exactly "
           "this code. Check that this run reproduces them:"),
    ("code", r'''saved = pd.read_csv("results/review/hashes_and_groups.csv")
print("identical subset assignment:", (saved.subset.values == df.subset.values).all())
print("identical pHashes:", (saved.phash.values == df.phash.values).all())'''),
    ])

nb("02_training_efficientnet_b0.ipynb", "02 · Training EfficientNet-B0 (leakage-aware and naive splits)",
   "Fine-tunes an ImageNet-pretrained EfficientNet-B0 (timm) on 224×224 greyscale slices. The *leakage-aware* run "
   "uses the group-based split from notebook 01; the *naive* run uses the original Training folder with a random "
   "image-level validation split, for comparison. Set `FORCE_TRAIN = True` to retrain (about 30 min per run on an "
   "8-core CPU, a few minutes on a Colab GPU). When checkpoints already exist the notebook shows the logged history "
   "of the run that produced the committed results.",
   [("code", DL),
    ("code", r'''FORCE_TRAIN = False
from pathlib import Path
for run in ("clean", "naive"):
    ckpt = Path(f"results/runs/{run}/best.pt")
    if FORCE_TRAIN or not ckpt.exists():
        subprocess.run([sys.executable, "-m", "btmri.train", "--run", run, "--epochs", "12"], cwd="src", check=True)
    else:
        print(f"[{run}] using existing checkpoint {ckpt} (set FORCE_TRAIN = True to retrain)")'''),
    ("code", r'''import pandas as pd
for run in ("clean", "naive"):
    h = pd.read_csv(f"results/runs/{run}/history.csv")
    print(f"=== {run} run: best epoch by validation macro-F1 = {int(h.loc[h.val_macro_f1.idxmax(), 'epoch'])}")
    display(h.round(4))'''),
    ("code", r'''from IPython.display import Image, display
for run in ("clean", "naive"):
    display(Image(f"results/figures/training_curves_{run}.png", width=760))'''),
    ("md", "**Training recipe** (see `src/btmri/train.py`): AdamW (lr 5e-4, weight decay 1e-4), one-cycle schedule, "
           "batch 32, 12 epochs, inverse-frequency class weights, augmentation = random resized crop (scale 0.8–1), "
           "horizontal flip, ±10° rotation / 5% translation, brightness/contrast jitter 0.2; seed 42; the checkpoint with "
           "the best validation macro-F1 is kept."),
    ])

nb("03_evaluation_and_calibration.ipynb", "03 · Evaluation, calibration and the leakage comparison",
   "Turns saved logits into the single prediction file (`results/predictions/*_test_predictions.csv`), fits temperature "
   "scaling and the *uncertain* threshold on validation, and derives every reported metric, table, figure and the web "
   "demo's statistics from that file.",
   [("code", r'''subprocess.run([sys.executable, "scripts/make_predictions.py"], check=True)
subprocess.run([sys.executable, "scripts/make_report.py"], check=True)
import json, pandas as pd
cal = json.load(open("results/predictions/calibration.json"))
print(json.dumps(cal, indent=1))'''),
    ("code", r'''pred = pd.read_csv("results/predictions/clean_test_predictions.csv")
display(pred.head())
m = json.load(open("results/metrics.json"))
rows = []
for run in ("clean", "naive"):
    for k, v in m[run].items():
        rows.append({"model": run, "subset": k, "n": v["n"], "accuracy": v["accuracy"], "acc 95% CI": v["accuracy_ci95"],
                     "macro_F1": v["macro_f1"], "ROC-AUC": v.get("roc_auc_macro_ovr"), "ECE": v["ece_calibrated"]})
display(pd.DataFrame(rows).round(4))'''),
    ("code", r'''full = m["clean"]["full_test"]
display(pd.DataFrame(full["per_class"]).T.round(4))
print("confusion matrix (rows = true, cols = predicted; glioma, meningioma, notumor, pituitary):")
print(pd.DataFrame(full["confusion_matrix"], index=["glioma", "meningioma", "notumor", "pituitary"], columns=["glioma", "meningioma", "notumor", "pituitary"]))
print("uncertain rate:", round(full["uncertain_rate"], 4), "| accuracy confident:", full["accuracy_confident"], "| accuracy uncertain:", full["accuracy_uncertain"])'''),
    ("md", "### Leakage by class: Testing images with vs without a strict near-duplicate in the original Training folder"),
    ("code", r'''lb = pd.DataFrame(m["leakage_by_class"]).T
lb.index = pd.MultiIndex.from_tuples([tuple(i.split("|")) for i in lb.index], names=["model", "class", "subset"])
display(lb.unstack(["model", "subset"]))
subprocess.run([sys.executable, "scripts/audit_augmented.py"], check=True)  # files named *-aug-* (augmented meningiomas)'''),
    ("md", "### Robustness check: a looser link rule (no retraining)\nHow many more Testing images would a rule that also links "
           "Hamming 7–10 with correlation ≥ 0.95 tie to our train/validation images, and what is the accuracy without them?"),
    ("code", r'''print(json.dumps(m["sensitivity_looser_rule"], indent=1))
sd = m["clean"]["test_with_strict_train_dup"]
print(sd["macro_f1_note"]); print("macro-F1 over present classes", sd["present_classes"], "=", round(sd["macro_f1_present_classes"], 4))'''),
    ("code", r'''from IPython.display import Image, display
for f in ("confusion_matrix", "roc_curves", "reliability"):
    display(Image(f"results/figures/{f}.png", width=480))'''),
    ])

nb("04_explainable_ai.ipynb", "04 · Explainable AI: five attribution methods, sanity checks and faithfulness",
   "Grad-CAM, Grad-CAM++, Integrated Gradients (Captum), occlusion sensitivity and KernelSHAP on SLIC superpixels, "
   "checked with the cascading model-parameter randomisation test (Adebayo et al., 2018) and scored with deletion / "
   "insertion AUC (Petsiuk et al., 2018). Set `FORCE_XAI = True` to recompute the full study (about 20–30 min on CPU).",
   [("code", r'''import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from pathlib import Path
from btmri import CLASSES, CLASS_LABELS
from btmri.model import build_model
from btmri.xai import GreyModel, explain, METHODS, METHOD_LABELS, deletion_insertion
from btmri.data import download_dataset, load_gray_square
model = build_model(pretrained=False)
model.load_state_dict(torch.load("results/runs/clean/best.pt", map_location="cpu")); model.eval()
gm = GreyModel(model).eval()
root = download_dataset()
pred = pd.read_csv("results/predictions/clean_test_predictions.csv")'''),
    ("md", "### Live example: all five methods on one test slice per class"),
    ("code", r'''fig, axes = plt.subplots(4, 6, figsize=(15, 10.5))
for i, c in enumerate(CLASSES):
    r = pred[(pred.true == c) & (pred.pred == c)].sample(1, random_state=3).iloc[0]
    x = torch.from_numpy(load_gray_square(str(root / r.relpath), 224)).float().div(255)[None, None]
    k = CLASSES.index(r.pred)
    axes[i, 0].imshow(x[0, 0], cmap="gray"); axes[i, 0].set_title(f"{CLASS_LABELS[c]} (p={r.confidence:.2f})", fontsize=9)
    for j, mth in enumerate(METHODS):
        s = explain(gm, x, k, mth)
        d, ins = deletion_insertion(gm, x, k, s)
        axes[i, j + 1].imshow(x[0, 0], cmap="gray"); axes[i, j + 1].imshow(s, cmap="jet", alpha=0.45)
        axes[i, j + 1].set_title(f"{METHOD_LABELS[mth]}\ndel {d:.2f} / ins {ins:.2f}", fontsize=8)
for a in axes.ravel(): a.axis("off")
plt.tight_layout(); plt.show()'''),
    ("md", "### Full study (100 test images, 25 per class; 12 images for the sanity check)"),
    ("code", r'''FORCE_XAI = False
if FORCE_XAI or not Path("results/xai/faithfulness_summary.csv").exists():
    subprocess.run([sys.executable, "scripts/run_xai.py"], check=True)
faith = pd.read_csv("results/xai/faithfulness_per_image.csv")
display(faith.groupby("method")[["deletion_auc", "insertion_auc", "inside_foreground"]].agg(["mean", "std"]).round(3))
san = pd.read_csv("results/xai/sanity_cascading_randomisation.csv")
display(san.groupby(["method", "stage"], sort=False).spearman.mean().unstack().round(3))
display(pd.read_csv("results/xai/method_agreement_spearman.csv", index_col=0).round(3))'''),
    ("code", r'''from IPython.display import Image, display
for f in ("xai_faithfulness", "xai_sanity_check", "xai_agreement", "xai_grid_correct", "xai_grid_misclassified"):
    if Path(f"results/figures/{f}.png").exists():
        display(Image(f"results/figures/{f}.png", width=900 if "grid" in f else 620))'''),
    ])

nb("05_onnx_export_and_web_demo.ipynb", "05 · ONNX export for the in-browser demo",
   "Exports the leakage-aware model to ONNX with two outputs, the class logits and the class activation maps, and checks "
   "the export against PyTorch. For a global-average-pooled linear head, Grad-CAM equals CAM up to a positive scale, so "
   "the browser can compute exact Grad-CAM without gradients.",
   [("code", r'''subprocess.run([sys.executable, "scripts/export_onnx.py"], check=True)
import json; print(json.dumps(json.load(open("results/onnx_verification.json")), indent=1))'''),
    ("code", r'''import onnxruntime as ort, numpy as np, pandas as pd
from btmri.data import download_dataset, load_gray_square
root = download_dataset()
sess = ort.InferenceSession("../docs/model/brain_tumor_effnetb0.onnx")
print([(i.name, i.shape) for i in sess.get_inputs()], [(o.name, o.shape) for o in sess.get_outputs()])
pred = pd.read_csv("results/predictions/clean_test_predictions.csv").sample(5, random_state=0)
for _, r in pred.iterrows():
    x = load_gray_square(str(root / r.relpath), 224).astype(np.float32)[None, None] / 255
    logits, cam = sess.run(None, {"image": x})
    print(r.relpath, "| true", r.true, "| ONNX argmax", ["glioma", "meningioma", "notumor", "pituitary"][logits.argmax()], "| saved pred", r.pred)'''),
    ("md", "Serve the demo locally with `cd docs && python -m http.server 8000`, or open the live site: "
           "https://rishabh-bgp.github.io/brain-tumor-mri-classification/"),
    ])
print("notebooks written")
