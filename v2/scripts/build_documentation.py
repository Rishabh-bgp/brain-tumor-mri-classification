"""Write the documentation set in ../documentation/ from the saved v2 (and v1) result files.

Every number in the generated pages is read here from results/*.json|csv (v2) or ../v1/results/ (v1); nothing is typed
in by hand. Re-run after regenerating results:  python scripts/build_documentation.py   (from v2/)."""
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # v2/
REPO = ROOT.parent
OUT = REPO / "documentation"
OUT.mkdir(exist_ok=True)
J = lambda p: json.loads((ROOT / p).read_text())    # noqa: E731
m, d, cal, onx = J("results/metrics.json"), J("results/review/dedup_summary.json"), J("results/predictions/calibration.json"), J("results/onnx_verification.json")
aug, pool = J("results/review/augmented_summary.json"), J("results/review/preset_pool.json")
m1 = json.loads((REPO / "v1/results/metrics.json").read_text())
d1 = json.loads((REPO / "v1/results/review/dedup_summary.json").read_text())
c1 = json.loads((REPO / "v1/results/predictions/calibration.json").read_text())
presets = json.loads((REPO / "docs/presets.json").read_text())
cand = pd.read_csv(ROOT / "results/review/candidate_pairs_hamming_le10.csv.gz")
fa = pd.read_csv(ROOT / "results/xai/faithfulness_per_image.csv")
san = pd.read_csv(ROOT / "results/xai/sanity_cascading_randomisation.csv")
ag = pd.read_csv(ROOT / "results/xai/method_agreement_spearman.csv", index_col=0)
hc, hn = pd.read_csv(ROOT / "results/runs/clean/history.csv"), pd.read_csv(ROOT / "results/runs/naive/history.csv")
p1 = pd.read_csv(REPO / "v1/results/predictions/clean_test_predictions.csv").set_index("relpath")
p2 = pd.read_csv(ROOT / "results/predictions/clean_test_predictions.csv").set_index("relpath")


def r(x, nd=3):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def pc(x, nd=1):
    return r(100 * float(x), nd) + "%"


def n(x):
    return f"{int(x):,}"


C, N = m["clean"], m["naive"]
F, FN = C["full_test"], N["full_test"]
U = cal["clean"]["uncertainty"]
S = m["sensitivity_looser_rule"]
LB = m["leakage_by_class"]
CL = ["glioma", "meningioma", "notumor", "pituitary"]
LAB = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumour", "pituitary": "Pituitary"}
GH = "https://github.com/Rishabh-bgp/brain-tumor-mri-classification"
PAGES = "https://rishabh-bgp.github.io/brain-tumor-mri-classification/"
COLAB = "https://colab.research.google.com/github/Rishabh-bgp/brain-tumor-mri-classification/blob/main/v2/notebooks/"
NB = ["01_data_and_deduplication", "02_training_efficientnet_b0", "03_evaluation_and_calibration", "04_explainable_ai",
      "05_onnx_export_and_web_demo"]


HG = pd.read_csv(ROOT / "results/review/hashes_and_groups.csv")
nconf_ex = int((HG.label_conflict & (HG.subset == "excluded")).sum())
nconf_te = int((HG.label_conflict & (HG.subset == "test")).sum())


def pair(a, b):
    q = cand[((cand.path_i.str.endswith("/" + a)) & (cand.path_j.str.endswith("/" + b))) |
             ((cand.path_i.str.endswith("/" + b)) & (cand.path_j.str.endswith("/" + a)))].iloc[0]
    return q


# --- derived numbers --------------------------------------------------------------------------------------------
both = p1.index.intersection(p2.index)
a1, a2 = p1.loc[both], p2.loc[both]
ok1, ok2 = a1.true == a1.pred, a2.true == a2.pred
chg = (a1.pred != a2.pred)
CH = {"changed": int(chg.sum()), "c2w": int((ok1 & ~ok2).sum()), "w2c": int((~ok1 & ok2).sum()), "w2w": int((~ok1 & ~ok2 & chg).sum())}
fm = fa.groupby("method")[["deletion_auc", "insertion_auc", "inside_foreground"]].mean()
fs = fa.groupby("method")[["deletion_auc", "insertion_auc"]].std()
sm = san.groupby(["method", "stage"]).spearman.mean()
stages = list(dict.fromkeys(san.stage))
n_san_img = san.image.nunique()
n_fa_img = fa.relpath.nunique()
const_gc = int(san[san.method == "gradcam"].constant_map.sum()) if "constant_map" in san else 0
rows_gc = int((san.method == "gradcam").sum())
MN = {"gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++", "integrated_gradients": "Integrated Gradients", "occlusion": "Occlusion",
      "kernel_shap": "KernelSHAP (SLIC superpixels)", "random": "Random baseline"}
ex_strict = cand[cand.strict_dup & cand.cross_split & ~cand.md5_equal].sort_values(["hamming", "path_i"]).iloc[0]
ex_v1 = cand[cand.linked_v1 & ~cand.strict_dup & cand.cross_split].sort_values(["hamming", "path_i"], ascending=[False, True]).iloc[0]
ex_v2 = pair("Tr-no_942.jpg", "Te-no_151.jpg")
ex_v2b = pair("Tr-gl_700.jpg", "Te-gl_342.jpg")
ex_conf = pair("Te-gl_68.jpg", "Te-me_96.jpg")
ex_near = pair("Tr-me_102.jpg", "Te-me_142.jpg")
n_cand = len(cand)
n_low = int(((cand.hamming <= 6) & (cand["corr"] < 0.95) & ~cand.md5_equal).sum())
ep_c = int(hc.loc[hc.val_macro_f1.idxmax(), "epoch"])
ep_n = int(hn.loc[hn.val_macro_f1.idxmax(), "epoch"])


def w(name, text):
    (OUT / name).write_text(text.strip() + "\n")


NAV = "[Documentation home](README.md) · [Top-level README](../README.md) · [v2 README](../v2/README.md)"
DISCLAIMER = ("> ⚠️ Research and education project, **not a medical device**. Do not use it for diagnosis or treatment "
              "decisions.")

# ================================================================ index
w("README.md", f"""
# Documentation: Brain Tumour MRI Classification

{DISCLAIMER}

This folder is the long-form documentation of the project. The short version is in the
[top-level README](../README.md); the white paper is [`v2/paper/brain_tumor_mri_white_paper.pdf`](../v2/paper/brain_tumor_mri_white_paper.pdf).
All numbers in these pages are generated by [`v2/scripts/build_documentation.py`](../v2/scripts/build_documentation.py)
from the saved result files, so they match the paper and the demo.

| # | Page | What it covers |
|---|---|---|
| 1 | [Overview and quick start](overview.md) | What the project does, headline results, running it in Colab or locally |
| 2 | [Dataset and data card](data_card.md) | Source, licence, class counts, known issues (near-duplicates, `-aug-` files, label conflicts) |
| 3 | [Duplicate audit and leakage-aware split](dedup_and_splits.md) | MD5, pHash and correlation; v1 and v2 rules with worked examples; sensitivity check |
| 4 | [Model and training](model_and_training.md) | Architecture, hyper-parameters, seeds, hardware and run times |
| 5 | [Evaluation and calibration](evaluation_and_calibration.md) | Metrics, bootstrap confidence intervals, temperature scaling, the uncertain threshold |
| 6 | [Explainability](explainability.md) | Grad-CAM, Grad-CAM++, Integrated Gradients, occlusion, KernelSHAP and LIME; faithfulness and sanity results; reading heatmaps |
| 7 | [Web demo and ONNX export](web_demo.md) | NEURO//SCAN architecture, ONNX export and verification, running the demo locally |
| 8 | [v1 vs v2 and changelog](changelog.md) | What changed between versions, side-by-side results, release tags |
| 9 | [Reproducibility guide](../REPRODUCIBILITY.md) | Exact steps, environment and expected outputs |
| 10 | [Repository structure](repository_structure.md) | Folder layout (`v1/`, `v2/`, shared `docs/`, `tests/`, `documentation/`) |
| 11 | [Script and API reference](scripts_reference.md) | Every script and module, its arguments, inputs and outputs |
| 12 | [Results catalogue](results_catalogue.md) | Every file in `v2/results/` and what it contains |
| 13 | [Model card](model_card.md) | Intended use, out-of-scope use, metrics, limitations, ethics |
| 14 | [Limitations](limitations.md) | Specific, known weaknesses of the study and the demo |
| 15 | [Future work](future_work.md) | Concrete next steps |
| 16 | [References](references.md) | The paper's bibliography with DOIs/URLs, dataset and software citations ([`references.bib`](../references.bib)) |
| 17 | [FAQ and troubleshooting](faq.md) | Common questions and fixes |

Project files at the repository root: [CONTRIBUTING](../CONTRIBUTING.md) · [Code of conduct](../CODE_OF_CONDUCT.md) ·
[CITATION.cff](../CITATION.cff) · [LICENSE](../LICENSE) (MIT).
""")

# ================================================================ overview
w("overview.md", f"""
# Overview and quick start

{NAV}

{DISCLAIMER}

## What the project does
The project classifies single 2-D brain MRI slices into four classes (glioma, meningioma, pituitary tumour, no tumour)
on the Kaggle *Brain Tumor MRI Dataset* (version 2) and puts evaluation hygiene first:

1. **Duplicate audit.** MD5, a 64-bit perceptual hash (pHash) and a 64×64 thumbnail correlation find exact and
   near-duplicate images, including {n(d['strict_cross_split_pairs'])} strict near-duplicate pairs that cross the official
   Training/Testing boundary ([details](dedup_and_splits.md)).
2. **Leakage-aware split.** Every Training image linked to a Testing image is removed ({n(d['train_imgs_removed_cross_split'])}
   images in v2), and the rest is split into train/validation by duplicate group.
3. **Model.** An ImageNet-pretrained EfficientNet-B0 fine-tuned on a CPU ([details](model_and_training.md)).
4. **Calibration and uncertainty.** Temperature scaling and a validation-chosen confidence threshold
   ([details](evaluation_and_calibration.md)).
5. **Explainability.** Five attribution methods with a randomisation sanity check and deletion/insertion metrics
   ([details](explainability.md)).
6. **Demo.** The model runs in the browser via ONNX Runtime Web ([live demo]({PAGES}), [details](web_demo.md)).

## Headline results (v2, leakage-aware model, official Testing folder, n = {n(F['n'])})
| Metric | Value |
|---|---|
| Accuracy (95% bootstrap CI) | {pc(F['accuracy'])} ({pc(F['accuracy_ci95'][0])}–{pc(F['accuracy_ci95'][1])}) |
| Macro-F1 | {r(F['macro_f1'])} |
| Macro one-vs-rest ROC-AUC | {r(F['roc_auc_macro_ovr'])} |
| ECE raw → after temperature scaling | {r(F['ece_raw'])} → {r(F['ece_calibrated'])} (T = {r(cal['clean']['temperature'], 2)}) |
| Uncertain rate (τ = {r(U['conf_threshold'], 2)}) | {pc(F['uncertain_rate'])} |
| Naive model trained on the original Training folder | {pc(FN['accuracy'])} |

The naive model is {r(100 * (FN['accuracy'] - F['accuracy']), 1)} points more accurate, and much of the gap appears to
come from near-copies of Testing images in its training data: on the {LB['naive|meningioma|dup']['n']} Testing meningiomas with a strict near-duplicate
in Training it is right on {LB['naive|meningioma|dup']['correct']}, the leakage-aware model on {LB['clean|meningioma|dup']['correct']}.
v1 reported {pc(m1['clean']['full_test']['accuracy'])}; see the [changelog](changelog.md) for why v2 differs.

## Quick start

### Google Colab (no installation)
Open any notebook with its badge; the first cell clones the repository, enters `v2/` and installs the dependencies, and
a later cell downloads the dataset with `kagglehub` (public, no API key needed).

| Notebook | Colab |
|---|---|
""" + "\n".join(f"| [`{k}.ipynb`](../v2/notebooks/{k}.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}{k}.ipynb) |" for k in NB) + f"""

Notebooks 2 and 4 reuse the committed checkpoint and XAI results unless you set `FORCE_TRAIN = True` /
`FORCE_XAI = True`. A GPU runtime makes training much faster but is not required.

### Local (tested on Linux with Python 3.13, CPU only)
```bash
git clone {GH}.git
cd brain-tumor-mri-classification/v2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_splits.py            # downloads the dataset with kagglehub, audits duplicates, writes the split
cd src && python -m btmri.train --run clean && cd ..
python scripts/make_predictions.py && python scripts/make_report.py
```
The full pipeline, expected outputs and run times are in the [reproducibility guide](../REPRODUCIBILITY.md).

### Docker
No Docker image is provided: the project was built and tested on a machine without Docker, and an untested Dockerfile
would not be reliable documentation. The local instructions above should work inside a standard Python container, but this has not been tested.

### Use the trained model
```python
import sys, torch; sys.path.insert(0, "src")          # run from v2/
from btmri.model import build_model, to_input
from btmri.data import load_gray_square
m = build_model(pretrained=False)
m.load_state_dict(torch.load("models/brain_tumor_effnetb0.pt", map_location="cpu")); m.eval()
x = torch.from_numpy(load_gray_square("slice.jpg", 224))[None]
print(torch.softmax(m(to_input(x)) / {r(cal['clean']['temperature'], 4)}, 1))   # glioma, meningioma, notumor, pituitary
```
""")

# ================================================================ data card
sbc = d["subset_by_class"]
w("data_card.md", f"""
# Dataset and data card

{NAV}

## Source and licence
| Field | Value |
|---|---|
| Name | Brain Tumor MRI Dataset, **version 2** (13 February 2026) |
| Author | Masoud Nickparvar (Kaggle) |
| URL | https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |
| Licence | CC BY 4.0 (redistribution allowed with attribution) |
| Download | `kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")`; the dataset is **not** committed to this repository |
| Composition (per the dataset description) | meningioma and pituitary images from figshare (Cheng et al., 2015) and SARTAJ; glioma images from figshare only (the SARTAJ gliomas were replaced in curation); no-tumour images from Br35H |
| Modality | 2-D MRI slices (JPEG), mixed sequences (T1, T1 with contrast, T2, FLAIR) and orientations (axial, coronal, sagittal); most images 512×512 |
| Identifiers | none: no patient IDs, so a patient-level split is impossible |

Only the eight demo presets in [`docs/samples/`](../docs/samples/) are redistributed, unmodified and with attribution.

## Class counts
| Folder | Glioma | Meningioma | No tumour | Pituitary | Total |
|---|---|---|---|---|---|
| Official Training | 1,400 | 1,400 | 1,400 | 1,400 | 5,600 |
| Official Testing | {sbc['test']['glioma']} | {sbc['test']['meningioma']} | {sbc['test']['notumor']} | {sbc['test']['pituitary']} | {n(d['subset_counts']['test'])} |

After the v2 leakage-aware split ([method](dedup_and_splits.md)):

| Subset | Glioma | Meningioma | No tumour | Pituitary | Total |
|---|---|---|---|---|---|
""" + "\n".join(f"| {s} | " + " | ".join(n(sbc[s][c]) for c in CL) + f" | {n(d['subset_counts'][s])} |" for s in ("train", "val", "test", "excluded")) + f"""

`excluded` = Training-folder images removed because they share a duplicate group with a Testing image
({n(nconf_ex)} of them are also in groups with conflicting labels). The no-tumour class loses {n(sbc['excluded']['notumor'])} of its 1,400 Training images,
so the leakage-aware model trains on only {n(sbc['train']['notumor'])} no-tumour images; inverse-frequency class weights
compensate ([training](model_and_training.md)).

## Known issues
1. **Near-duplicates across the official folders.** No file is byte-identical across Training and Testing
   ({n(d['exact_duplicate_files'])} files are involved in exact duplication, all within a folder;
   {n(d['n_unique_md5'])} unique MD5 values among {n(d['n_images'])} files). But {n(d['test_imgs_with_train_dup_strict'])}
   of {n(d['subset_counts']['test'])} Testing images ({pc(d['test_imgs_with_train_dup_strict'] / d['subset_counts']['test'])})
   have a strict near-duplicate in Training: no tumour {d['test_strict_dup_by_class']['notumor']}, meningioma
   {d['test_strict_dup_by_class']['meningioma']}, glioma {d['test_strict_dup_by_class']['glioma']}, pituitary
   {d['test_strict_dup_by_class']['pituitary']}. {n(d['test_imgs_with_train_dup_linked'])} share a linked group with a
   Training image under the v2 rule.
2. **Augmented files (`*-aug-*`).** The meningioma class contains {aug['n_aug_training']} files named `Tr-aug-me_*` and
   {aug['n_aug_testing']} named `Te-aug-me_*`, apparently augmented copies. Searching the 8 flips/90° rotations, only
   {aug['linked_any']} match another image under the linked rule ({aug['linked_non_identity']} only after a flip or
   rotation), so most are altered in ways the hash does not detect. Among Testing meningiomas without a strict
   near-duplicate, the leakage-aware model is right on {aug['clean_nodup_aug_correct']}/{aug['clean_nodup_aug_n']}
   augmented and {aug['clean_nodup_other_correct']}/{aug['clean_nodup_other_n']} other files; the naive model, which
   trained on all augmented Training files, on {aug['naive_nodup_aug_correct']}/{aug['naive_nodup_aug_n']} and
   {aug['naive_nodup_other_correct']}/{aug['naive_nodup_other_n']}. Leakage through augmented copies is possible but not
   demonstrated. Full list: [`augmented_meningioma_matches.csv`](../v2/results/review/augmented_meningioma_matches.csv).
3. **Label conflicts.** {d['imgs_in_label_conflict_groups']} images belong to duplicate groups that carry more than one
   class label ({d['label_conflict_pairs']} conflicting pairs): {n(nconf_ex)} Training images, all excluded from training, and
   {n(nconf_te)} Testing images, which stay in the test set. The Testing folder itself contains such pairs, e.g. `{Path(ex_conf.path_i).name}` (glioma) and `{Path(ex_conf.path_j).name}` (meningioma) at
   Hamming distance {int(ex_conf.hamming)} and correlation {r(ex_conf['corr'], 4)}, so a few test errors may be
   unavoidable label noise.
4. **Source confounding.** No-tumour images come from a different collection (Br35H) than the tumour classes, so a model
   can partly learn source characteristics (contrast, resolution, framing) instead of pathology.
5. **Version differences.** Version 2 rebalanced the Testing folder; results on earlier versions are not on the same
   test set.

## Files describing the data
[`hashes_and_groups.csv`](../v2/results/review/hashes_and_groups.csv) (every image: MD5, pHash, group, subset, flags),
[`duplicate_pairs.csv`](../v2/results/review/duplicate_pairs.csv),
[`candidate_pairs_hamming_le10.csv.gz`](../v2/results/review/candidate_pairs_hamming_le10.csv.gz) and the split lists;
see the [results catalogue](results_catalogue.md).
""")

# ================================================================ dedup and splits
def prow(q, note):
    return (f"| `{Path(q.path_i).name}` ↔ `{Path(q.path_j).name}` | {int(q.hamming)} | {r(q['corr'], 4)} | "
            f"{'yes' if q.strict_dup else 'no'} | {'yes' if q.linked_v1 else 'no'} | {'yes' if q.linked else 'no'} | {note} |")


w("dedup_and_splits.md", f"""
# Duplicate audit and leakage-aware split

{NAV}

Code: [`v2/src/btmri/dedup.py`](../v2/src/btmri/dedup.py), [`v2/src/btmri/splits.py`](../v2/src/btmri/splits.py),
[`v2/scripts/build_splits.py`](../v2/scripts/build_splits.py); notebook
[`01_data_and_deduplication.ipynb`](../v2/notebooks/01_data_and_deduplication.ipynb).

## Signals
For each of the {n(d['n_images'])} images:
- **MD5** of the file bytes (exact duplicates);
- **pHash**: a 64-bit DCT perceptual hash of the greyscale image (`imagehash.phash`); the Hamming distance between two
  hashes counts differing bits (0 = identical hash);
- **thumbnail correlation**: the Pearson correlation of 64×64 zero-padded greyscale thumbnails.

Every pair within Hamming distance 10 is a *candidate pair* ({n(n_cand)} pairs, all released in
[`candidate_pairs_hamming_le10.csv.gz`](../v2/results/review/candidate_pairs_hamming_le10.csv.gz)). pHash alone is too coarse
for MRI: {n(n_low)} candidate pairs are within Hamming 6 but have correlation below 0.95, mostly different slices of similar
anatomy. The correlation check separates them.

## Rules
| Relation | Rule |
|---|---|
| **strict near-duplicate** | identical MD5, or Hamming ≤ 2 **and** correlation ≥ 0.98 |
| **linked (v1)** | identical MD5, or Hamming ≤ 6 **and** correlation ≥ 0.95 |
| **linked (v2)** | linked (v1), **or** Hamming ≤ 10 **and** correlation ≥ 0.98 |

Connected components of linked images form **groups** (union-find). v2 added the third clause after a review found
contrast- or sharpening-altered copies of Testing images at Hamming 8–10 that v1 missed; it adds
{n(d['extra_pairs_v2_rule'])} linked pairs ({n(d['extra_cross_split_pairs_v2_rule'])} across Training/Testing).

| Count | v1 | v2 |
|---|---|---|
| Strict pairs (across Training/Testing) | {n(d1['strict_pairs'])} ({n(d1['strict_cross_split_pairs'])}) | {n(d['strict_pairs'])} ({n(d['strict_cross_split_pairs'])}) |
| Linked pairs (across) | {n(d1['linked_pairs'])} ({n(d1['linked_cross_split_pairs'])}) | {n(d['linked_pairs'])} ({n(d['linked_cross_split_pairs'])}) |
| Groups | {n(d1['groups'])} | {n(d['groups'])} |
| Testing images sharing a linked group with Training | {n(d1['test_imgs_with_train_dup_linked'])} | {n(d['test_imgs_with_train_dup_linked'])} |
| Training images removed | {n(d1['train_imgs_removed_cross_split'])} | {n(d['train_imgs_removed_cross_split'])} |
| Train / validation images | {n(d1['subset_counts']['train'])} / {n(d1['subset_counts']['val'])} | {n(d['subset_counts']['train'])} / {n(d['subset_counts']['val'])} |
| One-per-group Testing representatives | {n(d1['test_dedup_reps'])} | {n(d['test_dedup_reps'])} |

## Worked examples (real pairs from the candidate file)
| Pair | Hamming | Correlation | Strict | Linked v1 | Linked v2 | What it shows |
|---|---|---|---|---|---|---|
{prow(ex_strict, "byte-different but visually identical copy across the folders: strict, so the Training image is removed")}
{prow(ex_v1, "linked by the v1 rule at its Hamming limit")}
{prow(ex_v2, "altered copy found in the review: missed by v1, linked by v2")}
{prow(ex_v2b, "near-identical neighbouring slice at Hamming 10, linked only by v2")}
{prow(ex_near, "correlation just below 0.95: not linked")}
{prow(ex_conf, "two **Testing** images with different labels: label noise in the official test set")}

Reading the table: a pair is linked when **both** conditions of one clause hold. `Tr-me_102` ↔ `Te-me_142` has a low enough
Hamming distance for v1 but its correlation is below 0.95, so it stays unlinked.

## Building the split
1. Remove from the training pool every Training image whose group contains a Testing image.
2. Remove every Training-folder image in a group with more than one class label (in v2 all such images are already
   removed by step 1).
3. Assign whole groups to validation (about 15% per class, seed 42) and the rest to training, so near-duplicates never
   straddle train and validation.
4. Keep the official {n(d['subset_counts']['test'])} Testing images untouched, so results stay comparable with other work
   that uses the version 2 Testing folder.

After v2, no Testing image has a training or validation partner with correlation ≥ 0.98 within Hamming 10.

## Sensitivity check: a looser rule (no retraining)
pHash can still miss near-copies. As a robustness check, a looser rule that also links pairs at Hamming 7–10 with
correlation ≥ 0.95 adds {n(S['extra_pairs'])} pairs and links {S['extra_test_images_linked_to_train_val']} further Testing
images to the v2 training or validation images ({", ".join(f"{v} {LAB[k].lower()}" for k, v in S['extra_by_class'].items())}).
The leakage-aware model classifies {'all of them' if S['clean_acc_on_extra'] == 1 else pc(S['clean_acc_on_extra']) + ' of them'} correctly; without them its accuracy is
{pc(S['clean_acc_excluding_extra'])} (n = {n(S['n_excluding_extra'])}) against {pc(S['clean_acc_full'])} on all
{n(F['n'])} images. Stored in `results/metrics.json` → `sensitivity_looser_rule`.

## Subsets used in evaluation
- **Has strict dup. in Training**: Testing images with a strict near-duplicate in the *original* Training folder
  (n = {C['test_with_strict_train_dup']['n']}).
- **No linked dup.**: Testing images with no linked image in the original Training folder (n = {C['test_without_linked_train_dup']['n']}).
- **One per group**: one representative per Testing group (n = {C['dedup_test']['n']}).
""")

# ================================================================ model and training
trc = d["subset_by_class"]["train"]
ntr = d["subset_counts"]["train"]
weights = {c: ntr / (4 * trc[c]) for c in CL}
w("model_and_training.md", f"""
# Model and training

{NAV}

Code: [`v2/src/btmri/model.py`](../v2/src/btmri/model.py), [`v2/src/btmri/train.py`](../v2/src/btmri/train.py),
[`v2/src/btmri/data.py`](../v2/src/btmri/data.py); notebook [`02_training_efficientnet_b0.ipynb`](../v2/notebooks/02_training_efficientnet_b0.ipynb).

## Pre-processing (identical in Python and in the browser)
1. Convert to greyscale (`PIL.Image.convert("L")`, ITU-R 601 luma).
2. Zero-pad to a square, keeping the aspect ratio.
3. Resize to 224×224 with bilinear filtering.
4. Replicate to three channels and normalise with ImageNet mean/std.

Pre-processed images are cached as one uint8 array (`cache/all_224.npy`, git-ignored).

## Architecture
| Item | Value |
|---|---|
| Backbone | EfficientNet-B0, `timm` model `efficientnet_b0.ra_in1k`, ImageNet-pretrained |
| Parameters | about 4.0 M |
| Head | global average pooling → dropout 0.2 → linear layer with 4 outputs |
| Final feature map | 7 × 7 × 1280 (used for Grad-CAM) |
| Export wrapper | `ExportWrapper` returns the logits and the four class activation maps (7 × 7) for the browser |

## Training set-up
| Hyper-parameter | Value |
|---|---|
| Optimiser | AdamW, learning rate 5e-4, weight decay 1e-4 |
| Schedule | One-cycle (`OneCycleLR`, `pct_start` 0.1) over all steps |
| Batch size / epochs | 32 / 12 |
| Loss | cross-entropy with inverse-frequency class weights `n_train / (4 · n_class)` |
| Class weights (v2 leakage-aware split) | {", ".join(f"{LAB[c].lower()} {r(weights[c], 3)}" for c in CL)} |
| Augmentation | `RandomResizedCrop` (scale 0.8–1.0, ratio 0.9–1.1), horizontal flip, `RandomAffine` (±10°, 5% translation), `ColorJitter` (brightness and contrast 0.2) |
| Checkpoint | best validation macro-F1 (leakage-aware: epoch {ep_c}; naive: epoch {ep_n}) |
| Seed | 42 for Python, NumPy and PyTorch, and for the batch order |
| Arguments | `python -m btmri.train --run clean|naive [--epochs 12] [--bs 32] [--lr 5e-4] [--seed 42]` (run from `v2/src`) |

**Naive comparison run** (`--run naive`): the same model and settings trained on the original Training folder with a
random, image-level 15% validation split per class (seed 42); evaluated on the same Testing images.

## Hardware and run times
Both v2 runs were trained on an 8-core Intel Xeon CPU without a GPU (PyTorch CPU build, 8 threads).
| Run | Train / val images | Median s per epoch | Total training time | Best epoch | Validation accuracy (best checkpoint) |
|---|---|---|---|---|---|
| Leakage-aware (`clean`) | {n(d['subset_counts']['train'])} / {n(d['subset_counts']['val'])} | {r(hc.seconds.median(), 0)} | {r(hc.seconds.sum() / 60, 0)} min | {ep_c} | {pc(cal['clean']['val_accuracy'])} |
| Naive | 4,760 / 840 | {r(hn.seconds.median(), 0)} | {r(hn.seconds.sum() / 60, 0)} min | {ep_n} | {pc(cal['naive']['val_accuracy'])} |

Several epochs took much longer (up to {r(max(hc.seconds.max(), hn.seconds.max()), 0)} s), which inflates the totals; the median is the better guide to the compute needed.

Training curves: [`training_curves_clean.png`](../v2/results/figures/training_curves_clean.png),
[`training_curves_naive.png`](../v2/results/figures/training_curves_naive.png); per-epoch logs in
`v2/results/runs/<run>/history.csv`.

## Determinism
Seeds are fixed, but CPU kernels and thread scheduling can still change the last digits of the weights between machines,
so a re-run should land close to, not exactly on, the reported numbers. Only one seed was trained per configuration; the
confidence intervals do not include training variance ([limitations](limitations.md)).
""")

# ================================================================ evaluation and calibration
cm = F["confusion_matrix"]
SUBS = [("All Testing images", "full_test"), ("One per duplicate group", "dedup_test"),
        ("Has strict dup. in Training", "test_with_strict_train_dup"), ("No linked dup. in Training", "test_without_linked_train_dup")]


def ci(x):
    return f"{pc(x['accuracy'])} ({pc(x['accuracy_ci95'][0])}–{pc(x['accuracy_ci95'][1])})"


w("evaluation_and_calibration.md", f"""
# Evaluation and calibration

{NAV}

Code: [`v2/scripts/make_predictions.py`](../v2/scripts/make_predictions.py) (the single source of every prediction),
[`v2/src/btmri/calibrate.py`](../v2/src/btmri/calibrate.py), [`v2/scripts/make_report.py`](../v2/scripts/make_report.py);
notebook [`03_evaluation_and_calibration.ipynb`](../v2/notebooks/03_evaluation_and_calibration.ipynb).

## Metrics
- **Accuracy** and **macro-F1** (unweighted mean of per-class F1) on the official Testing folder (n = {n(F['n'])}, 400 per class).
- **ROC-AUC**, one-vs-rest per class and its macro average, from the predicted probabilities.
- **Expected calibration error (ECE)** with 15 equal-width confidence bins.
- **95% confidence intervals**: percentile bootstrap over Testing images, 1,000 resamples, seed 42. They describe
  test-set sampling only, not the variation between training runs.

## v2 leakage-aware model
| Class | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
""" + "\n".join(f"| {LAB[c]} | {r(F['per_class'][c]['precision'])} | {r(F['per_class'][c]['recall'])} | {r(F['per_class'][c]['f1'])} | {r(F['roc_auc_per_class'][c])} |" for c in CL) + f"""
| **Macro** | | | **{r(F['macro_f1'])}** ({r(F['macro_f1_ci95'][0])}–{r(F['macro_f1_ci95'][1])}) | **{r(F['roc_auc_macro_ovr'])}** |

Confusion matrix (rows = true, columns = predicted):

| | Glioma | Meningioma | No tumour | Pituitary |
|---|---|---|---|---|
""" + "\n".join(f"| **{LAB[c]}** | " + " | ".join(str(v) for v in cm[i]) + " |" for i, c in enumerate(CL)) + f"""

Most errors are gliomas and meningiomas predicted as *no tumour* ({cm[0][2]} and {cm[1][2]}), which shows as
low no-tumour precision ({r(F['per_class']['notumor']['precision'])}). Only {n(d['subset_by_class']['train']['notumor'])} no-tumour
training images remain after removing duplicates; class weights compensate for the count but not for the reduced variety.

## Results on duplicate-defined subsets
| Subset | n | Leakage-aware model, accuracy (95% CI) | Naive model, accuracy (95% CI) |
|---|---|---|---|
""" + "\n".join(f"| {a} | {n(C[k]['n'])} | {ci(C[k])} | {ci(N[k])} |" for a, k in SUBS) + f"""

The {C['test_with_strict_train_dup']['n']} strict-duplicate images contain no pituitary tumours, so the macro-F1 of that
subset is {r(C['test_with_strict_train_dup']['macro_f1'])} when pituitary counts as 0 and
{r(C['test_with_strict_train_dup']['macro_f1_present_classes'])} over the three present classes (leakage-aware model).

Leakage by class (correct / n):

| Class | Subset | Naive | Leakage-aware |
|---|---|---|---|
""" + "\n".join(f"| {LAB[c]} | {'strict dup.' if s == 'dup' else 'no strict dup.'} | {LB[f'naive|{c}|{s}']['correct']}/{LB[f'naive|{c}|{s}']['n']} | {LB[f'clean|{c}|{s}']['correct']}/{LB[f'clean|{c}|{s}']['n']} |" for c in CL for s in ("dup", "nodup") if f"naive|{c}|{s}" in LB and LB[f'naive|{c}|{s}']['n']) + f"""

## Temperature scaling
A single temperature T divides the logits before the softmax. It is fitted on the **validation** images by minimising the
negative log-likelihood (L-BFGS on log T); accuracy is unchanged because the arg-max does not move.

| | Leakage-aware | Naive |
|---|---|---|
| T | {r(cal['clean']['temperature'], 3)} | {r(cal['naive']['temperature'], 3)} |
| Validation ECE raw → calibrated | {r(cal['clean']['val_ece_raw'])} → {r(cal['clean']['val_ece_calibrated'])} | {r(cal['naive']['val_ece_raw'])} → {r(cal['naive']['val_ece_calibrated'])} |
| Testing ECE raw → calibrated | {r(F['ece_raw'])} → {r(F['ece_calibrated'])} | {r(FN['ece_raw'])} → {r(FN['ece_calibrated'])} |

Reliability diagram: [`reliability.png`](../v2/results/figures/reliability.png).

## The "uncertain" flag
A prediction is flagged **uncertain** when its calibrated top probability is below τ. τ is the smallest value in
0.50, 0.51, …, 0.99 for which the validation predictions at or above it are at least 99% accurate.

| | Leakage-aware | Naive |
|---|---|---|
| τ | {r(U['conf_threshold'], 2)} | {r(cal['naive']['uncertainty']['conf_threshold'], 2)} |
| Validation coverage / retained accuracy | {pc(U['val_coverage'])} / {pc(U['val_retained_accuracy'])} | {pc(cal['naive']['uncertainty']['val_coverage'])} / {pc(cal['naive']['uncertainty']['val_retained_accuracy'])} |
| Testing uncertain rate | {pc(F['uncertain_rate'])} | {pc(FN['uncertain_rate'])} |
| Testing accuracy: confident / uncertain | {pc(F['accuracy_confident'])} / {pc(F['accuracy_uncertain'])} | {pc(FN['accuracy_confident'])} / {pc(FN['accuracy_uncertain'])} |

The 99% target is reached on validation but not on Testing ({pc(F['accuracy_confident'])}): the validation images are probably closer
to the training images than some Testing images are. The flag therefore separates easier from harder
cases but is no guarantee. The normalised entropy of the probabilities is reported alongside in the prediction files.

## Where the numbers live
`v2/results/metrics.json` (all metrics), `v2/results/predictions/calibration.json` (T and τ), per-image predictions in
`v2/results/predictions/*_test_predictions.csv` and `v2/results/review/test_predictions.csv`; see the
[results catalogue](results_catalogue.md).
""")

# ================================================================ explainability
MO = ["gradcam", "gradcampp", "integrated_gradients", "occlusion", "kernel_shap"]
SN = {"classifier": "classifier only", "conv_head": "+ conv_head", "blocks.6": "+ blocks 6", "blocks.4": "+ blocks 5–4",
      "blocks.2": "+ blocks 3–2", "conv_stem": "all layers"}
w("explainability.md", f"""
# Explainability

{NAV}

Code: [`v2/src/btmri/xai.py`](../v2/src/btmri/xai.py), [`v2/scripts/run_xai.py`](../v2/scripts/run_xai.py),
[`v2/scripts/plot_xai_summary.py`](../v2/scripts/plot_xai_summary.py); notebook
[`04_explainable_ai.ipynb`](../v2/notebooks/04_explainable_ai.ipynb); results in [`v2/results/xai/`](../v2/results/xai/summary.md).

**Summary.** Heatmaps show which pixels a method associates with the prediction. They are *not* evidence that the model
uses tumour tissue, and in this study none of the five methods beats a random map on the deletion test. Treat them as a
debugging aid, not as a clinical explanation.

## Set-up
- Model: the v2 leakage-aware EfficientNet-B0, wrapped so that it takes the 1-channel 224×224 image in [0, 1] (`GreyModel`).
- Target: the **predicted** class.
- Images: {n_fa_img} Testing images (25 per true class, `sample(25, random_state=42)`) for faithfulness and agreement;
  {n_san_img} of them (3 per class) for the sanity check.
- Every map is clipped/absolute-valued as described below and min-max normalised to [0, 1].
- Run: `python scripts/run_xai.py [threads]` from `v2/` (default 8 torch threads).

## Methods
| Method | How it works | Settings in this project |
|---|---|---|
| **Grad-CAM** (Selvaraju et al., 2017) | Weights the last convolutional feature maps (7×7×1280) by the mean gradient of the class score, sums them and applies ReLU. | Last feature map, bilinear up-sampling to 224. For this network (global average pooling + linear head) the weights equal the classifier weights / 49, which is what the browser uses. |
| **Grad-CAM++** (Chattopadhay et al., 2018) | Like Grad-CAM, but weights each spatial position with higher-order gradient terms, which can highlight several regions. | Closed form for an exponential score. |
| **Integrated Gradients** (Sundararajan et al., 2017) | Averages gradients along the straight path from a baseline image to the input and multiplies by (input − baseline). | Captum, black baseline, 32 steps, absolute value. |
| **Occlusion** (Zeiler and Fergus, 2014) | Slides a black square over the image and records how much the class logit drops. | Captum, 32×32 window, stride 16, positive drops only. |
| **KernelSHAP on superpixels** (Lundberg and Lee, 2017) | Splits the image into superpixels, switches random subsets off (black) and fits a weighted linear model whose coefficients approximate Shapley values of the class logit. | SLIC with 64 segments and compactness 0.1, 200 samples, black baseline, positive values only. |
| **LIME** (Ribeiro et al., 2016) | Fits a sparse, locally weighted linear model on random superpixel perturbations. | **Not run in this project.** KernelSHAP on superpixels is the perturbation-based method that was evaluated; LIME is described here for completeness. |

### Running LIME yourself (not part of the reported results)
The same model wrapper and superpixels work with Captum's `Lime`:
```python
from captum.attr import Lime
from btmri.xai import GreyModel, superpixels
gm = GreyModel(model).eval()
seg = torch.from_numpy(superpixels(x)).long()[None, None]     # x: (1, 1, 224, 224) in [0, 1]
attr = Lime(gm).attribute(x, target=target, baselines=0.0, feature_mask=seg, n_samples=200, perturbations_per_eval=50)
```
No LIME numbers are reported anywhere in this repository.

## Faithfulness: deletion and insertion (Petsiuk et al., 2018)
Pixels are ranked by the map (ties broken randomly with a fixed seed). **Deletion** sets the top pixels to black in 32 steps
and records the predicted-class probability: a faithful map should make it drop fast (low area under the curve).
**Insertion** starts from a Gaussian-blurred image (kernel 31, σ = 10) and restores the top pixels: the probability should
rise fast (high area). A random map is the reference. Values: mean (standard deviation) over {n_fa_img} images.

| Method | Deletion AUC ↓ | Insertion AUC ↑ | Share of map inside the foreground |
|---|---|---|---|
""" + "\n".join(f"| {MN[k]} | {r(fm.loc[k, 'deletion_auc'])} ({r(fs.loc[k, 'deletion_auc'], 2)}) | {r(fm.loc[k, 'insertion_auc'])} ({r(fs.loc[k, 'insertion_auc'], 2)}) | {pc(fm.loc[k, 'inside_foreground'])} |" for k in MO + ["random"]) + f"""

- **Deletion:** no method is better than random; Integrated Gradients is level with it ({r(fm.loc['integrated_gradients', 'deletion_auc'])}
  vs {r(fm.loc['random', 'deletion_auc'])}) and the others are worse. Blacking out pixels creates out-of-distribution images,
  so this test is hard to interpret for MRI, but it gives no support to the maps.
- **Insertion:** every method beats random; Grad-CAM and Grad-CAM++ are best.
- **Foreground share:** the fraction of map mass on pixels brighter than 0.08 after a 7×7 box blur (a crude head mask,
  not a tumour mask). Higher than random for all methods, so the maps mostly avoid the black background.

Figure: [`xai_faithfulness.png`](../v2/results/figures/xai_faithfulness.png).

## Sanity check: cascading randomisation (Adebayo et al., 2018)
Layers are re-initialised from the classifier downwards and each new map is compared with the original by Spearman rank
correlation (maps average-pooled to 56×56). A method that depends on the learned weights should drop towards 0.

| Method | """ + " | ".join(SN[s] for s in stages) + " |\n|---|" + "---|" * len(stages) + "\n" + "\n".join(
    f"| {MN[k]} | " + " | ".join(r(sm[(k, s)], 2) for s in stages) + " |" for k in MO) + f"""

- **Grad-CAM, occlusion and KernelSHAP** change almost completely once the classifier is randomised: they depend on the model.
- **Integrated Gradients** stays at {r(sm[('integrated_gradients', 'conv_stem')], 2)} even with all layers random, so it mostly reflects the
  input image (edges and intensity), not the trained model: it fails the test.
- **KernelSHAP** keeps a small correlation ({r(sm[('kernel_shap', 'conv_stem')], 2)}) after full randomisation because its superpixels depend on the image.
- **Grad-CAM++** stays high ({r(sm[('gradcampp', 'classifier')], 2)}) after the classifier is randomised and only falls when the convolutional layers are.
- {const_gc} of {rows_gc} Grad-CAM rows produced a constant (all-zero after ReLU) map on the randomised model; their correlation is set to 0.

Figure: [`xai_sanity_check.png`](../v2/results/figures/xai_sanity_check.png).

## Agreement between methods
Mean Spearman correlation between maps on the same image ({n_fa_img} images):

| | """ + " | ".join(MN[k] for k in MO) + " |\n|---|" + "---|" * len(MO) + "\n" + "\n".join(
    f"| {MN[a]} | " + " | ".join(r(ag.loc[a, b], 2) for b in MO) + " |" for a in MO) + f"""

Only the two CAM methods agree closely; the perturbation and gradient methods highlight different pixels. Figure:
[`xai_agreement.png`](../v2/results/figures/xai_agreement.png).

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
""")

# ================================================================ web demo
pre = "\n".join(f"| `{Path(p['file']).name}` | {LAB[p['true']]} | {LAB[p['torch_pred']]} ({pc(max(p['torch_probs']))}) |" for p in presets)
w("web_demo.md", f"""
# Web demo (NEURO//SCAN) and ONNX export

{NAV}

Live: **{PAGES}** · source: [`docs/`](../docs/) (served by GitHub Pages from `main:/docs`) · notebook
[`05_onnx_export_and_web_demo.ipynb`](../v2/notebooks/05_onnx_export_and_web_demo.ipynb).

{DISCLAIMER}

The demo serves the **v2** leakage-aware model. The frozen v1 model is not deployed; v1 code and results are in
[`v1/`](../v1/README.md) and at the tag [`v1.0`]({GH}/tree/v1.0).

## Architecture
Everything runs in the browser; uploaded images never leave the device. There is no server-side code.

| File | Role |
|---|---|
| [`docs/index.html`](../docs/index.html) | Page layout, About section, the v1 vs v2 note; loads ONNX Runtime Web 1.20.1 (WebAssembly) from jsDelivr |
| [`docs/app.js`](../docs/app.js) | Image loading, pre-processing on a canvas, inference, heatmap drawing, browser occlusion, presets |
| [`docs/logic.js`](../docs/logic.js) | Shared pure functions (softmax with temperature, normalised entropy, CAM extraction, up-sampling, foreground mask, the templated rationale); also tested in Node by [`tests/test_logic.js`](../tests/test_logic.js) |
| [`docs/model/brain_tumor_effnetb0.onnx`](../docs/model/brain_tumor_effnetb0.onnx) | The exported model ({r(onx['onnx_size_mb'], 1)} MB) |
| [`docs/stats.json`](../docs/stats.json) | Temperature, threshold τ and measured test statistics used by the "why" panel (written by `make_report.py`) |
| [`docs/presets.json`](../docs/presets.json) + [`docs/samples/`](../docs/samples/) | Eight Testing images (unmodified originals) with saved PyTorch predictions and precomputed Grad-CAM++, IG and KernelSHAP maps (56×56) |
| [`docs/style.css`](../docs/style.css) | Styling |

**Inference path.** File → greyscale → zero-pad to square → bilinear 224×224 (canvas) → float32 in [0, 1] →
`InferenceSession.run` → `logits` (1×4) and `cam` (1×4×7×7) → softmax(logits / T) with T = {r(cal['clean']['temperature'], 3)} →
"uncertain" if the top probability is below τ = {r(U['conf_threshold'], 2)}. ImageNet normalisation is inside the ONNX graph.

**Heatmaps.**
- *Grad-CAM*: ReLU of the predicted class's CAM, min-max normalised and up-sampled; exact, because for this network
  Grad-CAM equals CAM up to a positive scale.
- *Occlusion*: a 32-pixel black square at stride 16 (13 × 13 = 169 forward passes) in batches of 4; the page yields to the
  browser after each batch so it stays responsive (about 9–12 seconds in our headless-Chrome tests).
- *Grad-CAM++, Integrated Gradients, KernelSHAP*: precomputed in Python for the presets only.

**"Why this prediction" panel.** Template text that uses only the model's outputs (probabilities, entropy, where the map
mass lies) and the measured statistics in `stats.json` (per-class recall and precision, uncertain-rate accuracy). It does not
describe anatomy.

## ONNX export and verification
[`v2/scripts/export_onnx.py`](../v2/scripts/export_onnx.py) wraps the model in `ExportWrapper` (input `image`, 1×1×224×224
in [0, 1]; outputs `logits` and `cam`; dynamic batch; opset 17), writes `v2/models/brain_tumor_effnetb0.onnx`, copies it
to `docs/model/`, and compares ONNX Runtime with PyTorch on 64 Testing images:

| Check | Result |
|---|---|
| Max absolute logit difference | {onx['max_abs_logit_diff_torch_vs_onnx']:.2g} |
| Arg-max agreement | {pc(onx['argmax_agreement'])} |
| Max absolute CAM difference | {onx['max_abs_cam_diff_torch_vs_onnx']:.2g} |
| ONNX CAM vs autograd Grad-CAM (normalised, 8 images) | {onx['max_abs_diff_onnx_cam_vs_autograd_gradcam_normalised']:.2g} |

## Presets
Chosen by [`v2/scripts/make_presets.py`](../v2/scripts/make_presets.py): two per class, drawn with a fixed seed from the
{pool['n']} Testing images that have **no candidate pair at all at Hamming ≤ 10**
(whatever the correlation) and are not `-aug-` files. Files are copied byte-for-byte.

| File | True class | Saved prediction (calibrated) |
|---|---|---|
{pre}

`Te-gl_391` is kept on purpose: it is the one misclassified preset and is marked with a "misclassified" badge.
[`v2/scripts/check_samples.py`](../v2/scripts/check_samples.py) and [`tests/check_samples_browser.py`](../tests/check_samples_browser.py)
check that the presets give the saved predictions (within 0.01, same uncertain flag) in ONNX Runtime and in the browser.

## Run the demo locally
```bash
cd docs
python3 -m http.server 8765
# open http://localhost:8765/
```
Opening `index.html` directly from disk does not work, because browsers block `fetch()` of the model and JSON files from
`file://` URLs. The page needs internet access for the ONNX Runtime script from jsDelivr.
""")

# ================================================================ changelog
assert (CH["changed"], CH["c2w"], CH["w2c"], CH["w2w"]) == (79, 38, 14, 27), CH
v1v2 = (ROOT / "results/v1_vs_v2.md").read_text().strip().replace("v2 (this version)", "v2")
w("changelog.md", f"""
# v1 vs v2 and changelog

{NAV}

## Two versions in one repository
| | Folder | Code at | Status |
|---|---|---|---|
| **v2** | [`v2/`](../v2/README.md) | `main` (tags `v2.0` … `v2.1.0`) | current; served by the [demo]({PAGES}) |
| **v1** | [`v1/`](../v1/README.md) | tag [`v1.0`]({GH}/tree/v1.0) (commit `a591cfa`) | frozen; results, notebooks, paper and code kept for comparison |

## Results side by side
Generated by [`v2/scripts/make_v1_v2.py`](../v2/scripts/make_v1_v2.py) from the saved metric files of both versions.

{v1v2}

## Why v2 is preferred
v2 links {n(d['extra_pairs_v2_rule'])} more pairs ({n(d['extra_cross_split_pairs_v2_rule'])} across Training/Testing), removes
{n(d['train_imgs_removed_cross_split'])} Training images instead of {n(d1['train_imgs_removed_cross_split'])}, and afterwards no Testing
image has a training or validation partner with correlation ≥ 0.98 within Hamming distance 10. It is preferred because its
split is cleaner, not because of its score.

Accuracy goes from {pc(m1['clean']['full_test']['accuracy'])} to {pc(F['accuracy'])}. The 19 Testing images whose near-copies v2
removes are classified correctly by both versions, so the drop does not come from them. {CH['changed']} predictions change
between the versions ({CH['c2w']} from correct to wrong, {CH['w2c']} from wrong to correct and {CH['w2w']} from one wrong class to
another; almost all gliomas and meningiomas). The training pool loses only 25 images and the group-wise train/validation
assignment is redrawn, so the difference mainly reflects run-to-run variance of a single training seed, which the confidence
intervals do not capture.

## Changelog
### v2.1.0
- **Repository restructured** into [`v1/`](../v1/README.md) and [`v2/`](../v2/README.md), each with its own scripts, source,
  results, notebooks, paper, README and requirements; shared [`docs/`](../docs/) (demo), [`tests/`](../tests/) and this
  [`documentation/`](README.md) at the root. Script paths, notebook set-up cells and Colab badges updated; v1 notebooks clone
  the `v1.0` tag. `v2/results/v1/` was replaced by the full `v1/results/`.
- **Documentation set** (this folder), generated from the result files by `v2/scripts/build_documentation.py`, including
  [limitations](limitations.md), [future work](future_work.md) and [references](references.md).
- Root files: [`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md), [`CONTRIBUTING.md`](../CONTRIBUTING.md),
  [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md), [`CITATION.cff`](../CITATION.cff), [`references.bib`](../references.bib),
  GitHub issue and pull-request templates.
- Demo: note that the page serves v2, with links to `v1/` and the `v1.0` tag.
- No retraining; no model, metric or prediction changed.

### Untagged (`3b74834`)
- Demo: inline SVG favicon, removing the browser's `/favicon.ico` 404 console error.

### v2.0.3 (`9366e05`)
- Demo: "misclassified" badge on preset tiles whose saved prediction differs from the true label.

### v2.0.2 (`ddb6803`)
- Demo presets are the original Testing files, copied byte-for-byte (earlier presets were resized and re-encoded, which changed the
  displayed confidence and uncertain flag of `Te-gl_391`). `Te-gl_391` now shows its saved prediction (no tumour, uncertain).
- `check_samples.py`, `tests/test_samples.py` and `tests/check_samples_browser.py` compare preset predictions with the
  prediction file (within 0.01, same uncertain flag).
- Browser occlusion yields to the event loop after every batch of 4, so the page stays responsive.

### v2.0.1 (`d04eef0`)
- Text-only fixes: the v1→v2 explanation with verified counts, wording of the deletion results and other review items.

### v2.0 (`c713a76`, `de37f33`)
- Stricter duplicate rule (adds Hamming ≤ 10 with correlation ≥ 0.98); leakage-aware model retrained with identical settings.
- 1,000 bootstrap resamples for every row; looser-rule sensitivity check; strict-duplicate macro-F1 note.
- XAI re-run with random tie-breaking in deletion/insertion and a constant-map flag in the sanity check.
- Stricter presets (no candidate pair at Hamming ≤ 10, no augmented files).
- Paper and README review fixes; v1 vs v2 table; demo fixes (samples disabled until the model loads, non-image message,
  occlusion stride 16).

### v1.0 (`a591cfa`)
- First release: duplicate audit, leakage-aware EfficientNet-B0, naive comparison, calibration, XAI study, notebooks, web demo,
  white paper.
""")

# ================================================================ repository structure
w("repository_structure.md", f"""
# Repository structure

{NAV}

```text
brain-tumor-mri-classification/
├── README.md                 overview, links to v1/, v2/ and this documentation
├── documentation/            this documentation set (generated by v2/scripts/build_documentation.py)
├── docs/                     GitHub Pages web demo (NEURO//SCAN), serves the v2 model
│   ├── index.html, app.js, logic.js, style.css
│   ├── model/brain_tumor_effnetb0.onnx
│   ├── stats.json, presets.json, samples/      (written by v2 scripts)
│   └── screenshot.png, screenshot_mobile.png
├── tests/                    pytest + Node tests for the v2 code and the demo
├── v2/                       current version (tags v2.0 … v2.1.0)
│   ├── README.md, requirements.txt
│   ├── src/btmri/            library: data, dedup, splits, model, train, calibrate, xai
│   ├── scripts/              pipeline scripts (run from v2/)
│   ├── notebooks/            five Colab-ready notebooks, executed with outputs
│   ├── results/              figures, metrics, predictions, review files, runs, XAI results
│   ├── models/               brain_tumor_effnetb0.pt and .onnx
│   └── paper/                white paper (.tex, .pdf), references.bib, numbers.tex, tables/
├── v1/                       frozen first version (code runs at tag v1.0)
│   └── README.md, requirements.txt, src/, scripts/, notebooks/, results/, models/, paper/
├── REPRODUCIBILITY.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, CITATION.cff, references.bib, LICENSE
└── .github/                  issue and pull-request templates
```

## Conventions
- **Run v2 scripts from `v2/`** (`python scripts/<name>.py`); they locate their files relative to the script, so the
  working directory matters only for convenience. Scripts that update the demo write into the shared `docs/` folder.
- **The dataset is not committed.** `kagglehub` downloads it to its own cache; pre-processed arrays go to `cache/` at the
  repository root (git-ignored).
- **Generated files.** `v2/README.md` (from `v2/scripts/README_template.md`), `v2/paper/numbers.tex`, `v2/paper/tables/*.tex`,
  `v2/results/**`, `docs/stats.json`, `docs/presets.json` and this documentation folder are generated; edit the generator,
  not the output.
- **v1 is frozen.** `v1/` is the `v1.0` tree with only its links and notebook set-up cells updated for the new location. Its
  scripts expect the old layout, so run them from a checkout of the tag: `git checkout v1.0`.
- **Tests** at the root test v2: `pytest -q tests` and `node tests/test_logic.js`. v1's tests are in the `v1.0` tag.
""")

# ================================================================ scripts reference
SCR = [
    ("scripts/build_splits.py", "—", "Downloads the dataset (kagglehub), hashes every image, finds candidate pairs, builds duplicate groups and the leakage-aware split.",
     "`results/review/hashes_and_groups.csv`, `duplicate_pairs.csv`, `candidate_pairs_hamming_le10.csv.gz`, `{{train,val,test,excluded}}_files.csv`, `dedup_summary.json`"),
    ("src/btmri/train.py", "`--run clean|naive` (default `clean`), `--epochs 12`, `--bs 32`, `--lr 5e-4`, `--seed 42`",
     "Trains EfficientNet-B0. Run as `cd src && python -m btmri.train --run clean`.", "`results/runs/<run>/best.pt`, `history.csv`, `val_logits.csv`, `test_logits.csv`"),
    ("scripts/make_predictions.py", "—", "Fits the temperature and threshold τ on validation logits and writes the per-image test predictions (the single source of truth).",
     "`results/predictions/{{clean,naive}}_test_predictions.csv`, `calibration.json`"),
    ("scripts/make_report.py", "—", "Computes every metric (with bootstrap CIs), the leakage-by-class table and the looser-rule sensitivity check; draws evaluation figures.",
     "`results/metrics.json`, `results/review/test_predictions.csv`, `results/figures/{{confusion_matrix,roc_curves,reliability,training_curves_*}}.png`, `paper/tables/*.tex`, `docs/stats.json`"),
    ("scripts/audit_augmented.py", "—", "Matches every `*-aug-*` file against non-augmented images under 8 flips/rotations and compares accuracy.",
     "`results/review/augmented_meningioma_matches.csv`, `augmented_summary.json`"),
    ("scripts/make_dedup_figures.py", "—", "Draws example duplicate pairs and per-class leakage counts.", "`results/figures/dedup_examples.png`, `dedup_per_class.png`"),
    ("scripts/export_onnx.py", "—", "Exports the model with logits + CAM outputs to ONNX and verifies it against PyTorch.",
     "`models/brain_tumor_effnetb0.onnx` and `.pt`, `docs/model/brain_tumor_effnetb0.onnx`, `results/onnx_verification.json`"),
    ("scripts/run_xai.py", "optional positional: number of torch threads (default 8)", "Runs the XAI study (faithfulness, foreground share, agreement, sanity check) and draws the XAI figures.",
     "`results/xai/*.csv`, `results/figures/xai_*.png`"),
    ("scripts/plot_xai_summary.py", "—", "Redraws the XAI summary figures from the saved CSVs.", "`results/figures/xai_{{sanity_check,faithfulness,agreement}}.png`"),
    ("scripts/xai_summary_md.py", "—", "Writes the XAI summary text used by the README.", "`results/xai/summary.md`"),
    ("scripts/make_presets.py", "—", "Selects the eight demo presets (no candidate pair at Hamming ≤ 10, no `-aug-` files, seed 42), copies the files byte-for-byte and precomputes Grad-CAM++, IG and KernelSHAP maps.",
     "`docs/presets.json`, `docs/samples/*`, `results/review/preset_pool.json`"),
    ("scripts/check_samples.py", "—", "Runs the shipped ONNX model on each preset and checks it against the prediction file (each probability within 0.01, same uncertain flag); exits non-zero on failure.", "console report"),
    ("scripts/make_paper_numbers.py", "—", "Writes every number used in the paper as a LaTeX macro, plus the XAI table.", "`paper/numbers.tex`, `paper/tables/xai.tex`"),
    ("scripts/make_v1_v2.py", "—", "Builds the v1 vs v2 table from `../v1/results/` and the current results; updates the note in the demo page.",
     "`results/v1_vs_v2.md`, `v1_vs_v2.json`, `paper/tables/v1v2.tex`, `docs/index.html` (between `<!--V1V2-->` markers)"),
    ("scripts/fill_readme.py", "—", "Fills `v2/README.md` from `scripts/README_template.md` and the result files.", "`README.md`"),
    ("scripts/build_notebooks.py", "—", "Generates the five notebooks (executed afterwards with `jupyter nbconvert --execute`).", "`notebooks/*.ipynb`"),
    ("scripts/build_documentation.py", "—", "Generates this documentation set from the result files.", "`../documentation/*.md`"),
]
MODS = [
    ("data", "`seed_everything(seed)`, `download_dataset()`, `list_images(root)`, `load_gray_square(path, size)` (greyscale, zero-pad, bilinear resize → uint8), `build_cache(recs, size, cache_file)`"),
    ("dedup", "`file_md5(path)`, `phash64(path)`, `hamming_matrix(a, b)`, `duplicate_groups(hashes, md5s, threshold)`"),
    ("splits", "`hash_table(root)`, `candidate_pairs(df, max_hamming=10)` (adds Hamming, correlation and the strict/linked flags), `assign_groups(df, pairs, col)`, `make_splits(df, pairs, seed=42)`"),
    ("model", "`build_model(pretrained=True, num_classes=4)`, `to_input(gray_uint8)` (→ normalised 3-channel float), `ExportWrapper` (logits + CAM)"),
    ("train", "`load_split_table(run, seed)`, `predict(model, X)`, `main()` (command-line entry point)"),
    ("calibrate", "`fit_temperature(logits, y)`, `softmax(logits, t)`, `norm_entropy(p)`, `ece(p, y, bins=15)`, `choose_threshold(p_val, y_val, target_acc=0.99)`"),
    ("xai", "`GreyModel`, `gradcam_family(gm, x, target, plus)`, `integrated_gradients`, `occlusion`, `superpixels`, `kernel_shap`, `explain(gm, x, target, method)`, `cascading_randomisation`, `rank_corr`, `blur`, `deletion_insertion`, `brain_mask`, `attention_summary`"),
]
w("scripts_reference.md", f"""
# Script and API reference

{NAV}

All paths are relative to `v2/`. Run scripts from `v2/` with the virtual environment active. The order below is the
pipeline order; see the [reproducibility guide](../REPRODUCIBILITY.md) for the full sequence.

## Scripts
| Script | Arguments | What it does | Writes |
|---|---|---|---|
""" + "\n".join(f"| [`{s}`](../v2/{s}) | {a} | {wd} | {o} |" for s, a, wd, o in SCR) + f"""

## Library modules (`v2/src/btmri/`)
| Module | Main functions and classes |
|---|---|
""" + "\n".join(f"| [`{k}.py`](../v2/src/btmri/{k}.py) | {v} |" for k, v in MODS) + f"""

Rule constants are defined at the top of [`splits.py`](../v2/src/btmri/splits.py) (strict, v1 and v2 thresholds).

## Tests (repository root)
| Command | What it checks |
|---|---|
| `pytest -q tests` | `test_core.py`: Hamming matrix vs. Python popcount, union-find grouping, temperature and threshold fitting, no duplicate group crossing train/val/test; `test_samples.py`: runs `check_samples.py` |
| `node tests/test_logic.js` | the browser logic in `docs/logic.js` (softmax, entropy, CAM, rationale) |
| `python tests/check_samples_browser.py [URL]` | opens the demo in headless Chrome (Playwright), runs every preset and checks the probabilities and flags; default URL `http://localhost:8000/` |
""")

# ================================================================ results catalogue
DESC = {
    "metrics.json": "All evaluation metrics: `clean` and `naive` × {full_test, dedup_test, test_with_strict_train_dup, test_without_linked_train_dup} (accuracy, macro-F1, per-class P/R/F1, confusion matrix, ECE, AUC, CIs, uncertain rates), `leakage_by_class`, `sensitivity_looser_rule`, calibration summary.",
    "onnx_verification.json": "ONNX vs PyTorch: max logit difference, arg-max agreement, CAM difference, ONNX CAM vs autograd Grad-CAM, file size.",
    "v1_vs_v2.json": "The v1 vs v2 comparison rows (metric, v1, v2).",
    "v1_vs_v2.md": "The same table in Markdown (used in the READMEs and the changelog).",
    "figures/confusion_matrix.png": "Confusion matrix of the leakage-aware model on the Testing folder (counts and row percentages).",
    "figures/dedup_examples.png": "Example cross-split near-duplicate pairs with their Hamming distance and correlation.",
    "figures/dedup_per_class.png": "Testing images with a strict / linked duplicate in Training, per class.",
    "figures/reliability.png": "Reliability diagram of the leakage-aware model, uncalibrated vs temperature-scaled (plot uses 10 bins with n ≥ 5; ECE in the legend uses 15).",
    "figures/roc_curves.png": "One-vs-rest ROC curves per class, leakage-aware model.",
    "figures/training_curves_clean.png": "Loss and validation metrics per epoch, leakage-aware run.",
    "figures/training_curves_naive.png": "Loss and validation metrics per epoch, naive run.",
    "figures/xai_agreement.png": "Spearman agreement matrix between attribution methods.",
    "figures/xai_faithfulness.png": "Deletion and insertion AUC per method vs random.",
    "figures/xai_grid_correct.png": "Attribution maps for correctly classified Testing images.",
    "figures/xai_grid_misclassified.png": "Attribution maps for the most confident errors.",
    "figures/xai_sanity_check.png": "Cascading-randomisation curves per method.",
    "predictions/calibration.json": "Temperature, validation ECE before/after, threshold τ with its rule, validation coverage and accuracy, for both runs.",
    "predictions/clean_test_predictions.csv": "Leakage-aware model: one row per Testing image with true class, prediction, calibrated and raw probabilities, confidence, normalised entropy, uncertain flag. **Single source of truth** for every metric.",
    "predictions/naive_test_predictions.csv": "The same for the naive model.",
    "review/augmented_meningioma_matches.csv": "Every `*-aug-*` file with its closest non-augmented match, transform, Hamming distance, correlation and whether it is linked.",
    "review/augmented_summary.json": "Counts and accuracies from the augmented-file audit.",
    "review/candidate_pairs_hamming_le10.csv.gz": "Every image pair within Hamming 10: paths, Hamming, correlation, MD5 equality, cross-split, strict / linked_v1 / linked flags.",
    "review/dedup_summary.json": "Counts from the duplicate audit and split (pairs, groups, removed images, subset sizes per class).",
    "review/duplicate_pairs.csv": "The linked pairs only.",
    "review/excluded_files.csv": "Training-folder images removed from training (cross-split groups and label conflicts).",
    "review/hashes_and_groups.csv": "Every image: path, folder, class, MD5, pHash, linked and strict group ids, subset, and leakage flags.",
    "review/preset_pool.json": "Size of the demo preset pool per class and the chosen presets.",
    "review/test_files.csv": "Testing images (the official folder).",
    "review/test_predictions.csv": "Testing predictions of the leakage-aware model joined with the duplicate flags; used by the demo checks and review.",
    "review/train_files.csv": "Training images of the leakage-aware split.",
    "review/val_files.csv": "Validation images of the leakage-aware split.",
    "runs/clean/best.pt": "Leakage-aware checkpoint (PyTorch state dict); identical to `v2/models/brain_tumor_effnetb0.pt`.",
    "runs/clean/history.csv": "Per-epoch train loss, validation loss/accuracy/macro-F1, learning rate and seconds.",
    "runs/clean/test_logits.csv": "Raw logits on the Testing images.",
    "runs/clean/val_logits.csv": "Raw logits on the validation images (used to fit T and τ).",
    "runs/naive/best.pt": "Naive checkpoint.",
    "runs/naive/history.csv": "Per-epoch log, naive run.",
    "runs/naive/test_logits.csv": "Raw logits on the Testing images, naive run.",
    "runs/naive/val_logits.csv": "Raw logits on the naive validation images.",
    "xai/faithfulness_per_image.csv": "Per image and method: deletion AUC, insertion AUC, foreground share.",
    "xai/faithfulness_summary.csv": "Mean and standard deviation of the above per method.",
    "xai/method_agreement_spearman.csv": "Mean Spearman correlation between methods (5 × 5).",
    "xai/sanity_cascading_randomisation.csv": "Per stage, method and image: Spearman correlation with the trained-model map, constant-map flag.",
    "xai/summary.md": "Text summary of the XAI results used in the README.",
}
files = sorted(p.relative_to(ROOT / "results").as_posix() for p in (ROOT / "results").rglob("*") if p.is_file())
missing = [f for f in files if f not in DESC]
assert not missing, missing
assert not [k for k in DESC if k not in files], [k for k in DESC if k not in files]


def info(f):
    p = ROOT / "results" / f
    sz = p.stat().st_size
    s = f"{sz / 1e6:.1f} MB" if sz >= 1e6 else f"{sz / 1e3:.1f} kB"
    if f.endswith((".csv", ".csv.gz")):
        t = pd.read_csv(p)
        s += f"; {n(len(t))} rows"
    return s


sec = {}
for f in files:
    sec.setdefault(f.split("/")[0] if "/" in f else "(top level)", []).append(f)
body = "\n\n".join(f"## `{k}`\n| File | Size | Contents |\n|---|---|---|\n" + "\n".join(
    f"| [`{f}`](../v2/results/{f}) | {info(f)} | {DESC[f]} |" for f in v) for k, v in sec.items())
tp_cols = ", ".join(f"`{c}`" for c in pd.read_csv(ROOT / "results/review/test_predictions.csv", nrows=1).columns)
hg_cols = ", ".join(f"`{c}`" for c in pd.read_csv(ROOT / "results/review/hashes_and_groups.csv", nrows=1).columns)
cp_cols = ", ".join(f"`{c}`" for c in cand.columns)
w("results_catalogue.md", f"""
# Results catalogue

{NAV}

Every file in [`v2/results/`](../v2/results/) ({len(files)} files). The v1 results are in [`v1/results/`](../v1/results/)
with the same layout as at the `v1.0` tag.

{body}

## Column reference for the main tables
- `review/test_predictions.csv`: {tp_cols}. `p_*` are calibrated probabilities (logits / T), `raw_p_*` uncalibrated.
- `review/hashes_and_groups.csv`: {hg_cols}.
- `review/candidate_pairs_hamming_le10.csv.gz`: {cp_cols}.
""")

# ================================================================ references
URLS = {"tan2019": "https://proceedings.mlr.press/v97/tan19a.html", "sundararajan2017": "https://proceedings.mlr.press/v70/sundararajan17a.html",
        "guo2017": "https://proceedings.mlr.press/v70/guo17a.html", "lundberg2017": "https://arxiv.org/abs/1705.07874",
        "adebayo2018": "https://arxiv.org/abs/1810.03292", "petsiuk2018": "https://arxiv.org/abs/1806.07421",
        "paszke2019": "https://arxiv.org/abs/1912.01703", "kokhlikyan2020": "https://arxiv.org/abs/2009.07896",
        "zauner2010": "https://www.phash.org/docs/pubs/thesis_zauner.pdf", "wightman2019": "https://github.com/huggingface/pytorch-image-models",
        "nickparvar2021": "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"}
TEX = [(r"{\v{z}}", "ž"), (r"{\v{C}}", "Č"), (r"{\'c}", "ć"), (r'{\"u}', "ü"), (r'{\"o}', "ö"), ("``", "“"), ("''", "”"), ("--", "–"), ("\\&", "&")]


def detex(s):
    for a, b in TEX:
        s = s.replace(a, b)
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    return s.replace("{", "").replace("}", "").strip()


def parse_bib(text):
    out = []
    for m_ in re.finditer(r"@(\w+)\{(\w+),(.*?)\}\s*(?=@|\Z)", text, re.S):
        typ, key, body = m_.groups()
        f = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", body):
            f[fm.group(1)] = fm.group(2)
        out.append((typ, key, f))
    return out


BIB = parse_bib((ROOT / "paper/references.bib").read_text())
assert len(BIB) == 28, len(BIB)


def authors(a):
    ps = [detex(x) for x in a.split(" and ")]
    ps = ["et al." if p == "others" else p for p in ps]
    if ps[-1] == "et al.":
        return "; ".join(ps[:-1]) + "; et al."
    return ps[0] if len(ps) == 1 else "; ".join(ps[:-1]) + " and " + ps[-1]


def fmt(typ, key, f):
    s = f"{authors(f['author'])} ({f['year']}). {detex(f['title'])}."
    venue = f.get("journal") or f.get("booktitle") or f.get("school") or f.get("howpublished")
    if venue:
        s += f" *{detex(venue)}*"
        if f.get("volume"):
            s += f", {detex(f['series']) + ' ' if f.get('series') else ''}{f['volume']}" + (f"({f['number']})" if f.get("number") else "")
        if f.get("pages"):
            s += f", {detex(f['pages'])}"
        s += "."
    if f.get("note"):
        s += f" {detex(f['note'])}."
    if f.get("doi"):
        s += f" DOI: [{f['doi']}](https://doi.org/{f['doi']})"
    elif key in URLS and URLS[key] not in s:
        s += f" <{URLS[key]}>"
    return s


refs = sorted(BIB, key=lambda t: (detex(t[2]["author"]).split(",")[0].lower(), t[2]["year"]))
w("references.md", f"""
# References

{NAV}

## Bibliography of the white paper
The {len(BIB)} entries of [`v2/paper/references.bib`](../v2/paper/references.bib), in alphabetical order, with a DOI where one
exists and otherwise a stable URL (proceedings page, arXiv record or project page). The BibTeX file at the repository root,
[`references.bib`](../references.bib), contains the same entries plus software citations.

""" + "\n".join(f"{i}. {fmt(*t)} `[{t[1]}]`" for i, t in enumerate(refs, 1)) + f"""

## Dataset
- Nickparvar, M. *Brain Tumor MRI Dataset*, version 2 (13 February 2026), Kaggle, CC BY 4.0:
  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Its sources, as stated in the dataset description: figshare brain tumour dataset (Cheng, 2017; DOI [10.6084/m9.figshare.1512427.v5](https://doi.org/10.6084/m9.figshare.1512427.v5), described in Cheng et al., 2015), SARTAJ (Bhuvaji et al., 2020; https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri,
  DOI [10.34740/KAGGLE/DSV/1183165](https://doi.org/10.34740/KAGGLE/DSV/1183165))
  and Br35H (https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection).

## Software
| Package | Used for | Link |
|---|---|---|
| PyTorch (Paszke et al., 2019) | training and inference | https://pytorch.org |
| timm (Wightman, 2019) | EfficientNet-B0 weights | https://github.com/huggingface/pytorch-image-models |
| Captum (Kokhlikyan et al., 2020) | Integrated Gradients, occlusion, KernelSHAP | https://captum.ai |
| scikit-learn (Pedregosa et al., 2011) | metrics | https://scikit-learn.org · https://jmlr.org/papers/v12/pedregosa11a.html |
| scikit-image (van der Walt et al., 2014) | SLIC superpixels | https://scikit-image.org · DOI [10.7717/peerj.453](https://doi.org/10.7717/peerj.453) |
| ImageHash | pHash | https://github.com/JohannesBuchner/imagehash |
| kagglehub | dataset download | https://github.com/Kaggle/kagglehub |
| ONNX / ONNX Runtime (Web) | export and in-browser inference | https://onnx.ai · https://github.com/microsoft/onnxruntime · https://onnxruntime.ai/docs/tutorials/web/ |

## Citing this project
See [`CITATION.cff`](../CITATION.cff) (GitHub shows a "Cite this repository" button) or use the `aryan2026brainmri` entry in
[`references.bib`](../references.bib). Part 2 of that file also has BibTeX entries for the dataset sources and the software above.
""")

# ================================================================ model card
w("model_card.md", f"""
# Model card: EfficientNet-B0 brain tumour MRI classifier (v2, leakage-aware)

{NAV}

{DISCLAIMER}

## Model details
| | |
|---|---|
| Developer | Rishabh Aryan (M.Tech in AI and Data Science, IIIT Bhagalpur) |
| Version | v2 (trained for tag `v2.0`; unchanged in v2.1.0) |
| Type | Image classifier, EfficientNet-B0 (`timm` `efficientnet_b0.ra_in1k`) fine-tuned from ImageNet |
| Input | One 2-D MRI slice, converted to greyscale, zero-padded to square, resized to 224×224 |
| Output | Probabilities for glioma, meningioma, no tumour, pituitary tumour (temperature T = {r(cal['clean']['temperature'], 2)}), an "uncertain" flag (top probability < {r(U['conf_threshold'], 2)}), Grad-CAM map |
| Files | [`v2/models/brain_tumor_effnetb0.pt`](../v2/models/brain_tumor_effnetb0.pt) (PyTorch), [`v2/models/brain_tumor_effnetb0.onnx`](../v2/models/brain_tumor_effnetb0.onnx) (ONNX, logits + CAM) |
| Licence | Repository licence MIT ([LICENSE](../LICENSE)); the training data is CC BY 4.0 |
| Paper | [`v2/paper/brain_tumor_mri_white_paper.pdf`](../v2/paper/brain_tumor_mri_white_paper.pdf) |

## Intended use
- Teaching and research on **evaluation hygiene** (duplicate leakage, calibration, explanation sanity checks) in medical image
  classification.
- A worked, reproducible example and an in-browser demo.

## Out-of-scope use
- **Any clinical use**: diagnosis, triage, screening, treatment planning or second opinions. The model has not been validated
  on clinical data, is not a medical device and has no regulatory approval.
- Images other than single 2-D brain MRI slices similar to the training data (other organs, CT, 3-D volumes, photos of screens).
- Tumour types outside the three classes (e.g. metastases, lymphoma), grading, or segmentation. The model always answers with
  one of the four classes.

## Training data
{n(d['subset_counts']['train'])} training and {n(d['subset_counts']['val'])} validation images from the Training folder of the Kaggle
*Brain Tumor MRI Dataset* v2, after removing the {n(d['train_imgs_removed_cross_split'])} Training-folder images linked to the Testing folder ([data card](data_card.md), [split method](dedup_and_splits.md)).

## Evaluation (official Testing folder, n = {n(F['n'])})
| Metric | Value |
|---|---|
| Accuracy (95% CI) | {ci(F)} |
| Macro-F1 / macro AUC | {r(F['macro_f1'])} / {r(F['roc_auc_macro_ovr'])} |
| Recall: glioma / meningioma / no tumour / pituitary | {" / ".join(pc(F['per_class'][c]['recall']) for c in CL)} |
| ECE after temperature scaling | {r(F['ece_calibrated'])} |
| Uncertain rate; accuracy of confident / uncertain predictions | {pc(F['uncertain_rate'])}; {pc(F['accuracy_confident'])} / {pc(F['accuracy_uncertain'])} |

More in [evaluation and calibration](evaluation_and_calibration.md).

## Known limitations and failure modes
- Gliomas and meningiomas are most often mistaken for **no tumour** ({cm[0][2]} and {cm[1][2]} of 400 each); a "no tumour" answer
  is the least reliable (precision {r(F['per_class']['notumor']['precision'])}).
- The no-tumour images come from a different source than the tumour images, so the model may use source cues.
- One training seed, CPU training, no external test set; see [limitations](limitations.md).
- Heatmaps are not validated localisation; Integrated Gradients fails the randomisation sanity check ([explainability](explainability.md)).

## Ethical considerations
- **Patient data.** The dataset is public and de-identified, without patient IDs or demographics, so performance across age,
  sex, scanner or hospital cannot be assessed and fairness cannot be checked.
- **Risk of over-trust.** High accuracy on a benchmark with near-duplicates can be mistaken for clinical readiness; this project
  reports the leakage-aware number, shows the naive number only for comparison, and labels the demo as not for diagnosis.
- **Privacy.** The web demo runs entirely in the browser; uploaded images are not sent anywhere.
- **Attribution.** The dataset (CC BY 4.0) and its sources are credited in the [references](references.md).
""")

# ================================================================ limitations
w("limitations.md", f"""
# Limitations

{NAV}

These are the specific known weaknesses of this study and demo. Each one limits what the numbers can support.

## Study design
1. **One training seed.** Each configuration was trained once (seed 42). The bootstrap confidence intervals
   ({pc(F['accuracy_ci95'][0])}–{pc(F['accuracy_ci95'][1])} for accuracy) cover test-set sampling only, **not training variance**.
   The v1 → v2 change ({pc(m1['clean']['full_test']['accuracy'])} → {pc(F['accuracy'])}, with {CH['changed']} changed predictions)
   suggests that run-to-run variation is of the same order as the differences discussed.
2. **CPU training with a small budget.** 12 epochs on an 8-core CPU, batch size 32, no hyper-parameter search, no
   ensembling and no test-time augmentation. Higher accuracy may be possible; the study focuses on evaluation, not
   on the best score.
3. **No external or clinical validation.** All results come from one public Kaggle dataset. There is no test on another
   hospital, scanner or protocol, no prospective data and no comparison with radiologists.
4. **2-D slices, not patients.** The dataset has no patient identifiers, so a patient-level split is impossible. Different
   slices of the same patient that do not look alike can still be in both training and Testing; the duplicate audit only
   catches images that look nearly identical.

## Data
5. **Label noise.** {d['imgs_in_label_conflict_groups']} images sit in duplicate groups with conflicting labels, including pairs
   inside the official Testing folder (e.g. `Te-gl_68` labelled glioma and `Te-me_96` labelled meningioma are near-identical),
   so some test errors may be unavoidable.
6. **Near-duplicates remain possible.** pHash plus thumbnail correlation misses copies that were cropped, rotated or edited
   more heavily. A looser rule would link {S['extra_test_images_linked_to_train_val']} more Testing images
   ([sensitivity check](dedup_and_splits.md#sensitivity-check-a-looser-rule-no-retraining)); most `-aug-` files match no
   other image ([data card](data_card.md#known-issues)).
7. **Source confounding and class imbalance after deduplication.** No-tumour images come from a different collection,
   and only {n(d['subset_by_class']['train']['notumor'])} of them remain for training. The model over-predicts "no tumour"
   (precision {r(F['per_class']['notumor']['precision'])}).
8. **Coarse labels.** Four classes only; no grading, no other tumour types, no segmentation.

## Calibration and uncertainty
9. **The uncertain threshold does not transfer fully.** τ = {r(U['conf_threshold'], 2)} gives {pc(U['val_retained_accuracy'])}
   accuracy on confident validation predictions but {pc(F['accuracy_confident'])} on Testing. Calibration was fitted on
   in-distribution validation images and is not guaranteed on new data.

## Explainability
10. **Weak faithfulness.** On the deletion test no method beats a random map (Integrated Gradients only matches it); on
    insertion all methods do. The black-fill deletion test is itself hard to interpret for MRI.
11. **Failed sanity checks.** Integrated Gradients keeps a Spearman correlation of {r(sm[('integrated_gradients', 'conv_stem')], 2)}
    with its original map after all weights are randomised, and Grad-CAM++ stays at {r(sm[('gradcampp', 'classifier')], 2)} after
    the classifier is randomised.
12. **Small XAI sample and no ground truth.** {n_fa_img} images for faithfulness, {n_san_img} for the sanity check, and no tumour
    masks, so localisation accuracy was not measured. LIME was not run.

## Web demo
13. **Image recompression (fixed in v2.0.2).** Until v2.0.2 the preset images were resized and re-encoded, which changed the
    displayed result of `Te-gl_391`. Since v2.0.2 the presets are the original files, byte-for-byte, and an automated check
    confirms they reproduce the saved predictions within 0.01. Uploaded images are still decoded and resized by the browser's
    canvas, which can differ slightly from PIL; in our test of 12 uploads the largest probability difference to PyTorch was 0.0011.
14. **Precomputed maps for presets only.** Grad-CAM++, Integrated Gradients and KernelSHAP are shown only for the eight presets;
    uploads get Grad-CAM and occlusion.
15. **Any image gets an answer.** The model has no out-of-distribution detector; a non-MRI image still receives one of the four
    classes, possibly with high confidence.
""")

# ================================================================ future work
w("future_work.md", f"""
# Future work

{NAV}

Concrete next steps, roughly in order of how much they would change the conclusions. Each one addresses an item in
[limitations](limitations.md).

1. **Several seeds.** Train the leakage-aware and naive models with at least five seeds each and report the mean and spread
   next to the bootstrap intervals. This would show how much of the v1/v2 difference and of the naive gap is training variance.
2. **External test set.** Evaluate the unchanged model on an independent collection, for example 2-D slices extracted from
   BraTS (Menze et al., 2015) for gliomas, to measure performance outside this dataset.
3. **Patient-level data.** Repeat the study on data with patient identifiers (the original figshare release by Cheng et al.,
   2015, has patient IDs for its three tumour classes) and compare image-level with patient-level splits.
4. **Stronger duplicate detection.** Add learned image embeddings (e.g. self-supervised features) next to pHash, check the
   `-aug-` files against them, and publish a cleaned file list.
5. **Relabel the conflicting groups.** Have the {d['imgs_in_label_conflict_groups']} images in label-conflict groups reviewed
   and report results with and without them.
6. **Address source confounding.** Test whether the model can tell the source collections apart from background or intensity
   cues alone, and try intensity standardisation or skull stripping.
7. **Better uncertainty.** Compare temperature scaling with ensembles or conformal prediction sets, and add an
   out-of-distribution check so that non-MRI inputs are rejected.
8. **Localisation ground truth for XAI.** Use the tumour masks in the figshare release to measure how much of each heatmap
   falls on the tumour, extend the sanity check to more images, and run LIME alongside KernelSHAP with the same superpixels.
9. **Larger or more modern backbones with a GPU.** Compare with ConvNeXt or ViT models and a hyper-parameter search, keeping the
   same leakage-aware split so the comparison is fair.
10. **Demo.** Show precomputed maps for uploads where feasible (e.g. Grad-CAM++ in the browser) and add an input check that
    warns when an image does not look like a brain MRI slice.
""")

# ================================================================ FAQ
w("faq.md", f"""
# FAQ and troubleshooting

{NAV}

## Questions
**Why is the accuracy lower than many published results, which are often above 94%?**
Because near-copies of Testing images were removed from training. The identical model trained on the original folders
reaches {pc(FN['accuracy'])} here, and it is right on {LB['naive|meningioma|dup']['correct']}/{LB['naive|meningioma|dup']['n']}
Testing meningiomas with a strict near-duplicate in Training against {LB['clean|meningioma|dup']['correct']}/{LB['clean|meningioma|dup']['n']}
for the leakage-aware model. See [dedup and splits](dedup_and_splits.md).

**Why is v2 less accurate than v1, and why is it still preferred?**
v2 is preferred because its split is cleaner. The accuracy change mainly reflects training variance of a single seed; see
[v1 vs v2](changelog.md#why-v2-is-preferred).

**Can I use this for diagnosis?** No. It is a research and education project ([model card](model_card.md)).

**Are uploaded images sent to a server?** No. The demo runs entirely in your browser with ONNX Runtime Web.

**Why does the demo say "uncertain"?** The calibrated top probability is below τ = {r(U['conf_threshold'], 2)}. On Testing, such
predictions are right only {pc(F['accuracy_uncertain'])} of the time.

**Why is LIME described but not reported?** It was not run; KernelSHAP on superpixels is the perturbation method evaluated.
[Explainability](explainability.md#running-lime-yourself-not-part-of-the-reported-results) shows how to run it.

**Where does each number come from?** From `v2/results/`; the README, paper macros, demo statistics and this documentation are
generated from those files ([results catalogue](results_catalogue.md)).

**How do I run the v1 code?** `git checkout v1.0` and follow the README there. The `v1/` folder keeps v1's results, notebooks and
paper for comparison.

## Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: btmri` | Run scripts from `v2/`; for training use `cd v2/src && python -m btmri.train`. In a notebook, run the first set-up cell. |
| Dataset download fails | `kagglehub` needs internet access; the dataset is public, so no API key is needed. |
| Training is very slow | Expect about 2 minutes per epoch on 8 CPU cores. Limit threads with `OMP_NUM_THREADS`/`torch.set_num_threads` if other jobs share the machine, or use a GPU runtime in Colab. |
| `run_xai.py` is slow or the machine is overloaded | Pass a thread count, e.g. `python scripts/run_xai.py 4`. KernelSHAP is the slowest step. |
| Numbers differ slightly after retraining | Expected on a different machine: CPU kernels are not bit-for-bit deterministic across hardware ([model and training](model_and_training.md#determinism)). |
| Demo stays on "loading model" | Serve `docs/` over HTTP (`python3 -m http.server`), not `file://`; check that jsDelivr is reachable; the model is {r(onx['onnx_size_mb'], 0)} MB, so the first load takes a moment. |
| Occlusion takes long | It runs 169 forward passes in the browser (about 9–12 s in our tests); the progress counter updates while it runs. |
| Colab badge opens a notebook that cannot find files | The set-up cell must run first: it clones the repository and changes into `v2/`. |
| LaTeX build of the paper fails | Use a full TeX distribution or `tectonic` (`tectonic brain_tumor_mri_white_paper.tex` in `v2/paper/`). |
""")

print("documentation written:", sorted(p.name for p in OUT.glob("*.md")))

# ================================================================ top-level README
THESIS = [
    ("Quantum and hybrid quantum machine learning", [
        ("hybrid-vqc-wind-speed-forecasting", "hybrid variational quantum circuit (Conv1D-LSTM + VQC) for wind speed forecasting"),
        ("cbma-tn-vqc-wind-green-hydrogen", "reimplementation of CBMA-TN-VQC (CNN-BiLSTM-multi-head attention + tensor network + VQC) for wind forecasting and green hydrogen"),
        ("hcq-lstm-clqgwo-wind-forecasting", "reimplementation of a physics-informed hybrid classical-quantum LSTM tuned by a chaotic Lévy-flight quantum-inspired grey wolf optimiser")]),
    ("Metaheuristic hybrids", [
        ("geoai-hybrid-ga-pso-anfis-wind-forecasting", "GA-PSO-ANFIS-CNN-LSTM hybrid for probabilistic multi-site wind speed forecasting")]),
    ("Deep learning with attention and gating", [
        ("cnn-bilstm-attention-synthetic-scada", "CNN-BiLSTM-attention forecasting on synthetic SCADA data, with baselines"),
        ("cnn-bilstm-attention-real-scada", "CNN-BiLSTM-attention forecasting on wind-turbine SCADA data, with statistical tests"),
        ("cnn-bilstm-attention-benchmarking-indian-scada", "benchmark of CNN-BiLSTM-attention vs ANN, LSTM, SVM and ARIMA-ANN"),
        ("xai-shap-lime-cnn-bilstm-attention-wind", "CNN-BiLSTM-attention explained with SHAP, LIME and attention"),
        ("wind-forecasting-mathematical-formulations-validation", "hybrid deep-learning forecasting with Diebold-Mariano and Ljung-Box validation"),
        ("wuweidl-daoist-gated-wind-forecasting", "WuWeiDL: Daoist-inspired turbulence-gated attention"),
        ("rso-net-wind-grid-stabilization", "reimplementation of RSO-Net: TCN-attention probabilistic forecasting with human-approved grid proposals")]),
    ("IoT and multi-agent systems", [
        ("iot-n8n-wind-turbine-predictive-maintenance", "wind-turbine predictive maintenance with IoT data, a FastAPI service and an n8n workflow"),
        ("multi-agent-turbine-fleet-chief-of-staff", "multi-agent AI chief of staff for wind-turbine fleet operations")]),
    ("Statistical baselines", [
        ("arima-sarima-wind-speed-chennai", "ARIMA and SARIMA daily wind speed forecasting for Chennai")]),
]
GHU = "https://github.com/Rishabh-bgp/"
thesis_md = "\n".join(f"- **{g}**\n" + "\n".join(f"  - [{k}]({GHU}{k}): {v}" for k, v in items) for g, items in THESIS)
DOCS_TOC = [("overview.md", "Overview and quick start"), ("data_card.md", "Dataset and data card"),
            ("dedup_and_splits.md", "Duplicate audit and leakage-aware split"), ("model_and_training.md", "Model and training"),
            ("evaluation_and_calibration.md", "Evaluation and calibration"), ("explainability.md", "Explainability"),
            ("web_demo.md", "Web demo and ONNX export"), ("changelog.md", "v1 vs v2 and changelog"),
            ("../REPRODUCIBILITY.md", "Reproducibility guide"), ("repository_structure.md", "Repository structure"),
            ("scripts_reference.md", "Script and API reference"), ("results_catalogue.md", "Results catalogue"),
            ("model_card.md", "Model card"), ("limitations.md", "Limitations"), ("future_work.md", "Future work"),
            ("references.md", "References"), ("faq.md", "FAQ and troubleshooting")]
toc = "\n".join(f"{i}. [{t}]({'documentation/' + f if not f.startswith('../') else f[3:]})" for i, (f, t) in enumerate(DOCS_TOC, 1))
(REPO / "README.md").write_text(f"""<div align="center">

# Brain Tumour MRI Classification: Leakage-Aware, Calibrated and Explainable EfficientNet-B0

[![Live demo](https://img.shields.io/badge/live_demo-NEURO%2F%2FSCAN-2DD4BF?logo=githubpages&logoColor=black)]({PAGES})
[![White paper](https://img.shields.io/badge/white_paper-PDF-22D3EE)](v2/paper/brain_tumor_mri_white_paper.pdf)
[![Documentation](https://img.shields.io/badge/docs-documentation%2F-0F766E)](documentation/README.md)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue)

**[▶ Live demo: NEURO//SCAN]({PAGES})** · **[📄 White paper (PDF)](v2/paper/brain_tumor_mri_white_paper.pdf)** ·
**[📚 Documentation](documentation/README.md)**

</div>

{DISCLAIMER}

Four-way classification of brain MRI slices (glioma, meningioma, pituitary tumour, no tumour) on the Kaggle
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), with evaluation
hygiene first: a perceptual-hash audit of near-duplicates shared by the official Training and Testing folders, a
leakage-aware split, an EfficientNet-B0 fine-tuned on a CPU, temperature scaling with an "uncertain" flag, five
sanity-checked explanation methods, and an in-browser demo.

By **Rishabh Aryan**, M.Tech (Artificial Intelligence and Data Science), Department of Computer Science and Engineering,
Indian Institute of Information Technology Bhagalpur, Bihar, India.

<p align="center"><img src="docs/screenshot.png" alt="NEURO//SCAN web demo: MRI viewer with a Grad-CAM overlay, class probabilities and the Why this prediction panel" width="100%"></p>

## Versions
| Version | Folder | Status | Test accuracy (95% CI) | Macro-F1 |
|---|---|---|---|---|
| **v2** | [`v2/`](v2/README.md) | current; served by the [live demo]({PAGES}) | {ci(F)} | {r(F['macro_f1'])} |
| v1 | [`v1/`](v1/README.md) | frozen; code runs at tag [`v1.0`]({GH}/tree/v1.0) | {ci(m1['clean']['full_test'])} | {r(m1['clean']['full_test']['macro_f1'])} |

Both on the official Testing folder (n = {n(F['n'])}). v2 uses a stricter duplicate rule and removes
{n(d['train_imgs_removed_cross_split'])} instead of {n(d1['train_imgs_removed_cross_split'])} Training images; it is
preferred because its split is cleaner, not because of its score. The difference in accuracy mainly reflects training
variance of a single seed ([v1 vs v2](documentation/changelog.md)). An identical model trained on the original folders
scores {pc(FN['accuracy'])}; much of that gap appears to come from near-copies of Testing images in its training data.

## Notebooks (v2, Google Colab)
| # | Notebook | Colab |
|---|---|---|
""" + "\n".join(f"| {i} | [`{k}.ipynb`](v2/notebooks/{k}.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}{k}.ipynb) |" for i, k in enumerate(NB, 1)) + f"""

Quick start (local): `git clone {GH}.git && cd brain-tumor-mri-classification/v2 && pip install -r requirements.txt`,
then follow the [reproducibility guide](REPRODUCIBILITY.md).

## Documentation
The full documentation is in [`documentation/`](documentation/README.md):

{toc}

Project files: [CONTRIBUTING](CONTRIBUTING.md) · [Code of conduct](CODE_OF_CONDUCT.md) · [CITATION.cff](CITATION.cff) ·
[references.bib](references.bib) · [LICENSE](LICENSE)

## Repository layout
```text
v2/              current version: src/, scripts/, notebooks/, results/, models/, paper/, README, requirements
v1/              frozen first version (same layout; code runs at tag v1.0)
docs/            GitHub Pages web demo (serves the v2 model)
documentation/   long-form documentation
tests/           tests for the v2 code and the demo
```
Details: [repository structure](documentation/repository_structure.md).

## Limitations
One training seed and CPU training; no external or clinical validation; label noise and possible remaining near-duplicates
in the dataset; confidence intervals that exclude training variance; explanation methods that do not beat a random map on
the deletion test, with Integrated Gradients failing the randomisation sanity check. The preset-image recompression in the
demo was fixed in v2.0.2. Full list: [limitations](documentation/limitations.md).

## Future work
Several seeds, an external test set, patient-level data, stronger duplicate detection, relabelling of conflicting groups,
better uncertainty and out-of-distribution checks, and tumour masks to test explanation localisation:
[future work](documentation/future_work.md).

## References
The white paper's {len(BIB)} references with DOIs/URLs, plus dataset and software citations, are in
[references](documentation/references.md) and [`references.bib`](references.bib). To cite this project, use
[`CITATION.cff`](CITATION.cff).

## Related projects
**Machine learning and responsible AI**
- [machine-learning-part-1]({GHU}machine-learning-part-1): scikit-learn classification and regression notebooks
- [machine-learning-part-2]({GHU}machine-learning-part-2): computer vision with YOLO11 (detection, segmentation, pose, leaf disease)
- [machine-learning-part-3]({GHU}machine-learning-part-3): YOLO11 road-damage detection with the in-browser [ROAD//SCAN](https://rishabh-bgp.github.io/machine-learning-part-3/) demo
- [responsible-ai-uci-adult]({GHU}responsible-ai-uci-adult): responsible-AI audit on UCI Adult with the [FAIR//AUDIT](https://rishabh-bgp.github.io/responsible-ai-uci-adult/) dashboard
- [spam-scan]({GHU}spam-scan): explainable spam, phishing and scam detector ([live](https://spam-scan.onrender.com))
- [beed-ml-study]({GHU}beed-ml-study): leakage-aware seizure-type benchmark on the Bangalore EEG Epilepsy Dataset
- [pan-cancer-classification]({GHU}pan-cancer-classification): leakage, conformal reliability and data efficiency on TCGA Pan-Cancer RNA-Seq

**M.Tech in AI and Data Science, IIIT Bhagalpur**: thesis repositories for *An Investigation into Advanced Techniques for
Wind Speed Forecasting with Applications to Renewable Energy Grid Integration*

{thesis_md}

[All repositories](https://github.com/Rishabh-bgp?tab=repositories) · [Profile](https://github.com/Rishabh-bgp)

## License
[MIT](LICENSE) © 2026 Rishabh Aryan. Dataset © its Kaggle author and source collections (CC BY 4.0); EfficientNet-B0
weights from timm (Apache-2.0); ONNX Runtime Web by Microsoft (MIT); Captum (BSD-3-Clause).
""")
print("README.md written")
