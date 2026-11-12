"""Derive every reported number from results/predictions/*_test_predictions.csv (single source of truth).

Writes results/metrics.json, paper/tables/*.tex, docs/stats.json, results/review/test_predictions.csv and the
evaluation figures in results/figures/."""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support,  # noqa: E402
                             roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btmri import CLASS_LABELS, CLASSES  # noqa: E402
from btmri.calibrate import ece  # noqa: E402

PRED = ROOT / "results/predictions"
FIG = ROOT / "results/figures"
TAB = ROOT / "paper/tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight"})
NAMES = [CLASS_LABELS[c] for c in CLASSES]


def arrays(df, raw=False):
    pre = "raw_p_" if raw else "p_"
    y = np.array([CLASSES.index(c) for c in df.true])
    p = df[[pre + c for c in CLASSES]].values
    return y, p


def metrics(df, boot=1000, seed=42):
    y, p = arrays(df)
    yhat = p.argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(y, yhat, labels=range(4), zero_division=0)
    m = {"n": int(len(df)), "accuracy": float((yhat == y).mean()), "macro_f1": float(f1_score(y, yhat, average="macro")),
         "per_class": {c: {"precision": float(pr[k]), "recall": float(rc[k]), "f1": float(f1[k]), "support": int(sup[k])}
                       for k, c in enumerate(CLASSES)},
         "confusion_matrix": confusion_matrix(y, yhat, labels=range(4)).tolist(),
         "ece_calibrated": ece(p, y), "ece_raw": ece(arrays(df, raw=True)[1], y)}
    if len(np.unique(y)) == 4:
        m["roc_auc_macro_ovr"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
        m["roc_auc_per_class"] = {c: float(roc_auc_score(y == k, p[:, k])) for k, c in enumerate(CLASSES)}
    rng = np.random.default_rng(seed)
    acc, mf1 = [], []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y))
        acc.append((yhat[i] == y[i]).mean())
        mf1.append(f1_score(y[i], yhat[i], average="macro"))
    m["accuracy_ci95"] = [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]
    m["macro_f1_ci95"] = [float(np.percentile(mf1, 2.5)), float(np.percentile(mf1, 97.5))]
    m["uncertain_rate"] = float(df.uncertain.mean())
    conf = ~df.uncertain.values
    m["accuracy_confident"] = float((yhat[conf] == y[conf]).mean()) if conf.any() else None
    m["accuracy_uncertain"] = float((yhat[~conf] == y[~conf]).mean()) if (~conf).any() else None
    return m


calib = json.loads((PRED / "calibration.json").read_text())
runs = {r: pd.read_csv(PRED / f"{r}_test_predictions.csv") for r in ("clean", "naive") if (PRED / f"{r}_test_predictions.csv").exists()}
res = {"calibration": calib}
for r, df in runs.items():
    res[r] = {"full_test": metrics(df),
              "dedup_test": metrics(df[df.test_dedup_rep]),
              "test_with_strict_train_dup": metrics(df[df.test_has_train_dup_strict]),
              "test_without_linked_train_dup": metrics(df[~df.test_has_train_dup])}
    sd = df[df.test_has_train_dup_strict]
    present = sorted(set(sd.true))
    res[r]["test_with_strict_train_dup"]["macro_f1_note"] = (
        "macro_f1 averages over all four classes; pituitary has no images in this subset (support 0) but receives "
        "predictions, so its F1 is counted as 0. macro_f1_present_classes averages only the classes present.")
    res[r]["test_with_strict_train_dup"]["present_classes"] = present
    res[r]["test_with_strict_train_dup"]["macro_f1_present_classes"] = float(
        f1_score(sd.true, sd.pred, labels=present, average="macro", zero_division=0))
(ROOT / "results/metrics.json").write_text(json.dumps(res, indent=2))

clean = runs["clean"]
shutil.copy(PRED / "clean_test_predictions.csv", ROOT / "results/review/test_predictions.csv")
M = res["clean"]["full_test"]

# ---------------- demo stats (same numbers as the paper) ----------------
cm = np.array(M["confusion_matrix"])
stats = {"temperature": calib["clean"]["temperature"], "conf_threshold": calib["clean"]["uncertainty"]["conf_threshold"],
         "threshold_rule": calib["clean"]["uncertainty"]["rule"], "test_n": M["n"], "test_accuracy": M["accuracy"],
         "test_macro_f1": M["macro_f1"], "classes": CLASSES, "labels": NAMES, "per_class": {}}
for k, c in enumerate(CLASSES):
    row = cm[k].copy(); row[k] = -1
    col = cm[:, k].copy(); col[k] = -1
    stats["per_class"][c] = {**{x: M["per_class"][c][x] for x in ("precision", "recall", "f1", "support")},
                             "auc": M["roc_auc_per_class"][c],
                             "missed_as": CLASSES[int(row.argmax())], "missed_as_n": int(row.max()),
                             "false_alarm_from": CLASSES[int(col.argmax())], "false_alarm_from_n": int(col.max())}
(ROOT / "docs/stats.json").write_text(json.dumps(stats, indent=2))


# ---------------- LaTeX tables ----------------
from decimal import ROUND_HALF_UP, Decimal  # noqa: E402


def f(x):
    """Round half up to 3 decimals (e.g. 0.8975 -> 0.898), consistent with the percentages in the text."""
    return str(Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


lines = [r"\begin{tabular}{lcccc}", r"\toprule", r"Class & Precision & Recall & F1 & ROC-AUC \\", r"\midrule"]
for c in CLASSES:
    pc = M["per_class"][c]
    lines.append(f"{CLASS_LABELS[c]} & {f(pc['precision'])} & {f(pc['recall'])} & {f(pc['f1'])} & {f(M['roc_auc_per_class'][c])} \\\\")
lines += [r"\midrule", f"Macro average & -- & -- & {f(M['macro_f1'])} & {f(M['roc_auc_macro_ovr'])} \\\\", r"\bottomrule", r"\end{tabular}"]
(TAB / "per_class.tex").write_text("\n".join(lines))

lines = [r"\begin{tabular}{llcccc}", r"\toprule", r"Model & Test subset & $n$ & Accuracy [95\% CI] & Macro-F1 [95\% CI] & ECE \\", r"\midrule"]
for r in runs:
    for key, lab in (("full_test", "Full Testing"), ("dedup_test", "Testing, one per group"),
                     ("test_with_strict_train_dup", "Has strict dup.\\ in Training"),
                     ("test_without_linked_train_dup", "No linked dup.\\ in Training")):
        m = res[r][key]
        f1s = (f"{f(m['macro_f1'])} [{f(m['macro_f1_ci95'][0])}, {f(m['macro_f1_ci95'][1])}]"
               if key in ("full_test", "dedup_test") else "-- (class-skewed subset)")
        lines.append(f"{'Leakage-aware' if r == 'clean' else 'Naive'} & {lab} & {m['n']} & {f(m['accuracy'])} [{f(m['accuracy_ci95'][0])}, {f(m['accuracy_ci95'][1])}] & "
                     f"{f1s} & {f(m['ece_calibrated'])} \\\\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines.append(r"\end{tabular}")
(TAB / "subsets.tex").write_text("\n".join(lines))

lines = [r"\begin{tabular}{l" + "c" * 4 + "}", r"\toprule", "True $\\downarrow$ / Pred. $\\rightarrow$ & " + " & ".join(NAMES) + r" \\", r"\midrule"]
for k in range(4):
    lines.append(NAMES[k] + " & " + " & ".join(str(v) for v in cm[k]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "confusion.tex").write_text("\n".join(lines))

# ---------------- figures ----------------
fig, ax = plt.subplots(figsize=(4.6, 4))
cmn = cm / cm.sum(1, keepdims=True)
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.1%}", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
ax.set_xticks(range(4), NAMES, rotation=30)
ax.set_yticks(range(4), NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix (Testing, n={M['n']})")
fig.savefig(FIG / "confusion_matrix.png"); plt.close(fig)

y, p = arrays(clean)
fig, ax = plt.subplots(figsize=(4.6, 4))
for k, c in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(y == k, p[:, k])
    ax.plot(fpr, tpr, label=f"{CLASS_LABELS[c]} (AUC {M['roc_auc_per_class'][c]:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("One-vs-rest ROC (Testing)")
ax.legend(fontsize=8); fig.savefig(FIG / "roc_curves.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 4))
for raw, lab in ((True, "uncalibrated"), (False, f"temperature T={calib['clean']['temperature']:.2f}")):
    yy, pp = arrays(clean, raw)
    conf, pred = pp.max(1), pp.argmax(1)
    edges = np.linspace(0, 1, 11)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5:
            xs.append(conf[m].mean()); ys.append((pred[m] == yy[m]).mean())
    ax.plot(xs, ys, "o-", label=f"{lab} (ECE {ece(pp, yy):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("Mean confidence"); ax.set_ylabel("Accuracy"); ax.set_title("Reliability (Testing, bins with n>=5)")
ax.legend(fontsize=8); fig.savefig(FIG / "reliability.png"); plt.close(fig)

for r in runs:
    h = pd.read_csv(ROOT / f"results/runs/{r}/history.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="validation")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Cross-entropy"); ax[0].legend()
    ax[1].plot(h.epoch, h.val_acc, label="val accuracy"); ax[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1")
    ax[1].set_xlabel("Epoch"); ax[1].legend()
    fig.suptitle(f"Training curves ({'leakage-aware' if r == 'clean' else 'naive'} split)")
    fig.savefig(FIG / f"training_curves_{r}.png"); plt.close(fig)
print(json.dumps({r: {k: (v["accuracy"], v["macro_f1"], v["n"]) for k, v in res[r].items()} for r in runs}, indent=1))

# ---------------- leakage by class: accuracy on Testing images with / without a strict duplicate in Training ----------------
rows, lb = [], {}
for c in CLASSES:
    cells = []
    for dup in (True, False):
        for r in runs:
            d = runs[r][(runs[r].true == c) & (runs[r].test_has_train_dup_strict == dup)]
            acc = float((d.true == d.pred).mean()) if len(d) else None
            lb[f"{r}|{c}|{'dup' if dup else 'nodup'}"] = {"n": int(len(d)), "correct": int((d.true == d.pred).sum()), "accuracy": acc}
            cells.append("--" if acc is None else f"{acc:.3f} ({int((d.true == d.pred).sum())}/{len(d)})")
    rows.append(f"{CLASS_LABELS[c]} & " + " & ".join(cells) + r" \\")
res["leakage_by_class"] = lb
(ROOT / "results/metrics.json").write_text(json.dumps(res, indent=2))
hdr = " & ".join(("Leakage-aware" if r == "clean" else "Naive") for r in runs)
lines = [r"\begin{tabular}{l" + "c" * (2 * len(runs)) + "}", r"\toprule",
         r" & \multicolumn{" + str(len(runs)) + r"}{c}{Strict dup.\ in Training} & \multicolumn{" + str(len(runs)) + r"}{c}{No strict dup.} \\",
         r"Class & " + hdr + " & " + hdr + r" \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
(TAB / "leakage_by_class.tex").write_text("\n".join(lines))
print(json.dumps(lb, indent=1))


# ---------------- sensitivity: a looser link rule (Hamming 7-10 with correlation >= 0.95) ----------------
cand = pd.read_csv(ROOT / "results/review/candidate_pairs_hamming_le10.csv.gz")
grp = pd.read_csv(ROOT / "results/review/hashes_and_groups.csv").set_index("relpath")
loose = cand[(cand.hamming >= 7) & (cand.hamming <= 10) & (cand["corr"] >= 0.95) & ~cand.linked]
pool = set(grp.index[grp.subset.isin(["train", "val"])])
extra = set()
for a, b in zip(loose.path_i, loose.path_j):
    if a.startswith("Testing") and b in pool:
        extra.add(a)
    if b.startswith("Testing") and a in pool:
        extra.add(b)
c = runs["clean"]
keep = c[~c.relpath.isin(extra)]
sens = {"rule": "extra links: Hamming 7-10 and thumbnail correlation >= 0.95 (not already linked)",
        "extra_pairs": int(len(loose)), "extra_test_images_linked_to_train_val": len(extra),
        "extra_by_class": {k: int(v) for k, v in pd.Series([grp.loc[e, "cls"] for e in extra]).value_counts().items()},
        "clean_acc_full": float((c.true == c.pred).mean()), "clean_acc_excluding_extra": float((keep.true == keep.pred).mean()),
        "n_excluding_extra": int(len(keep)), "clean_acc_on_extra": float((c[c.relpath.isin(extra)].true == c[c.relpath.isin(extra)].pred).mean()) if extra else None}
res["sensitivity_looser_rule"] = sens
(ROOT / "results/metrics.json").write_text(json.dumps(res, indent=2))
print(json.dumps(sens, indent=1))
