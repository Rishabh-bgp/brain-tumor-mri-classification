"""Side-by-side v1 (commit a591cfa, tag v1.0) vs v2 table, built only from saved files:
results/v1/*.json (copied unchanged from a591cfa) and the current results/metrics.json, calibration.json, dedup_summary.json.
Writes paper/tables/v1v2.tex, results/v1_vs_v2.md (README) and results/v1_vs_v2.json."""
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ld = lambda p: json.loads((ROOT / p).read_text())  # noqa: E731
m1, m2 = ld("results/v1/metrics.json"), ld("results/metrics.json")
c1, c2 = ld("results/v1/calibration.json"), ld("results/predictions/calibration.json")
d1, d2 = ld("results/v1/dedup_summary.json"), ld("results/review/dedup_summary.json")


def r(x, nd):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def pc(x, nd=1):
    return r(100 * x, nd) + "%"


def ci(v):
    return f"{pc(v[0])}–{pc(v[1])}"


def side(m, c, d):
    f, n = m["clean"]["full_test"], m["naive"]["full_test"]
    lb = m["leakage_by_class"]
    return {
        "Duplicate link rule": None,
        "Linked pairs": f"{d['linked_pairs']:,}",
        "Training-folder images removed": f"{d['train_imgs_removed_cross_split']:,}",
        "Train / validation images": f"{d['subset_counts']['train']:,} / {d['subset_counts']['val']:,}",
        "Test accuracy (95% CI)": f"{pc(f['accuracy'])} ({ci(f['accuracy_ci95'])})",
        "Macro-F1": r(f["macro_f1"], 3),
        "Macro ROC-AUC": r(f["roc_auc_macro_ovr"], 3),
        "ECE raw → calibrated": f"{r(f['ece_raw'], 3)} → {r(f['ece_calibrated'], 3)}",
        "Temperature T / threshold τ": f"{r(c['clean']['temperature'], 2)} / {r(c['clean']['uncertainty']['conf_threshold'], 2)}",
        "Uncertain rate": pc(f["uncertain_rate"]),
        "Accuracy confident / uncertain": f"{pc(f['accuracy_confident'])} / {pc(f['accuracy_uncertain'])}",
        **{f"Recall {k}": pc(f["per_class"][k]["recall"]) for k in ("glioma", "meningioma", "notumor", "pituitary")},
        "One-per-group test accuracy": pc(m["clean"]["dedup_test"]["accuracy"]),
        "Naive model accuracy (same test)": pc(n["accuracy"]),
        "Gap naive − leakage-aware": f"{r(100 * (n['accuracy'] - f['accuracy']), 1)} pts",
        "Meningiomas with strict dup.: naive / leakage-aware": f"{lb['naive|meningioma|dup']['correct']}/{lb['naive|meningioma|dup']['n']} / {lb['clean|meningioma|dup']['correct']}/{lb['clean|meningioma|dup']['n']}",
        "Bootstrap resamples": None,
    }


v1, v2 = side(m1, c1, d1), side(m2, c2, d2)
v1["Duplicate link rule"] = "MD5, or Ham ≤ 6 & r ≥ 0.95"
v2["Duplicate link rule"] = "v1, or Ham ≤ 10 & r ≥ 0.98"
v1["Bootstrap resamples"] = "1,000 (200 for two subset rows)"
v2["Bootstrap resamples"] = "1,000 for every row"
rows = [(k, v1[k], v2[k]) for k in v1]
(ROOT / "results/v1_vs_v2.json").write_text(json.dumps([{"metric": k, "v1": a, "v2": b} for k, a, b in rows], indent=1, ensure_ascii=False))
md = ["| Metric (leakage-aware model unless stated) | v1 (`a591cfa`, tag `v1.0`) | v2 (this version) |", "|---|---|---|"]
md += [f"| {k} | {a} | {b} |" for k, a, b in rows]
(ROOT / "results/v1_vs_v2.md").write_text("\n".join(md) + "\n")


def tex(s):
    return (s.replace("%", "\\%").replace("≤", "$\\le$").replace("≥", "$\\ge$").replace("→", "$\\to$").replace("τ", "$\\tau$")
            .replace("−", "$-$").replace("–", "--").replace("&", "\\&").replace("Ham", "Hamming").replace(" r ", " $r$ ")
            .replace("notumor", "no tumour"))


t = ["\\begin{tabular}{lll}", "\\toprule", "Metric & v1 (a591cfa) & v2 \\\\", "\\midrule"]
t += [f"{tex(k)} & {tex(a)} & {tex(b)} \\\\" for k, a, b in rows]
t += ["\\bottomrule", "\\end{tabular}"]
(ROOT / "paper/tables/v1v2.tex").write_text("\n".join(t) + "\n")
print("\n".join(md))

# demo About/footer note (between the <!--V1V2--> markers in docs/index.html)
import re  # noqa: E402
f1, f2 = m1["clean"]["full_test"], m2["clean"]["full_test"]
note = (f"<b>About this version (v2).</b> After a team review, v2 uses a stricter duplicate rule (+{d2['extra_pairs_v2_rule']} linked pairs; "
        f"{d2['train_imgs_removed_cross_split']:,} instead of {d1['train_imgs_removed_cross_split']:,} Training images removed) and a retrained model. "
        f"Test accuracy is {pc(f2['accuracy'])} (macro-F1 {r(f2['macro_f1'], 3)}, ROC-AUC {r(f2['roc_auc_macro_ovr'], 3)}) versus "
        f"{pc(f1['accuracy'])} (macro-F1 {r(f1['macro_f1'], 3)}, ROC-AUC {r(f1['roc_auc_macro_ovr'], 3)}) in v1. The v2 score is lower because it is the more honest estimate "
        f"(part of the difference is single-seed training variance); see the <a href=\"https://github.com/Rishabh-bgp/brain-tumor-mri-classification#-v2-vs-v1-results-side-by-side\">v1 vs v2 table</a>.")
ip = ROOT / "docs/index.html"
html = ip.read_text()
html = re.sub(r"<!--V1V2-->.*?<!--/V1V2-->", lambda _: f"<!--V1V2-->{note}<!--/V1V2-->", html, flags=re.S)
ip.write_text(html)
