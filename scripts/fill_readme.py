"""Fill README.md from scripts/README_template.md using results/metrics.json and the other saved result files."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT / "results/metrics.json").read_text())
d = json.loads((ROOT / "results/review/dedup_summary.json").read_text())
cal = m["calibration"]["clean"]
C, N = m["clean"], m.get("naive")
from decimal import ROUND_HALF_UP, Decimal  # noqa: E402
pc = lambda x: str(Decimal(repr(100 * float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)) + "%"  # noqa: E731
f = C["full_test"]
LAB = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumour", "pituitary": "Pituitary"}
rows = "\n".join(f"| {LAB[c]} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {f['roc_auc_per_class'][c]:.3f} |"
                 for c, v in f["per_class"].items())
rows += f"\n| **Macro** | | | **{f['macro_f1']:.3f}** | **{f['roc_auc_macro_ovr']:.3f}** |"
val = {
    "DUP_FILES": d["exact_duplicate_files"], "UNIQ_MD5": f"{d['n_unique_md5']:,}", "STRICT_PAIRS": f"{d['strict_pairs']:,}",
    "STRICT_CROSS": f"{d['strict_cross_split_pairs']:,}", "LINKED_PAIRS": f"{d['linked_pairs']:,}", "LINKED_CROSS": f"{d['linked_cross_split_pairs']:,}",
    "TEST_STRICT": d["test_imgs_with_train_dup_strict"], "TEST_STRICT_PCT": pc(d["test_imgs_with_train_dup_strict"] / 1600),
    "TEST_LINKED": d["test_imgs_with_train_dup_linked"], "TEST_LINKED_PCT": pc(d["test_imgs_with_train_dup_linked"] / 1600),
    "SD_NO": d["test_strict_dup_by_class"]["notumor"], "SD_ME": d["test_strict_dup_by_class"]["meningioma"],
    "SD_GL": d["test_strict_dup_by_class"]["glioma"], "SD_PI": d["test_strict_dup_by_class"]["pituitary"],
    "CONFLICT": d["imgs_in_label_conflict_groups"], "REMOVED": f"{d['train_imgs_removed_cross_split']:,}",
    "EXTRA_PAIRS": f"{d.get('extra_pairs_v2_rule', 0):,}", "EXTRA_CROSS": f"{d.get('extra_cross_split_pairs_v2_rule', 0):,}",
    "REMOVED_V1": "1,198",
    "V1V2_TABLE": (ROOT / "results/v1_vs_v2.md").read_text().strip(),
    "REC_ME": pc(f["per_class"]["meningioma"]["recall"]),
    "SENS_TEST": str(m["sensitivity_looser_rule"]["extra_test_images_linked_to_train_val"]),
    "SENS_ACC": pc(m["sensitivity_looser_rule"]["clean_acc_excluding_extra"]),
    "SENS_N": f"{m['sensitivity_looser_rule']['n_excluding_extra']:,}",
    "EXCL_NO": d["subset_by_class"]["excluded"]["notumor"], "N_TRAIN": f"{d['subset_counts']['train']:,}", "N_VAL": d["subset_counts"]["val"],
    "ACC": pc(f["accuracy"]), "ACC_CI": f"{pc(f['accuracy_ci95'][0])}–{pc(f['accuracy_ci95'][1])}", "F1": f"{f['macro_f1']:.3f}",
    "AUC": f"{f['roc_auc_macro_ovr']:.3f}", "ECE_RAW": f"{f['ece_raw']:.3f}", "ECE_CAL": f"{f['ece_calibrated']:.3f}", "TEMP": f"{cal['temperature']:.2f}",
    "N_DEDUP": f"{C['dedup_test']['n']:,}", "ACC_DEDUP": pc(C["dedup_test"]["accuracy"]), "F1_DEDUP": f"{C['dedup_test']['macro_f1']:.3f}",
    "PERCLASS_ROWS": rows, "GLI_NO": f["confusion_matrix"][0][2], "MEN_NO": f["confusion_matrix"][1][2],
    "PREC_NO": pc(f["per_class"]["notumor"]["precision"]), "REC_NO": pc(f["per_class"]["notumor"]["recall"]), "VAL_ACC": pc(cal["val_accuracy"]),
    "N_DUP": C["test_with_strict_train_dup"]["n"], "N_NODUP": C["test_without_linked_train_dup"]["n"],
    "ACC_C_DUP": pc(C["test_with_strict_train_dup"]["accuracy"]), "ACC_C_NODUP": pc(C["test_without_linked_train_dup"]["accuracy"]),
    "THR": f"{cal['uncertainty']['conf_threshold']:.2f}", "VAL_COV": pc(cal["uncertainty"]["val_coverage"]),
    "UNC_RATE": pc(f["uncertain_rate"]), "ACC_CONF": pc(f["accuracy_confident"]), "ACC_UNC": pc(f["accuracy_uncertain"]),
}
if N:
    lb = m["leakage_by_class"]
    md, mc = lb["naive|meningioma|dup"], lb["clean|meningioma|dup"]
    mnd, mcnd = lb["naive|meningioma|nodup"], lb["clean|meningioma|nodup"]
    val.update({"ACC_N": pc(N["full_test"]["accuracy"]), "ACC_N_DUP": pc(N["test_with_strict_train_dup"]["accuracy"]),
                "ACC_N_NODUP": pc(N["test_without_linked_train_dup"]["accuracy"]),
                "NAIVE_TEXT": (f"An identical model trained on the original Training folder (random image-level validation split) scores "
                               f"**{pc(N['full_test']['accuracy'])}** on the same Testing folder, {abs(N['full_test']['accuracy'] - f['accuracy']) * 100:.1f} points higher. "
                               f"Part of the gap may come from more training data (4,760 vs {d['subset_counts']['train']:,} training images; 1,190 vs {d['subset_by_class']['train']['notumor']} no-tumour), "
                               f"but the per-class split shows memorisation: on the {md['n']} Testing meningiomas that have a strict near-duplicate in Training "
                               f"the naive model is correct on **{md['correct']}/{md['n']}**, the leakage-aware model on {mc['correct']}/{mc['n']}, while on the "
                               f"{mnd['n']} meningiomas without one the two are close ({mnd['correct']} vs {mcnd['correct']} correct). "
                               f"{md['correct'] - mc['correct']} of the {round(abs(N['full_test']['accuracy'] - f['accuracy']) * 1600)}-image gap comes from these {md['n']} meningiomas, "
                               f"which suggests that much of the gap reflects memorisation of near-copies; part of it also reflects the leakage-aware model losing "
                               f"all training examples of that sub-population, so its leak-free accuracy is unknown.")})
xai = ROOT / "results/xai/faithfulness_per_image.csv"
if xai.exists():
    fa = pd.read_csv(xai); san = pd.read_csv(ROOT / "results/xai/sanity_cascading_randomisation.csv")
    L = {"gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++", "integrated_gradients": "Integrated Gradients", "occlusion": "Occlusion",
         "kernel_shap": "KernelSHAP (SLIC)", "random": "Random baseline"}
    t = ["| Method | Deletion AUC ↓ | Insertion AUC ↑ | Mass on head region | ρ after classifier randomised | ρ after full randomisation |", "|---|---|---|---|---|---|"]
    for k in L:
        r = fa[fa.method == k]
        s1 = san[(san.method == k) & (san.stage == "classifier")].spearman.mean() if k != "random" else None
        s2 = san[(san.method == k) & (san.stage == "conv_stem")].spearman.mean() if k != "random" else None
        t.append(f"| {L[k]} | {r.deletion_auc.mean():.3f} ± {r.deletion_auc.std():.3f} | {r.insertion_auc.mean():.3f} ± {r.insertion_auc.std():.3f} | "
                 f"{100 * r.inside_foreground.mean():.1f}% | {'—' if s1 is None else f'{s1:.2f}'} | {'—' if s2 is None else f'{s2:.2f}'} |")
    val["XAI_TABLE"] = "\n".join(t)
    val["XAI_TEXT"] = (ROOT / "results/xai/summary.md").read_text().strip() if (ROOT / "results/xai/summary.md").exists() else ""
else:
    val["XAI_TABLE"], val["XAI_TEXT"] = "_XAI study in progress; results will appear here._", ""
s = (ROOT / "scripts/README_template.md").read_text()
for k, v in val.items():
    s = s.replace("{" + k + "}", str(v))
(ROOT / "README.md").write_text(s)
import re
left = re.findall(r"\{[A-Z_]+\}", s)
print("unfilled:", left)
