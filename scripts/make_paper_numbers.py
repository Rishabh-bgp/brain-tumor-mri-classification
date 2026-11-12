"""Write paper/numbers.tex (LaTeX macros) and paper/tables/xai.tex from the saved result files only."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT / "results/metrics.json").read_text())
d = json.loads((ROOT / "results/review/dedup_summary.json").read_text())
cal = m["calibration"]
onx = json.loads((ROOT / "results/onnx_verification.json").read_text())
out = {}


def pc(x, nd=1):
    """Percent, rounded half up (consistent with the tables)."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(repr(100 * float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)) + "\\%"


for run, tag in (("clean", "C"), ("naive", "N")):
    if run not in m:
        continue
    for sub, st in (("full_test", "Full"), ("dedup_test", "Dedup"), ("test_with_strict_train_dup", "Dup"), ("test_without_linked_train_dup", "NoDup")):
        r = m[run][sub]
        out[f"Acc{tag}{st}"] = pc(r["accuracy"])
        out[f"AccLo{tag}{st}"] = pc(r["accuracy_ci95"][0])
        out[f"AccHi{tag}{st}"] = pc(r["accuracy_ci95"][1])
        out[f"FOne{tag}{st}"] = f"{r['macro_f1']:.3f}"
        out[f"N{tag}{st}"] = f"{r['n']:,}"
    f = m[run]["full_test"]
    out[f"AUC{tag}"] = f"{f['roc_auc_macro_ovr']:.3f}"
    out[f"ECERaw{tag}"] = f"{f['ece_raw']:.3f}"
    out[f"ECECal{tag}"] = f"{f['ece_calibrated']:.3f}"
    out[f"Temp{tag}"] = f"{cal[run]['temperature']:.2f}"
    out[f"Thr{tag}"] = f"{cal[run]['uncertainty']['conf_threshold']:.2f}"
    out[f"ValAcc{tag}"] = pc(cal[run]["val_accuracy"])
    out[f"ValCov{tag}"] = pc(cal[run]["uncertainty"]["val_coverage"])
    out[f"ValRet{tag}"] = pc(cal[run]["uncertainty"]["val_retained_accuracy"])
    out[f"UncRate{tag}"] = pc(f["uncertain_rate"])
    out[f"AccConf{tag}"] = pc(f["accuracy_confident"])
    out[f"AccUnc{tag}"] = pc(f["accuracy_uncertain"]) if f["accuracy_uncertain"] is not None else "--"
    for c in ("glioma", "meningioma", "notumor", "pituitary"):
        k = c.capitalize()
        out[f"Rec{tag}{k}"] = pc(f["per_class"][c]["recall"])
        out[f"Prec{tag}{k}"] = pc(f["per_class"][c]["precision"])
    cm = f["confusion_matrix"]
    out[f"GliToNo{tag}"] = str(cm[0][2]); out[f"GliToMen{tag}"] = str(cm[0][1]); out[f"MenToNo{tag}"] = str(cm[1][2])
    out[f"MenToPit{tag}"] = str(cm[1][3])
for k in ("n_images", "n_unique_md5", "exact_duplicate_files", "strict_pairs", "strict_cross_split_pairs", "linked_pairs",
          "linked_cross_split_pairs", "groups", "test_imgs_with_train_dup_strict", "test_imgs_with_train_dup_linked",
          "train_imgs_removed_cross_split", "imgs_in_label_conflict_groups", "test_dedup_reps"):
    out["D" + "".join(w.capitalize() for w in k.split("_")).replace("Imgs", "Img").replace("1", "One")] = f"{d[k]:,}"
for s in ("train", "val", "test", "excluded"):
    out["Sub" + s.capitalize()] = f"{d['subset_counts'][s]:,}"
for c in ("glioma", "meningioma", "notumor", "pituitary"):
    out["StrictDup" + c.capitalize()] = str(d["test_strict_dup_by_class"][c])
    out["Excl" + c.capitalize()] = str(d["subset_by_class"]["excluded"][c])
    out["Train" + c.capitalize()] = str(d["subset_by_class"]["train"][c])
out["StrictDupPct"] = pc(d["test_imgs_with_train_dup_strict"] / 1600)
out["LinkedDupPct"] = pc(d["test_imgs_with_train_dup_linked"] / 1600)
out["OnnxMB"] = f"{onx['onnx_size_mb']:.1f}"
out["OnnxLogitDiff"] = f"{onx['max_abs_logit_diff_torch_vs_onnx']:.1e}"
out["CamDiff"] = f"{onx['max_abs_diff_onnx_cam_vs_autograd_gradcam_normalised']:.1e}"
h = pd.read_csv(ROOT / "results/runs/clean/history.csv")
out["BestEpochC"] = str(int(h.loc[h.val_macro_f1.idxmax(), "epoch"]))
out["EpochSecC"] = f"{h.seconds.median():.0f}"
if (ROOT / "results/runs/naive/history.csv").exists():
    hn = pd.read_csv(ROOT / "results/runs/naive/history.csv")
    out["BestEpochN"] = str(int(hn.loc[hn.val_macro_f1.idxmax(), "epoch"]))
    out["ValAccHistN"] = pc(hn.val_acc.max())

lbc = m.get("leakage_by_class", {})
for key, v in lbc.items():
    r, c, dup = key.split("|")
    tag = ("C" if r == "clean" else "N") + c.capitalize() + ("Dup" if dup == "dup" else "NoDup")
    out["LB" + tag] = f"{v['correct']}/{v['n']}" if v["n"] else "--"
    out["LBAcc" + tag] = pc(v["accuracy"]) if v["accuracy"] is not None else "--"
sv = m.get("sensitivity_looser_rule")
if sv:
    out["SensExtraPairs"] = f"{sv['extra_pairs']:,}"
    out["SensExtraTest"] = str(sv["extra_test_images_linked_to_train_val"])
    out["SensExtraNo"] = str(sv["extra_by_class"].get("notumor", 0))
    out["SensAccExcl"] = pc(sv["clean_acc_excluding_extra"])
    out["SensNExcl"] = f"{sv['n_excluding_extra']:,}"
    out["SensAccOnExtra"] = pc(sv["clean_acc_on_extra"]) if sv["clean_acc_on_extra"] is not None else "--"
for k in ("linked_pairs_v1_rule", "extra_pairs_v2_rule", "extra_cross_split_pairs_v2_rule"):
    if k in d:
        out["D" + "".join(w.capitalize() for w in k.split("_")).replace("1", "One").replace("2", "Two")] = f"{d[k]:,}"
sdm = m["clean"]["test_with_strict_train_dup"]
if "macro_f1_present_classes" in sdm:
    out["DupFOnePresentC"] = f"{sdm['macro_f1_present_classes']:.3f}"
    out["DupFOneAllC"] = f"{sdm['macro_f1']:.3f}"
if "naive" in m:
    lbc_ = m["leakage_by_class"]
    out["MenGapImgs"] = str(lbc_["naive|meningioma|dup"]["correct"] - lbc_["clean|meningioma|dup"]["correct"])
    out["MenDupN"] = str(lbc_["naive|meningioma|dup"]["n"])
    out["DiffFull"] = f"{abs(m['naive']['full_test']['accuracy'] - m['clean']['full_test']['accuracy']) * 100:.1f}"
    out["DiffImgs"] = str(round(abs(m['naive']['full_test']['accuracy'] - m['clean']['full_test']['accuracy']) * 1600))
au = ROOT / "results/review/augmented_summary.json"
if au.exists():
    a = json.loads(au.read_text())
    out["AugTr"], out["AugTe"], out["AugLinked"], out["AugLinkedTf"] = (str(a["n_aug_training"]), str(a["n_aug_testing"]),
                                                                       str(a["linked_any"]), str(a["linked_non_identity"]))
    out["AugAccC"], out["AugOtherAccC"] = pc(a["clean_acc_test_aug_meningioma"]), pc(a["clean_acc_test_other_meningioma"])
    out["AugAccN"], out["AugOtherAccN"] = pc(a["naive_acc_test_aug_meningioma"]), pc(a["naive_acc_test_other_meningioma"])
    for run, t in (("clean", "C"), ("naive", "N")):
        out[f"AugFair{t}"] = f"{a[run + '_nodup_aug_correct']}/{a[run + '_nodup_aug_n']}"
        out[f"AugFairOther{t}"] = f"{a[run + '_nodup_other_correct']}/{a[run + '_nodup_other_n']}"
        out[f"AugFairAcc{t}"] = pc(a[run + "_nodup_aug_correct"] / a[run + "_nodup_aug_n"])
        out[f"AugFairOtherAcc{t}"] = pc(a[run + "_nodup_other_correct"] / a[run + "_nodup_other_n"])
xai = ROOT / "results/xai"
LAB = {"gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++", "integrated_gradients": "Integrated Gradients",
       "occlusion": "Occlusion", "kernel_shap": "KernelSHAP (SLIC)", "random": "Random baseline"}
if (xai / "faithfulness_per_image.csv").exists():
    fa = pd.read_csv(xai / "faithfulness_per_image.csv")
    san = pd.read_csv(xai / "sanity_cascading_randomisation.csv")
    ag = pd.read_csv(xai / "method_agreement_spearman.csv", index_col=0)
    g = fa.groupby("method")
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Method & Deletion AUC $\downarrow$ & Insertion AUC $\uparrow$ & Mass on head region & $\rho$ after classifier rand. & $\rho$ after full rand. \\",
             r"\midrule"]
    for k in ["gradcam", "gradcampp", "integrated_gradients", "occlusion", "kernel_shap", "random"]:
        r = g.get_group(k)
        s1 = san[(san.method == k) & (san.stage == "classifier")].spearman.mean() if k in set(san.method) else float("nan")
        s2 = san[(san.method == k) & (san.stage == "conv_stem")].spearman.mean() if k in set(san.method) else float("nan")
        fmt = lambda v: "--" if v != v else f"{v:.2f}"
        lines.append(f"{LAB[k]} & {r.deletion_auc.mean():.3f} $\\pm$ {r.deletion_auc.std():.3f} & {r.insertion_auc.mean():.3f} $\\pm$ {r.insertion_auc.std():.3f} & "
                     f"{100 * r.inside_foreground.mean():.1f}\\% & {fmt(s1)} & {fmt(s2)} \\\\")
        key = "".join(w.capitalize() for w in k.split("_")).replace("Gradcampp", "GradcamPP")
        out[f"Del{key}"] = f"{r.deletion_auc.mean():.3f}"
        out[f"Ins{key}"] = f"{r.insertion_auc.mean():.3f}"
        out[f"Fg{key}"] = f"{100 * r.inside_foreground.mean():.1f}\\%"
        if k != "random":
            out[f"SanCls{key}"] = fmt(s1); out[f"SanAll{key}"] = fmt(s2)
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/xai.tex").write_text("\n".join(lines))
    out["AgGcPP"] = f"{ag.loc['gradcam', 'gradcampp']:.2f}"
    out["AgGcIG"] = f"{ag.loc['gradcam', 'integrated_gradients']:.2f}"
    out["AgGcOcc"] = f"{ag.loc['gradcam', 'occlusion']:.2f}"
    out["AgGcShap"] = f"{ag.loc['gradcam', 'kernel_shap']:.2f}"
    out["AgOccShap"] = f"{ag.loc['occlusion', 'kernel_shap']:.2f}"
    out["AgIGOcc"] = f"{ag.loc['integrated_gradients', 'occlusion']:.2f}"
    fgs = [g.get_group(k).inside_foreground.mean() for k in ("gradcam", "gradcampp", "occlusion", "kernel_shap")]
    out["FgMin"], out["FgMax"] = f"{100 * min(fgs):.0f}", f"{100 * max(fgs):.0f}"
    if "constant_map" in san:
        gc = san[san.method == "gradcam"]
        out["SanConstGc"], out["SanRowsGc"] = str(int(gc.constant_map.sum())), str(len(gc))
        out["SanConstAll"] = str(int(san.constant_map.sum()))
import re
DIG = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
out = {re.sub(r"\d", lambda mm: DIG[mm.group()], k): v for k, v in out.items()}
tex = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in out.items())
(ROOT / "paper/numbers.tex").write_text("% generated by scripts/make_paper_numbers.py, do not edit\n" + tex + "\n")
print(tex)
