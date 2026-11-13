"""Turn saved logits into the single source of truth: results/predictions/<run>_test_predictions.csv.

Temperature and the 'uncertain' threshold are fitted on the validation split only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btmri import CLASSES  # noqa: E402
from btmri.calibrate import choose_threshold, ece, fit_temperature, norm_entropy, softmax  # noqa: E402

out = ROOT / "results" / "predictions"
out.mkdir(parents=True, exist_ok=True)
groups = pd.read_csv(ROOT / "results/review/hashes_and_groups.csv").set_index("relpath")
calib = {}
for run in ("clean", "naive"):
    d = ROOT / "results/runs" / run
    if not (d / "test_logits.csv").exists():
        continue
    lcols = [f"logit_{c}" for c in CLASSES]
    val, test = pd.read_csv(d / "val_logits.csv"), pd.read_csv(d / "test_logits.csv")
    t = fit_temperature(val[lcols].values, val.true.values)
    pv = softmax(val[lcols].values, t)
    thr = choose_threshold(pv, val.true.values)
    raw, p = softmax(test[lcols].values, 1.0), softmax(test[lcols].values, t)
    df = pd.DataFrame({"relpath": test.relpath, "true": [CLASSES[i] for i in test.true]})
    df["pred"] = [CLASSES[i] for i in p.argmax(1)]
    for k, c in enumerate(CLASSES):
        df[f"p_{c}"] = p[:, k].round(6)
    for k, c in enumerate(CLASSES):
        df[f"raw_p_{c}"] = raw[:, k].round(6)
    df["confidence"] = p.max(1).round(6)
    df["norm_entropy"] = norm_entropy(p).round(6)
    df["uncertain"] = df.confidence < thr["conf_threshold"]
    g = groups.loc[df.relpath]
    for col in ("test_has_train_dup_strict", "test_has_train_dup", "test_dedup_rep"):
        df[col] = g[col].values
    df.to_csv(out / f"{run}_test_predictions.csv", index=False)
    y = test.true.values
    calib[run] = {"temperature": t, "val_ece_raw": ece(softmax(val[lcols].values), val.true.values),
                  "val_ece_calibrated": ece(pv, val.true.values), "uncertainty": thr,
                  "val_accuracy": float((pv.argmax(1) == val.true.values).mean())}
    print(run, json.dumps(calib[run], indent=1))
(out / "calibration.json").write_text(json.dumps(calib, indent=2))
