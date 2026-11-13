"""Check that every demo preset in docs/samples/ reproduces the saved prediction file.

Runs the shipped ONNX model (docs/model/) on each docs/samples/* file with the same pre-processing as the browser,
applies the fitted temperature and threshold from docs/stats.json, and compares with
results/review/test_predictions.csv: every class probability within 0.01 and the same 'uncertain' flag.
The browser itself is checked the same way by tests/check_samples_browser.py (Playwright, optional).
Exit code 1 on any mismatch."""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # the v2/ folder
DOCS = ROOT.parent / "docs"  # GitHub Pages demo, shared at the repository root
sys.path.insert(0, str(ROOT / "src"))
from btmri import CLASSES  # noqa: E402
from btmri.data import load_gray_square  # noqa: E402

TOL = 0.01
stats = json.loads((DOCS / "stats.json").read_text())
T, tau = stats["temperature"], stats["conf_threshold"]
presets = json.loads((DOCS / "presets.json").read_text())
pred = pd.read_csv(ROOT / "results/review/test_predictions.csv").set_index("relpath")
sess = ort.InferenceSession(str(DOCS / "model/brain_tumor_effnetb0.onnx"), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name

ok, rows = True, []
for p in presets:
    x = load_gray_square(str(DOCS / p["file"]), 224).astype(np.float32)[None, None] / 255.0
    logits = sess.run(["logits"], {inp: x})[0][0] / T
    prob = np.exp(logits - logits.max()); prob /= prob.sum()
    ref = pred.loc[p["source"]]
    refp = np.array([ref[f"p_{c}"] for c in CLASSES])
    diff = float(np.abs(prob - refp).max())
    unc = bool(prob.max() < tau)
    good = diff <= TOL and unc == bool(ref["uncertain"]) and CLASSES[int(prob.argmax())] == ref["pred"]
    ok &= good
    rows.append({"sample": p["file"], "true": ref["true"], "pred": CLASSES[int(prob.argmax())], "csv_pred": ref["pred"],
                 "conf": round(float(prob.max()), 4), "csv_conf": round(float(refp.max()), 4), "max_abs_diff": round(diff, 5),
                 "uncertain": unc, "csv_uncertain": bool(ref["uncertain"]), "ok": good})
print(pd.DataFrame(rows).to_string(index=False))
print("ALL OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
