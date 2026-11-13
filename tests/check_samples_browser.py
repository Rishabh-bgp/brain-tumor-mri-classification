"""Browser check of the demo presets (optional; needs Playwright and Chrome):
python tests/check_samples_browser.py [URL]  ->  each preset's in-browser probabilities vs results/review/test_predictions.csv
(within 0.01, same uncertain flag), plus a responsiveness probe while occlusion runs."""
import json
import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/"
CL = ["glioma", "meningioma", "notumor", "pituitary"]
pred = pd.read_csv(ROOT / "v2/results/review/test_predictions.csv").set_index("relpath")
presets = json.loads((ROOT / "docs/presets.json").read_text())
tau = json.loads((ROOT / "docs/stats.json").read_text())["conf_threshold"]
ok = True
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1280, "height": 900})
    pg.goto(URL + ("&" if "?" in URL else "?") + "nocache=1")
    pg.wait_for_function("document.getElementById('status').textContent.includes('Model ready')", timeout=90000)
    for i, pr in enumerate(presets):
        pg.evaluate("document.getElementById('status').textContent=''")
        pg.evaluate(f"document.querySelectorAll('.sample')[{i}].click()")
        pg.wait_for_function("document.getElementById('status').textContent.startsWith('Done')", timeout=30000)
        probs = pg.evaluate("Array.from(window.__neuro.cur.probs)")
        badge = pg.inner_text("#verdict")
        ref = pred.loc[pr["source"]]
        diff = max(abs(a - ref[f"p_{c}"]) for a, c in zip(probs, CL))
        unc = max(probs) < tau
        good = diff <= 0.01 and unc == bool(ref["uncertain"]) and (("uncertain" in badge) == unc)
        ok &= good
        print(f"{pr['file']:24s} browser_conf={max(probs):.4f} csv_conf={max(ref[f'p_{c}'] for c in CL):.4f} "
              f"max_diff={diff:.5f} uncertain={unc}/{bool(ref['uncertain'])} badge={'uncertain' if 'uncertain' in badge else 'confident'} {'OK' if good else 'FAIL'}")
    # responsiveness while occlusion runs: count timer ticks and distinct progress texts
    pg.evaluate("""window.__ticks = 0; window.__seen = new Set(); window.__iv = setInterval(() => { window.__ticks++;
                   window.__seen.add(document.getElementById('status').textContent); }, 50);""")
    pg.click("button:has-text('Occlusion')")
    pg.wait_for_function("document.getElementById('status').textContent.startsWith('Occlusion done')", timeout=180000)
    ticks, seen = pg.evaluate("[window.__ticks, Array.from(window.__seen).filter(s => s.startsWith('Occlusion ')).length]")
    print(f"occlusion: timer ticks during run={ticks}, distinct progress texts observed={seen}")
    ok &= ticks > 10 and seen > 3
    b.close()
print("ALL OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
