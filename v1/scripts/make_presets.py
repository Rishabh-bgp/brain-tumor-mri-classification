"""Build the web demo's preset gallery: two Testing images per class that belong to NO duplicate pair at all
(their linked group has size 1 across the whole dataset), with precomputed multi-method explanations.

Dataset licence: CC BY 4.0 (Masoud Nickparvar, Kaggle), which permits redistribution with attribution."""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btmri import CLASSES  # noqa: E402
from btmri.data import download_dataset, load_gray_square, seed_everything  # noqa: E402
from btmri.model import build_model  # noqa: E402
from btmri.xai import METHODS, GreyModel, attention_summary, explain, rank_corr  # noqa: E402

seed_everything(42)
groups = pd.read_csv(ROOT / "results/review/hashes_and_groups.csv")
size = groups.group.map(groups.group.value_counts())
pairs = pd.read_csv(ROOT / "results/review/candidate_pairs_hamming_le10.csv.gz")
in_any_pair = set(pairs[pairs.linked | pairs.strict_dup].path_i) | set(pairs[pairs.linked | pairs.strict_dup].path_j)
solo = set(groups.relpath[(size == 1) & (groups.split == "Testing")]) - in_any_pair
pred = pd.read_csv(ROOT / "results/predictions/clean_test_predictions.csv")
pool = pred[pred.relpath.isin(solo)]
model = build_model(pretrained=False)
model.load_state_dict(torch.load(ROOT / "results/runs/clean/best.pt", map_location="cpu"))
gm = GreyModel(model.eval()).eval()
root = download_dataset()
DOC = ROOT / "docs"
(DOC / "samples").mkdir(parents=True, exist_ok=True)
for f in (DOC / "samples").glob("*.jpg"):
    f.unlink()
presets = []
for c in CLASSES:
    for _, r in pool[pool.true == c].sample(2, random_state=11).iterrows():
        x = torch.from_numpy(load_gray_square(str(root / r.relpath), 224)).float().div(255)[None, None]
        tgt = CLASSES.index(r.pred)
        maps = {m: explain(gm, x, tgt, m) for m in METHODS}
        name = Path(r.relpath).stem + ".jpg"
        im = Image.open(root / r.relpath).convert("L")
        im.thumbnail((256, 256))
        im.save(DOC / "samples" / name, quality=80, optimize=True)
        enc = {}
        for m, s in maps.items():
            small = F.adaptive_avg_pool2d(torch.tensor(s)[None, None], 56)[0, 0].numpy()
            small = (255 * (small - small.min()) / max(small.max() - small.min(), 1e-9)).astype(np.uint8)
            enc[m] = base64.b64encode(small.tobytes()).decode()
        presets.append({"file": f"samples/{name}", "source": r.relpath, "true": r.true, "torch_pred": r.pred,
                        "torch_probs": [float(r[f"p_{k}"]) for k in CLASSES], "maps56": enc,
                        "agreement_with_gradcam": {m: rank_corr(maps["gradcam"], maps[m]) for m in METHODS if m != "gradcam"},
                        "attention": {m: attention_summary(maps[m], x[0, 0].numpy()) for m in METHODS}})
        print(r.relpath, r.true, "->", r.pred, round(r.confidence, 3), flush=True)
(DOC / "presets.json").write_text(json.dumps(presets))
print("pool size (Testing images in no duplicate pair):", len(pool), pool.true.value_counts().to_dict())
