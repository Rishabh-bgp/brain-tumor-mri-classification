"""Audit the files named '*-aug-*' (augmented meningioma images shipped in both folders).

For each, find the closest non-augmented image under the 8 flips/90-degree rotations (pHash), verify with thumbnail
correlation, and compare model accuracy on augmented vs other Testing meningiomas."""
import json
import sys
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btmri.data import download_dataset  # noqa: E402
from btmri.dedup import hamming_matrix  # noqa: E402

root = download_dataset()
g = pd.read_csv(ROOT / "results/review/hashes_and_groups.csv")
g["aug"] = g.relpath.str.contains("-aug-")
H = np.array([int(x, 16) for x in g.phash], dtype=np.uint64)
na = np.nonzero(~g.aug.values)[0]
T = {"identity": lambda x: x, "flip_lr": ImageOps.mirror, "flip_ud": ImageOps.flip,
     "rot90": lambda x: x.rotate(90, expand=True), "rot180": lambda x: x.rotate(180),
     "rot270": lambda x: x.rotate(270, expand=True), "transpose": lambda x: x.transpose(Image.TRANSPOSE),
     "transverse": lambda x: x.transpose(Image.TRANSVERSE)}


def thumb(im):
    w, h = im.size
    s = max(w, h)
    c = Image.new("L", (s, s), 0)
    c.paste(im, ((s - w) // 2, (s - h) // 2))
    a = np.asarray(c.resize((64, 64), Image.BILINEAR), dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


rows = []
for i in np.nonzero(g.aug.values)[0]:
    im = Image.open(root / g.relpath[i]).convert("L")
    best = (99, -1, "")
    for name, t in T.items():
        d = hamming_matrix(np.array([int(str(imagehash.phash(t(im))), 16)], dtype=np.uint64), H[na])[0]
        j = int(d.argmin())
        if d[j] < best[0]:
            best = (int(d[j]), int(na[j]), name)
    corr = float((thumb(T[best[2]](im)) * thumb(Image.open(root / g.relpath[best[1]]).convert("L"))).mean())
    rows.append({"aug_file": g.relpath[i], "best_match": g.relpath[best[1]], "transform": best[2], "hamming": best[0], "corr": corr})
r = pd.DataFrame(rows)
r["strict"] = (r.hamming <= 2) & (r["corr"] >= 0.98)
r["linked"] = (r.hamming <= 6) & (r["corr"] >= 0.95)
r.to_csv(ROOT / "results/review/augmented_meningioma_matches.csv", index=False)
out = {"n_aug_training": int(r.aug_file.str.startswith("Training").sum()), "n_aug_testing": int(r.aug_file.str.startswith("Testing").sum()),
       "linked_any": int(r.linked.sum()), "linked_non_identity": int((r.linked & (r["transform"] != "identity")).sum())}
for run in ("clean", "naive"):
    p = pd.read_csv(ROOT / f"results/predictions/{run}_test_predictions.csv")
    men = p[p.true == "meningioma"]
    aug = men.relpath.str.contains("-aug-")
    out[f"{run}_acc_test_aug_meningioma"] = float((men[aug].true == men[aug].pred).mean())
    out[f"{run}_acc_test_other_meningioma"] = float((men[~aug].true == men[~aug].pred).mean())
    out[f"{run}_n_aug"], out[f"{run}_n_other"] = int(aug.sum()), int((~aug).sum())
    # fair comparison: only meningiomas WITHOUT a strict near-duplicate in Training
    nd = men[~men.test_has_train_dup_strict]
    a_nd = nd.relpath.str.contains("-aug-")
    out[f"{run}_nodup_aug_correct"], out[f"{run}_nodup_aug_n"] = int((nd[a_nd].true == nd[a_nd].pred).sum()), int(a_nd.sum())
    out[f"{run}_nodup_other_correct"], out[f"{run}_nodup_other_n"] = int((nd[~a_nd].true == nd[~a_nd].pred).sum()), int((~a_nd).sum())
(ROOT / "results/review/augmented_summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
