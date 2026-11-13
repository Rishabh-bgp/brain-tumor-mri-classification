"""Train EfficientNet-B0 on cached 224x224 greyscale MRI slices (CPU friendly).

python -m btmri.train --run clean   # leakage-aware split (default)
python -m btmri.train --run naive   # original Training folder, random image-level 85/15 split (for comparison)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torchvision.transforms import v2 as T

from .data import build_cache, download_dataset, seed_everything
from .model import IMG_SIZE, build_model, to_input

ROOT = Path(__file__).resolve().parents[2]


def load_split_table(run: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results/review/hashes_and_groups.csv")
    if run == "naive":
        rng = np.random.default_rng(seed)
        tr = df.split == "Training"
        df["subset"] = np.where(tr, "train", "test")
        idx = np.nonzero(tr.values)[0]
        for c in sorted(df.cls.unique()):
            ci = idx[df.cls.values[idx] == c]
            v = rng.choice(ci, int(round(0.15 * len(ci))), replace=False)
            df.loc[v, "subset"] = "val"
    return df


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, X, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            out.append(model(to_input(torch.from_numpy(X[s:s + bs]).to(DEVICE))).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="clean", choices=["clean", "naive"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    seed_everything(a.seed)
    torch.set_num_threads(8)
    out = ROOT / "results" / "runs" / a.run
    out.mkdir(parents=True, exist_ok=True)
    root = download_dataset()
    df = load_split_table(a.run, a.seed)
    df["path"] = [str(root / p) for p in df.relpath]
    X = build_cache(df.to_dict("records"), IMG_SIZE, ROOT.parent / "cache" / f"all_{IMG_SIZE}.npy")
    y = df.label.values
    tr, va, te = (np.nonzero((df.subset == s).values)[0] for s in ("train", "val", "test"))
    print(f"[{a.run}] device={DEVICE} train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    aug = T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
    ])
    counts = np.bincount(y[tr], minlength=4)
    cw = torch.tensor(len(tr) / (4 * counts), dtype=torch.float32).to(DEVICE)  # inverse-frequency class weights
    print("class counts", counts.tolist(), "weights", [round(float(w), 3) for w in cw], flush=True)
    model = build_model(pretrained=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * int(np.ceil(len(tr) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(a.seed)
    hist, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        perm = torch.randperm(len(tr), generator=g).numpy()
        for s in range(0, len(perm), a.bs):
            b = tr[perm[s:s + a.bs]]
            xb = torch.from_numpy(X[b]).unsqueeze(1)
            xb = torch.stack([aug(x) for x in xb])
            logits = model(to_input(xb.to(DEVICE)))
            loss = F.cross_entropy(logits, torch.from_numpy(y[b]).long().to(DEVICE), weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
            n += len(b)
        vl = predict(model, X[va])
        vloss = F.cross_entropy(torch.from_numpy(vl), torch.from_numpy(y[va]).long()).item()
        vacc = float((vl.argmax(1) == y[va]).mean())
        vf1 = float(f1_score(y[va], vl.argmax(1), average="macro"))
        hist.append({"epoch": ep, "train_loss": tot / n, "val_loss": vloss, "val_acc": vacc, "val_macro_f1": vf1,
                     "lr": sched.get_last_lr()[0], "seconds": time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        if vf1 > best:
            best = vf1
            torch.save(model.state_dict(), out / "best.pt")
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    model.load_state_dict(torch.load(out / "best.pt", map_location=DEVICE))
    for name, idx in (("val", va), ("test", te)):
        lg = predict(model, X[idx])
        p = pd.DataFrame(lg, columns=[f"logit_{c}" for c in ("glioma", "meningioma", "notumor", "pituitary")])
        p.insert(0, "relpath", df.relpath.values[idx])
        p.insert(1, "true", y[idx])
        p.to_csv(out / f"{name}_logits.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
