"""Temperature scaling (Guo et al., 2017) and the val-chosen uncertainty threshold."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def softmax(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = logits / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def norm_entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(1) / np.log(p.shape[1])


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def choose_threshold(p_val: np.ndarray, y_val: np.ndarray, target_acc: float = 0.99,
                     grid=np.round(np.arange(0.50, 0.991, 0.01), 2)) -> dict:
    """Smallest calibrated-confidence threshold whose retained validation predictions reach target_acc.

    Predictions below the threshold are flagged 'uncertain'. If no threshold in the grid reaches the target,
    the threshold maximising retained accuracy is used."""
    conf, pred = p_val.max(1), p_val.argmax(1)
    rows = []
    for t in grid:
        m = conf >= t
        rows.append((float(t), float((pred[m] == y_val[m]).mean()) if m.any() else np.nan, float(m.mean())))
    ok = [r for r in rows if r[1] >= target_acc]
    t, acc, cov = ok[0] if ok else max(rows, key=lambda r: (r[1], r[2]))
    return {"conf_threshold": t, "target_val_accuracy": target_acc, "val_retained_accuracy": acc,
            "val_coverage": cov, "rule": f"smallest t in [0.50,0.99] (step 0.01) with val accuracy >= {target_acc} on predictions with calibrated confidence >= t"}
