"""Explanation methods, sanity checks and faithfulness metrics.

All methods explain a GreyModel: input (N,1,224,224) in [0,1] -> logits (N,4). Every map is returned as a
non-negative (224,224) float array scaled to [0,1] (absolute values for signed attributions).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from skimage.segmentation import slic

from .model import MEAN, STD

METHODS = ["gradcam", "gradcampp", "integrated_gradients", "occlusion", "kernel_shap"]
METHOD_LABELS = {"gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++", "integrated_gradients": "Integrated Gradients",
                 "occlusion": "Occlusion", "kernel_shap": "KernelSHAP (superpixels)", "lime": "LIME (superpixels)", "random": "Random"}


class GreyModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def features(self, x):
        return self.model.forward_features((x.expand(-1, 3, -1, -1) - self.mean) / self.std)

    def forward(self, x):
        return self.model.forward_head(self.features(x))


def _norm(m: np.ndarray) -> np.ndarray:
    m = np.nan_to_num(np.asarray(m, dtype=np.float64))
    m = m - m.min()
    return (m / m.max()) if m.max() > 0 else m


def _up(cam: torch.Tensor) -> np.ndarray:
    return F.interpolate(cam[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0].detach().numpy()


def gradcam_family(gm: GreyModel, x: torch.Tensor, target: int, plus: bool = False) -> np.ndarray:
    x = x.clone().requires_grad_(False)
    feats = gm.features(x)
    feats.retain_grad()
    score = gm.model.forward_head(feats)[0, target]
    grads = torch.autograd.grad(score, feats)[0][0]  # (C,h,w)
    A = feats[0].detach()
    if not plus:
        w = grads.mean((1, 2))
    else:  # Grad-CAM++ (Chattopadhay et al., 2018), exponential-score closed form
        g2, g3 = grads ** 2, grads ** 3
        denom = 2 * g2 + (A * g3).sum((1, 2), keepdim=True)
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        w = (alpha * F.relu(grads)).sum((1, 2))
    cam = F.relu((w[:, None, None] * A).sum(0))
    return _norm(_up(cam))


def integrated_gradients(gm, x, target, steps=32):
    from captum.attr import IntegratedGradients
    attr = IntegratedGradients(gm).attribute(x, baselines=torch.zeros_like(x), target=target, n_steps=steps,
                                             internal_batch_size=16)
    return _norm(attr[0, 0].abs().detach().numpy())


def occlusion(gm, x, target, window=32, stride=16):
    from captum.attr import Occlusion
    with torch.no_grad():
        attr = Occlusion(gm).attribute(x, target=target, sliding_window_shapes=(1, window, window),
                                       strides=(1, stride, stride), baselines=0.0, perturbations_per_eval=64)
    return _norm(np.clip(attr[0, 0].numpy(), 0, None))


def superpixels(x: torch.Tensor, n: int = 64) -> np.ndarray:
    img = x[0, 0].numpy()
    return slic(img, n_segments=n, compactness=0.1, channel_axis=None, start_label=0)


def kernel_shap(gm, x, target, n_samples=200, seed=0):
    from captum.attr import KernelShap
    seg = torch.from_numpy(superpixels(x)).long()[None, None]
    torch.manual_seed(seed)
    with torch.no_grad():
        attr = KernelShap(gm).attribute(x, baselines=0.0, target=target, feature_mask=seg, n_samples=n_samples,
                                        perturbations_per_eval=50)
    return _norm(np.clip(attr[0, 0].numpy(), 0, None))


def lime(gm, x, target, n_samples=200, seed=0, kernel_width=0.25):
    """LIME (Ribeiro et al., 2016) on the same SLIC superpixels and black baseline as kernel_shap.

    As in the reference `lime` image explainer: each perturbation switches a random subset of superpixels off
    (Captum's default: each kept with probability 0.5), samples are weighted by an exponential kernel on the cosine
    distance between the binary superpixel vector and the all-on vector (kernel width 0.25), and a weighted ridge
    regression (alpha = 1) is fitted to the class logit. Positive coefficients are kept, as for KernelSHAP."""
    from captum._utils.models.linear_model import SkLearnRidge
    from captum.attr import Lime

    def similarity(original_inp, perturbed_inp, perturbed_interpretable_inp, **kwargs):
        z = perturbed_interpretable_inp.float().flatten()
        cos = z.sum() / (z.norm() * z.numel() ** 0.5 + 1e-12)  # cosine with the all-ones vector
        d = 1.0 - cos
        return torch.exp(-(d ** 2) / kernel_width ** 2)

    seg = torch.from_numpy(superpixels(x)).long()[None, None]
    torch.manual_seed(seed)
    with torch.no_grad():
        attr = Lime(gm, interpretable_model=SkLearnRidge(alpha=1.0), similarity_func=similarity).attribute(
            x, target=target, baselines=0.0, feature_mask=seg, n_samples=n_samples, perturbations_per_eval=50)
    return _norm(np.clip(attr[0, 0].numpy(), 0, None))


def explain(gm, x, target, method):
    gm.eval()
    if method == "gradcam":
        return gradcam_family(gm, x, target, plus=False)
    if method == "gradcampp":
        return gradcam_family(gm, x, target, plus=True)
    if method == "integrated_gradients":
        return integrated_gradients(gm, x, target)
    if method == "occlusion":
        return occlusion(gm, x, target)
    if method == "kernel_shap":
        return kernel_shap(gm, x, target)
    if method == "lime":
        return lime(gm, x, target)
    if method == "random":
        return np.random.default_rng(int(x.sum().item() * 1000) % 2**32).random((224, 224))
    raise ValueError(method)


# --------------------------- sanity check (Adebayo et al., 2018) ---------------------------
def cascade_stages(model: nn.Module) -> list[tuple[str, list[str]]]:
    """Top-down cumulative randomisation stages for timm EfficientNet-B0."""
    names = [("classifier", ["classifier"]), ("conv_head", ["conv_head", "bn2"])]
    for i in reversed(range(len(model.blocks))):
        names.append((f"blocks.{i}", [f"blocks.{i}"]))
    names.append(("conv_stem", ["conv_stem", "bn1"]))
    return names


def randomise(module: nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for m in module.modules():
        for name, p in list(m.named_parameters(recurse=False)):
            with torch.no_grad():
                if p.dim() > 1:
                    p.copy_(torch.randn(p.shape, generator=g) * (2.0 / p[0].numel()) ** 0.5)
                elif "weight" in name:
                    p.fill_(1.0)
                else:
                    p.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.zero_(); m.running_var.fill_(1.0)


def rank_corr(a: np.ndarray, b: np.ndarray, size: int = 56) -> float:
    ta = F.adaptive_avg_pool2d(torch.tensor(a)[None, None], size)[0, 0].numpy().ravel()
    tb = F.adaptive_avg_pool2d(torch.tensor(b)[None, None], size)[0, 0].numpy().ravel()
    r = spearmanr(ta, tb).statistic
    return float(0.0 if np.isnan(r) else r)


def cascading_randomisation(model, xs, targets, methods, stages_to_use=None, seed=0):
    """Return rows (stage, method, image_idx, spearman) comparing maps of progressively randomised models."""
    base = GreyModel(model).eval()
    ref = {(m, i): explain(base, xs[i:i + 1], targets[i], m) for m in methods for i in range(len(xs))}
    rows = []
    rnd = copy.deepcopy(model)
    stages = cascade_stages(model)
    if stages_to_use:
        keep = set(stages_to_use)
    for k, (stage, mods) in enumerate(stages):
        for mname in mods:
            randomise(rnd.get_submodule(mname), seed + k)
        if stages_to_use and stage not in keep:
            continue
        gm = GreyModel(rnd).eval()
        for m in methods:
            for i in range(len(xs)):
                e = explain(gm, xs[i:i + 1], targets[i], m)
                rows.append({"stage": stage, "method": m, "image": i, "spearman": rank_corr(ref[(m, i)], e),
                             "constant_map": bool(np.ptp(e) == 0 or np.ptp(ref[(m, i)]) == 0)})
    return rows


# --------------------------- deletion / insertion (Petsiuk et al., 2018) ---------------------------
def blur(x: torch.Tensor, k: int = 31, sigma: float = 10.0) -> torch.Tensor:
    ax = torch.arange(k) - k // 2
    g = torch.exp(-ax.float() ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))


def deletion_insertion(gm, x, target, sal: np.ndarray, steps: int = 32) -> tuple[float, float]:
    # ties (e.g. zeros after ReLU/clipping) are broken randomly with a fixed seed, not in raster order
    jitter = np.random.default_rng(0).uniform(0, 1e-9, sal.size)
    order = np.argsort(-(sal.ravel() + jitter), kind="stable")
    n = order.size
    blurred = blur(x)
    dels, ins = [], []
    for s in range(steps + 1):
        mask = torch.zeros(n)
        mask[torch.from_numpy(order[: int(round(n * s / steps))].copy())] = 1
        mask = mask.view(1, 1, 224, 224)
        dels.append(x * (1 - mask))
        ins.append(blurred * (1 - mask) + x * mask)
    with torch.no_grad():
        pd_ = F.softmax(gm(torch.cat(dels)), 1)[:, target].numpy()
        pi_ = F.softmax(gm(torch.cat(ins)), 1)[:, target].numpy()
    t = np.linspace(0, 1, steps + 1)
    return float(np.trapezoid(pd_, t)), float(np.trapezoid(pi_, t))


def brain_mask(x: np.ndarray, thr: float = 0.08) -> np.ndarray:
    """Crude foreground mask (pixels brighter than thr after a 7x7 box blur); used for the 'inside brain' share."""
    t = torch.tensor(x, dtype=torch.float32)[None, None]
    return (F.avg_pool2d(t, 7, 1, 3)[0, 0].numpy() > thr)


def attention_summary(sal: np.ndarray, img: np.ndarray) -> dict:
    s = sal / (sal.sum() + 1e-12)
    m = brain_mask(img)
    h = w = 224
    quads = {"upper-left": s[:h // 2, :w // 2].sum(), "upper-right": s[:h // 2, w // 2:].sum(),
             "lower-left": s[h // 2:, :w // 2].sum(), "lower-right": s[h // 2:, w // 2:].sum()}
    q = max(quads, key=quads.get)
    return {"inside_foreground": float(s[m].sum()), "top_quadrant": q, "top_quadrant_share": float(quads[q])}
