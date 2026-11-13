"""Dataset discovery, deterministic preprocessing and caching."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

from . import CLASSES

KAGGLE_ID = "masoudnickparvar/brain-tumor-mri-dataset"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def download_dataset() -> Path:
    """Download (or reuse the cached copy of) the Kaggle dataset with kagglehub."""
    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_ID)
    return Path(path)


def list_images(root: Path) -> list[dict]:
    """Return one record per image: split, class, label index and path (sorted, deterministic)."""
    recs = []
    for split in ("Training", "Testing"):
        for ci, cls in enumerate(CLASSES):
            d = Path(root) / split / cls
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    recs.append({"split": split, "cls": cls, "label": ci, "path": str(p), "name": p.name})
    return recs


def load_gray_square(path: str, size: int) -> np.ndarray:
    """Load as greyscale, zero-pad to a square (keeps aspect ratio) and resize to size x size (uint8).

    The browser demo applies exactly the same steps (docs/app.js)."""
    im = Image.open(path).convert("L")
    w, h = im.size
    s = max(w, h)
    canvas = Image.new("L", (s, s), 0)
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return np.asarray(canvas.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def build_cache(recs: list[dict], size: int, cache_file: Path) -> np.ndarray:
    cache_file = Path(cache_file)
    if cache_file.exists():
        arr = np.load(cache_file)
        if arr.shape[0] == len(recs) and arr.shape[1] == size:
            return arr
    arr = np.stack([load_gray_square(r["path"], size) for r in recs])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, arr)
    return arr
