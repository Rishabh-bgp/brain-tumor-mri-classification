"""Leakage-aware split construction.

1. Hash every image (MD5 + 64-bit pHash) and compute a 64x64 thumbnail Pearson correlation for candidate pairs.
2. Link two images when the MD5 matches, when pHash Hamming distance <= HAMMING_MAX and the thumbnail
   correlation >= CORR_MIN (near-duplicate or near-identical neighbouring slice), or when Hamming distance <=
   HAMMING_MAX_HICORR and correlation >= CORR_HI (contrast- or sharpening-altered copies that move the pHash).
   Connected components = groups.
3. Remove from the training pool every Training image whose group touches the Testing folder (cross-split leakage)
   and every image in a group carrying more than one class label.
4. Split the remaining training pool into train/validation by group (stratified by class), so no group spans both.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import list_images, load_gray_square
from .dedup import UnionFind, file_md5, hamming_matrix, phash64

HAMMING_MAX = 6
CORR_MIN = 0.95
HAMMING_MAX_HICORR, CORR_HI = 10, 0.98  # v2: added after review (altered copies at Hamming 8-10)
STRICT_HAMMING, STRICT_CORR = 2, 0.98
VAL_FRACTION = 0.15


def hash_table(root: Path) -> pd.DataFrame:
    df = pd.DataFrame(list_images(root))
    df["relpath"] = df.split + "/" + df.cls + "/" + df.name
    df["md5"] = [file_md5(p) for p in df.path]
    df["phash"] = [f"{phash64(p):016x}" for p in df.path]
    return df


def candidate_pairs(df: pd.DataFrame, max_hamming: int = 10) -> pd.DataFrame:
    h = np.array([int(x, 16) for x in df.phash], dtype=np.uint64)
    thumbs = np.stack([load_gray_square(p, 64).astype(np.float32).ravel() for p in df.path])
    thumbs = (thumbs - thumbs.mean(1, keepdims=True)) / (thumbs.std(1, keepdims=True) + 1e-6)
    rows = []
    for s in range(0, len(h), 1024):
        d = hamming_matrix(h[s:s + 1024], h)
        ii, jj = np.nonzero(d <= max_hamming)
        keep = ii + s < jj
        ii, jj = ii[keep] + s, jj[keep]
        corr = (thumbs[ii] * thumbs[jj]).mean(1)
        rows.append(pd.DataFrame({"i": ii, "j": jj, "hamming": d[ii - s, jj], "corr": corr}))
    pairs = pd.concat(rows, ignore_index=True)
    pairs["md5_equal"] = df.md5.values[pairs.i] == df.md5.values[pairs.j]
    for side in ("i", "j"):
        pairs[f"path_{side}"] = df.relpath.values[pairs[side]]
    pairs["strict_dup"] = pairs.md5_equal | ((pairs.hamming <= STRICT_HAMMING) & (pairs["corr"] >= STRICT_CORR))
    pairs["linked_v1"] = pairs.md5_equal | ((pairs.hamming <= HAMMING_MAX) & (pairs["corr"] >= CORR_MIN))
    pairs["linked"] = pairs.linked_v1 | ((pairs.hamming <= HAMMING_MAX_HICORR) & (pairs["corr"] >= CORR_HI))
    pairs["cross_split"] = df.split.values[pairs.i] != df.split.values[pairs.j]
    pairs["label_conflict"] = df.cls.values[pairs.i] != df.cls.values[pairs.j]
    return pairs


def assign_groups(df: pd.DataFrame, pairs: pd.DataFrame, col: str = "linked") -> np.ndarray:
    uf = UnionFind(len(df))
    seen: dict[str, int] = {}
    for i, m in enumerate(df.md5):  # exact duplicates (also catches pairs beyond the pHash radius)
        if m in seen:
            uf.union(seen[m], i)
        else:
            seen[m] = i
    sel = pairs[pairs[col]]
    for a, b in zip(sel.i, sel.j):
        uf.union(int(a), int(b))
    return np.array([uf.find(i) for i in range(len(df))])


def make_splits(df: pd.DataFrame, pairs: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    df["group"] = assign_groups(df, pairs, "linked")
    df["group_strict"] = assign_groups(df, pairs, "strict_dup")
    te = df.split == "Testing"
    test_groups = set(df.group[te])
    conflict = df.groupby("group").cls.nunique()
    conflict_groups = set(conflict[conflict > 1].index)
    df["leaks_into_test"] = (~te) & df.group.isin(test_groups)
    df["label_conflict"] = df.group.isin(conflict_groups)
    df["test_has_train_dup"] = te & df.group.isin(set(df.group[~te]))
    strict_train = set(df.group_strict[~te])
    df["test_has_train_dup_strict"] = te & df.group_strict.isin(strict_train)
    df["subset"] = np.where(te, "test", "")
    pool = (~te) & ~df.leaks_into_test & ~df.label_conflict
    df.loc[(~te) & ~pool, "subset"] = "excluded"
    rng = np.random.default_rng(seed)
    for cls in sorted(df.cls.unique()):
        groups = np.array(sorted(df[pool & (df.cls == cls)].group.unique()))
        rng.shuffle(groups)
        n_imgs = (pool & (df.cls == cls)).sum()
        val_groups, count = [], 0
        sizes = df[pool & (df.cls == cls)].group.value_counts()
        for g in groups:
            if count >= VAL_FRACTION * n_imgs:
                break
            val_groups.append(g)
            count += sizes[g]
        m = pool & (df.cls == cls)
        df.loc[m, "subset"] = np.where(df.loc[m, "group"].isin(val_groups), "val", "train")
    # one representative per group inside the test set (for the deduplicated test variant)
    df["test_dedup_rep"] = False
    reps = df[te].sort_values("relpath").groupby("group").head(1).index
    df.loc[reps, "test_dedup_rep"] = True
    return df
