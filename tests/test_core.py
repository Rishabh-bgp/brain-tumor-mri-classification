import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2" / "src"))  # tests cover the current (v2) code

from btmri.calibrate import choose_threshold, ece, fit_temperature, norm_entropy, softmax  # noqa: E402
from btmri.dedup import UnionFind, hamming_matrix  # noqa: E402


def test_hamming_matrix_matches_python_popcount():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2**63, 20, dtype=np.uint64)
    b = rng.integers(0, 2**63, 15, dtype=np.uint64)
    d = hamming_matrix(a, b)
    for i in range(20):
        for j in range(15):
            assert d[i, j] == bin(int(a[i]) ^ int(b[j])).count("1")


def test_union_find_groups():
    uf = UnionFind(5)
    uf.union(0, 3); uf.union(3, 4)
    assert uf.find(4) == uf.find(0) != uf.find(1)


def test_temperature_and_threshold():
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=2.0, size=(4000, 4))
    pt = softmax(logits)
    y = np.array([rng.choice(4, p=q) for q in pt])  # labels drawn from calibrated probabilities
    t = fit_temperature(logits * 3, y)  # logits scaled x3 are over-confident: T should be ~3
    assert 2.4 < t < 3.6
    p = softmax(logits, 1.0)
    assert np.allclose(p.sum(1), 1)
    assert 0 <= ece(p, y) <= 1
    assert np.all((norm_entropy(p) >= 0) & (norm_entropy(p) <= 1 + 1e-9))
    thr = choose_threshold(p, y, target_acc=0.9)
    assert 0.5 <= thr["conf_threshold"] <= 0.99


SPLITS = ROOT / "v2/results/review/hashes_and_groups.csv"


@pytest.mark.skipif(not SPLITS.exists(), reason="split file not built")
def test_no_group_crosses_train_val_test():
    df = pd.read_csv(SPLITS)
    used = df[df.subset.isin(["train", "val", "test"])]
    per_group = used.groupby("group").subset.nunique()
    assert (per_group == 1).all(), "a duplicate group spans more than one of train/val/test"
    assert not df[df.subset.isin(["train", "val"])].label_conflict.any()
