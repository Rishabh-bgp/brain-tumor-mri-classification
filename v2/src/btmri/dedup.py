"""Exact and near-duplicate detection with MD5 and 64-bit perceptual hashes (pHash)."""
from __future__ import annotations

import hashlib

import imagehash
import numpy as np
from PIL import Image


def file_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def phash64(path: str) -> int:
    """64-bit DCT perceptual hash of the greyscale image (imagehash.phash, hash_size=8)."""
    h = imagehash.phash(Image.open(path).convert("L"), hash_size=8)
    return int(str(h), 16)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between two uint64 hash vectors (uint8 matrix)."""
    a = a.astype(np.uint64)[:, None]
    b = b.astype(np.uint64)[None, :]
    x = np.bitwise_xor(a, b)
    out = np.zeros(x.shape, dtype=np.uint8)
    for shift in range(0, 64, 8):
        byte = ((x >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.uint8)
        out += np.unpackbits(byte[..., None], axis=-1).sum(-1).astype(np.uint8)
    return out


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.p[max(ri, rj)] = min(ri, rj)


def duplicate_groups(hashes: np.ndarray, md5s: list[str], threshold: int) -> np.ndarray:
    """Group ids linking images whose MD5 matches or whose pHash distance <= threshold."""
    n = len(hashes)
    uf = UnionFind(n)
    seen: dict[str, int] = {}
    for i, m in enumerate(md5s):
        if m in seen:
            uf.union(seen[m], i)
        else:
            seen[m] = i
    if threshold >= 0:
        block = 1024
        for s in range(0, n, block):
            d = hamming_matrix(hashes[s:s + block], hashes)
            ii, jj = np.nonzero(d <= threshold)
            for i, j in zip(ii + s, jj):
                if i < j:
                    uf.union(int(i), int(j))
    return np.array([uf.find(i) for i in range(n)])
