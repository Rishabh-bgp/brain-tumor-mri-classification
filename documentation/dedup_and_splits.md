# Duplicate audit and leakage-aware split

[Documentation home](README.md) · [Top-level README](../README.md) · [v2 README](../v2/README.md)

Code: [`v2/src/btmri/dedup.py`](../v2/src/btmri/dedup.py), [`v2/src/btmri/splits.py`](../v2/src/btmri/splits.py),
[`v2/scripts/build_splits.py`](../v2/scripts/build_splits.py); notebook
[`01_data_and_deduplication.ipynb`](../v2/notebooks/01_data_and_deduplication.ipynb).

## Signals
For each of the 7,200 images:
- **MD5** of the file bytes (exact duplicates);
- **pHash**: a 64-bit DCT perceptual hash of the greyscale image (`imagehash.phash`); the Hamming distance between two
  hashes counts differing bits (0 = identical hash);
- **thumbnail correlation**: the Pearson correlation of 64×64 zero-padded greyscale thumbnails.

Every pair within Hamming distance 10 is a *candidate pair* (31,956 pairs, all released in
[`candidate_pairs_hamming_le10.csv.gz`](../v2/results/review/candidate_pairs_hamming_le10.csv.gz)). pHash alone is too coarse
for MRI: 2,837 candidate pairs are within Hamming 6 but have correlation below 0.95, mostly different slices of similar
anatomy. The correlation check separates them.

## Rules
| Relation | Rule |
|---|---|
| **strict near-duplicate** | identical MD5, or Hamming ≤ 2 **and** correlation ≥ 0.98 |
| **linked (v1)** | identical MD5, or Hamming ≤ 6 **and** correlation ≥ 0.95 |
| **linked (v2)** | linked (v1), **or** Hamming ≤ 10 **and** correlation ≥ 0.98 |

Connected components of linked images form **groups** (union-find). v2 added the third clause after a review found
contrast- or sharpening-altered copies of Testing images at Hamming 8–10 that v1 missed; it adds
215 linked pairs (71 across Training/Testing).

| Count | v1 | v2 |
|---|---|---|
| Strict pairs (across Training/Testing) | 3,484 (1,270) | 3,484 (1,270) |
| Linked pairs (across) | 5,962 (2,158) | 6,177 (2,229) |
| Groups | 5,215 | 5,189 |
| Testing images sharing a linked group with Training | 620 | 630 |
| Training images removed | 1,198 | 1,223 |
| Train / validation images | 3,739 / 663 | 3,709 / 668 |
| One-per-group Testing representatives | 1,347 | 1,342 |

## Worked examples (real pairs from the candidate file)
| Pair | Hamming | Correlation | Strict | Linked v1 | Linked v2 | What it shows |
|---|---|---|---|---|---|---|
| `Tr-gl_1057.jpg` ↔ `Te-gl_311.jpg` | 0 | 1.0000 | yes | yes | yes | byte-different but visually identical copy across the folders: strict, so the Training image is removed |
| `Tr-gl_1066.jpg` ↔ `Te-gl_267.jpg` | 6 | 0.9632 | no | yes | yes | linked by the v1 rule at its Hamming limit |
| `Tr-no_942.jpg` ↔ `Te-no_151.jpg` | 8 | 0.9955 | no | no | yes | altered copy found in the review: missed by v1, linked by v2 |
| `Tr-gl_700.jpg` ↔ `Te-gl_342.jpg` | 10 | 0.9815 | no | no | yes | near-identical neighbouring slice at Hamming 10, linked only by v2 |
| `Tr-me_102.jpg` ↔ `Te-me_142.jpg` | 6 | 0.9419 | no | no | no | correlation just below 0.95: not linked |
| `Te-gl_68.jpg` ↔ `Te-me_96.jpg` | 0 | 0.9999 | yes | yes | yes | two **Testing** images with different labels: label noise in the official test set |

Reading the table: a pair is linked when **both** conditions of one clause hold. `Tr-me_102` ↔ `Te-me_142` has a low enough
Hamming distance for v1 but its correlation is below 0.95, so it stays unlinked.

## Building the split
1. Remove from the training pool every Training image whose group contains a Testing image.
2. Remove every Training-folder image in a group with more than one class label (in v2 all such images are already
   removed by step 1).
3. Assign whole groups to validation (about 15% per class, seed 42) and the rest to training, so near-duplicates never
   straddle train and validation.
4. Keep the official 1,600 Testing images untouched, so results stay comparable with other work
   that uses the version 2 Testing folder.

After v2, no Testing image has a training or validation partner with correlation ≥ 0.98 within Hamming 10.

## Sensitivity check: a looser rule (no retraining)
pHash can still miss near-copies. As a robustness check, a looser rule that also links pairs at Hamming 7–10 with
correlation ≥ 0.95 adds 644 pairs and links 84 further Testing
images to the v2 training or validation images (58 pituitary, 11 glioma, 10 no tumour, 5 meningioma).
The leakage-aware model classifies all of them correctly; without them its accuracy is
91.4% (n = 1,516) against 91.9% on all
1,600 images. Stored in `results/metrics.json` → `sensitivity_looser_rule`.

## Subsets used in evaluation
- **Has strict dup. in Training**: Testing images with a strict near-duplicate in the *original* Training folder
  (n = 441).
- **No linked dup.**: Testing images with no linked image in the original Training folder (n = 970).
- **One per group**: one representative per Testing group (n = 1342).
