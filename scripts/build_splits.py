"""Audit duplicates and write the leakage-aware split files to results/splits and results/review."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btmri.data import download_dataset, seed_everything  # noqa: E402
from btmri.splits import candidate_pairs, hash_table, make_splits  # noqa: E402

seed_everything(42)
out = ROOT / "results" / "review"
out.mkdir(parents=True, exist_ok=True)
data_root = download_dataset()
df = hash_table(data_root)
pairs = candidate_pairs(df)
df = make_splits(df, pairs)
cols = ["relpath", "split", "cls", "label", "md5", "phash", "group", "group_strict", "subset",
        "leaks_into_test", "label_conflict", "test_has_train_dup", "test_has_train_dup_strict", "test_dedup_rep"]
df[cols].to_csv(out / "hashes_and_groups.csv", index=False)
pairs.drop(columns=["i", "j"]).query("linked or strict_dup").sort_values(["hamming", "corr"], ascending=[True, False]) \
    .to_csv(out / "duplicate_pairs.csv", index=False)
pairs.drop(columns=["i", "j"]).to_csv(out / "candidate_pairs_hamming_le10.csv.gz", index=False)
for s in ("train", "val", "test", "excluded"):
    df[df.subset == s][["relpath", "cls", "label"]].to_csv(out / f"{s}_files.csv", index=False)
te = df.split == "Testing"
summary = {
    "n_images": len(df), "n_unique_md5": int(df.md5.nunique()),
    "exact_duplicate_files": int(df.md5.duplicated(keep=False).sum()),
    "exact_cross_split_pairs": int((pairs.md5_equal & pairs.cross_split).sum()),
    "strict_pairs": int(pairs.strict_dup.sum()), "strict_cross_split_pairs": int((pairs.strict_dup & pairs.cross_split).sum()),
    "linked_pairs": int(pairs.linked.sum()), "linked_cross_split_pairs": int((pairs.linked & pairs.cross_split).sum()),
    "groups": int(df.group.nunique()), "groups_strict": int(df.group_strict.nunique()),
    "test_imgs_with_train_dup_strict": int(df.test_has_train_dup_strict.sum()),
    "test_imgs_with_train_dup_linked": int(df.test_has_train_dup.sum()),
    "train_imgs_removed_cross_split": int(df.leaks_into_test.sum()),
    "imgs_in_label_conflict_groups": int(df.label_conflict.sum()),
    "label_conflict_pairs": int((pairs.linked & pairs.label_conflict).sum()),
    "subset_counts": df.subset.value_counts().to_dict(),
    "subset_by_class": {s: df[df.subset == s].cls.value_counts().sort_index().to_dict() for s in ("train", "val", "test", "excluded")},
    "test_dedup_reps": int(df.test_dedup_rep.sum()),
    "test_strict_dup_by_class": df[te].groupby("cls").test_has_train_dup_strict.sum().to_dict(),
    "test_linked_dup_by_class": df[te].groupby("cls").test_has_train_dup.sum().to_dict(),
    "linked_pairs_v1_rule": int(pairs.linked_v1.sum()), "extra_pairs_v2_rule": int((pairs.linked & ~pairs.linked_v1).sum()),
    "extra_cross_split_pairs_v2_rule": int((pairs.linked & ~pairs.linked_v1 & pairs.cross_split).sum()),
    "rule": {"hamming_max": 6, "corr_min": 0.95, "or_hamming_max": 10, "or_corr_min": 0.98, "strict_hamming": 2, "strict_corr": 0.98},
}
summary = json.loads(json.dumps(summary, default=int))
(out / "dedup_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
