# Future work

[Documentation home](README.md) · [Top-level README](../README.md) · [v2 README](../v2/README.md)

Concrete next steps, roughly in order of how much they would change the conclusions. Each one addresses an item in
[limitations](limitations.md).

1. **Several seeds.** Train the leakage-aware and naive models with at least five seeds each and report the mean and spread
   next to the bootstrap intervals. This would show how much of the v1/v2 difference and of the naive gap is training variance.
2. **External test set.** Evaluate the unchanged model on an independent collection, for example 2-D slices extracted from
   BraTS (Menze et al., 2015) for gliomas, to measure performance outside this dataset.
3. **Patient-level data.** Repeat the study on data with patient identifiers (the original figshare release by Cheng et al.,
   2015, has patient IDs for its three tumour classes) and compare image-level with patient-level splits.
4. **Stronger duplicate detection.** Add learned image embeddings (e.g. self-supervised features) next to pHash, check the
   `-aug-` files against them, and publish a cleaned file list.
5. **Relabel the conflicting groups.** Have the 13 images in label-conflict groups reviewed
   and report results with and without them.
6. **Address source confounding.** Test whether the model can tell the source collections apart from background or intensity
   cues alone, and try intensity standardisation or skull stripping.
7. **Better uncertainty.** Compare temperature scaling with ensembles or conformal prediction sets, and add an
   out-of-distribution check so that non-MRI inputs are rejected.
8. **Localisation ground truth for XAI.** Use the tumour masks in the figshare release to measure how much of each heatmap
   falls on the tumour, extend the sanity check to more images, and run the data (label) randomisation test, which needs a
   model retrained on shuffled labels.
9. **Larger or more modern backbones with a GPU.** Compare with ConvNeXt or ViT models and a hyper-parameter search, keeping the
   same leakage-aware split so the comparison is fair.
10. **Demo.** Show precomputed maps for uploads where feasible (e.g. Grad-CAM++ in the browser) and add an input check that
    warns when an image does not look like a brain MRI slice.
