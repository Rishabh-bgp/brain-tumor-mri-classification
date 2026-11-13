# FAQ and troubleshooting

[Documentation home](README.md) · [Top-level README](../README.md) · [v2 README](../v2/README.md)

## Questions
**Why is the accuracy lower than many published results, which are often above 94%?**
Because near-copies of Testing images were removed from training. The identical model trained on the original folders
reaches 96.1% here, and it is right on 105/105
Testing meningiomas with a strict near-duplicate in Training against 55/105
for the leakage-aware model. See [dedup and splits](dedup_and_splits.md).

**Why is v2 less accurate than v1, and why is it still preferred?**
v2 is preferred because its split is cleaner. The accuracy change mainly reflects training variance of a single seed; see
[v1 vs v2](changelog.md#why-v2-is-preferred).

**Can I use this for diagnosis?** No. It is a research and education project ([model card](model_card.md)).

**Are uploaded images sent to a server?** No. The demo runs entirely in your browser with ONNX Runtime Web.

**Why does the demo say "uncertain"?** The calibrated top probability is below τ = 0.69. On Testing, such
predictions are right only 39.3% of the time.

**Is LIME included?** Yes, since v2.1.1, alongside KernelSHAP and on the same superpixels
([results](explainability.md#lime-added-in-v211)). In the demo it is shown for the eight presets only (precomputed): with the
evaluated setting its 200 forward passes alone took 10.9 s in headless Chrome, above the ~10 s budget.

**Where does each number come from?** From `v2/results/`; the README, paper macros, demo statistics and this documentation are
generated from those files ([results catalogue](results_catalogue.md)).

**How do I run the v1 code?** `git checkout v1.0` and follow the README there. The `v1/` folder keeps v1's results, notebooks and
paper for comparison.

## Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: btmri` | Run scripts from `v2/`; for training use `cd v2/src && python -m btmri.train`. In a notebook, run the first set-up cell. |
| Dataset download fails | `kagglehub` needs internet access; the dataset is public, so no API key is needed. |
| Training is very slow | Expect about 2 minutes per epoch on 8 CPU cores. Limit threads with `OMP_NUM_THREADS`/`torch.set_num_threads` if other jobs share the machine, or use a GPU runtime in Colab. |
| `run_xai.py` is slow or the machine is overloaded | Pass a thread count, e.g. `python scripts/run_xai.py 4`. KernelSHAP is the slowest step. |
| Numbers differ slightly after retraining | Expected on a different machine: CPU kernels are not bit-for-bit deterministic across hardware ([model and training](model_and_training.md#determinism)). |
| Demo stays on "loading model" | Serve `docs/` over HTTP (`python3 -m http.server`), not `file://`; check that jsDelivr is reachable; the model is 16 MB, so the first load takes a moment. |
| Occlusion takes long | It runs 169 forward passes in the browser (about 9–12 s in our tests); the progress counter updates while it runs. |
| Colab badge opens a notebook that cannot find files | The set-up cell must run first: it clones the repository and changes into `v2/`. |
| LaTeX build of the paper fails | Use a full TeX distribution or `tectonic` (`tectonic brain_tumor_mri_white_paper.tex` in `v2/paper/`). |
