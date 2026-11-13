# Web demo (NEURO//SCAN) and ONNX export

[Documentation home](README.md) · [Top-level README](../README.md) · [v2 README](../v2/README.md)

Live: **https://rishabh-bgp.github.io/brain-tumor-mri-classification/** · source: [`docs/`](../docs/) (served by GitHub Pages from `main:/docs`) · notebook
[`05_onnx_export_and_web_demo.ipynb`](../v2/notebooks/05_onnx_export_and_web_demo.ipynb).

> ⚠️ Research and education project, **not a medical device**. Do not use it for diagnosis or treatment decisions.

The demo serves the **v2** leakage-aware model. The frozen v1 model is not deployed; v1 code and results are in
[`v1/`](../v1/README.md) and at the tag [`v1.0`](https://github.com/Rishabh-bgp/brain-tumor-mri-classification/tree/v1.0).

## Architecture
Everything runs in the browser; uploaded images never leave the device. There is no server-side code.

| File | Role |
|---|---|
| [`docs/index.html`](../docs/index.html) | Page layout, About section, the v1 vs v2 note; loads ONNX Runtime Web 1.20.1 (WebAssembly) from jsDelivr |
| [`docs/app.js`](../docs/app.js) | Image loading, pre-processing on a canvas, inference, heatmap drawing, browser occlusion, presets |
| [`docs/logic.js`](../docs/logic.js) | Shared pure functions (softmax with temperature, normalised entropy, CAM extraction, up-sampling, foreground mask, the templated rationale); also tested in Node by [`tests/test_logic.js`](../tests/test_logic.js) |
| [`docs/model/brain_tumor_effnetb0.onnx`](../docs/model/brain_tumor_effnetb0.onnx) | The exported model (16.0 MB) |
| [`docs/stats.json`](../docs/stats.json) | Temperature, threshold τ and measured test statistics used by the "why" panel (written by `make_report.py`) |
| [`docs/presets.json`](../docs/presets.json) + [`docs/samples/`](../docs/samples/) | Eight Testing images (unmodified originals) with saved PyTorch predictions and precomputed Grad-CAM++, IG, KernelSHAP and LIME maps (56×56) |
| [`docs/style.css`](../docs/style.css) | Styling |

**Inference path.** File → greyscale → zero-pad to square → bilinear 224×224 (canvas) → float32 in [0, 1] →
`InferenceSession.run` → `logits` (1×4) and `cam` (1×4×7×7) → softmax(logits / T) with T = 1.909 →
"uncertain" if the top probability is below τ = 0.69. ImageNet normalisation is inside the ONNX graph.

**Heatmaps.**
- *Grad-CAM*: ReLU of the predicted class's CAM, min-max normalised and up-sampled; exact, because for this network
  Grad-CAM equals CAM up to a positive scale.
- *Occlusion*: a 32-pixel black square at stride 16 (13 × 13 = 169 forward passes) in batches of 4; the page yields to the
  browser after each batch so it stays responsive (about 9–12 seconds in our headless-Chrome tests).
- *Grad-CAM++, Integrated Gradients, KernelSHAP, LIME*: precomputed in Python for the presets only.
- *Why LIME is not computed for uploads*: with the evaluated setting LIME needs 200 forward passes, plus SLIC superpixels
  (not available in the page) and a ridge fit. The forward passes alone took 11.0 s, 10.9 s
  and 11.8 s (batches of 4, 10 and 50) in headless Chrome with one WebAssembly thread
  ([`time_lime_browser.py`](../v2/scripts/time_lime_browser.py), [`lime_browser_timing.json`](../v2/results/xai/lime_browser_timing.json)),
  above the ~10 s budget for the page. Fewer samples would be faster but would no longer be the evaluated method.

**"Why this prediction" panel.** Template text that uses only the model's outputs (probabilities, entropy, where the map
mass lies) and the measured statistics in `stats.json` (per-class recall and precision, uncertain-rate accuracy). It does not
describe anatomy.

## ONNX export and verification
[`v2/scripts/export_onnx.py`](../v2/scripts/export_onnx.py) wraps the model in `ExportWrapper` (input `image`, 1×1×224×224
in [0, 1]; outputs `logits` and `cam`; dynamic batch; opset 17), writes `v2/models/brain_tumor_effnetb0.onnx`, copies it
to `docs/model/`, and compares ONNX Runtime with PyTorch on 64 Testing images:

| Check | Result |
|---|---|
| Max absolute logit difference | 0.00024 |
| Arg-max agreement | 100.0% |
| Max absolute CAM difference | 0.0014 |
| ONNX CAM vs autograd Grad-CAM (normalised, 8 images) | 1.6e-05 |

## Presets
Chosen by [`v2/scripts/make_presets.py`](../v2/scripts/make_presets.py): two per class, drawn with a fixed seed from the
75 Testing images that have **no candidate pair at all at Hamming ≤ 10**
(whatever the correlation) and are not `-aug-` files. Files are copied byte-for-byte.

| File | True class | Saved prediction (calibrated) |
|---|---|---|
| `Te-gl_391.jpg` | Glioma | No tumour (59.2%) |
| `Te-gl_73.jpg` | Glioma | Glioma (100.0%) |
| `Te-me_248.jpg` | Meningioma | Meningioma (96.4%) |
| `Te-me_215.jpg` | Meningioma | Meningioma (100.0%) |
| `Te-no_65.jpg` | No tumour | No tumour (100.0%) |
| `Te-no_9.jpg` | No tumour | No tumour (100.0%) |
| `Te-pi_80.jpg` | Pituitary | Pituitary (96.4%) |
| `Te-pi_96.jpg` | Pituitary | Pituitary (100.0%) |

`Te-gl_391` is kept on purpose: it is the one misclassified preset and is marked with a "misclassified" badge.
[`v2/scripts/check_samples.py`](../v2/scripts/check_samples.py) and [`tests/check_samples_browser.py`](../tests/check_samples_browser.py)
check that the presets give the saved predictions (within 0.01, same uncertain flag) in ONNX Runtime and in the browser.

## Run the demo locally
```bash
cd docs
python3 -m http.server 8765
# open http://localhost:8765/
```
Opening `index.html` directly from disk does not work, because browsers block `fetch()` of the model and JSON files from
`file://` URLs. The page needs internet access for the ONNX Runtime script from jsDelivr.
