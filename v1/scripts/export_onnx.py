"""Export the leakage-aware model to ONNX (logits + CAM maps) and verify it against PyTorch."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btmri.model import ExportWrapper, build_model  # noqa: E402
from btmri.xai import GreyModel, gradcam_family  # noqa: E402

model = build_model(pretrained=False)
model.load_state_dict(torch.load(ROOT / "results/runs/clean/best.pt", map_location="cpu"))
model.eval()
wrap = ExportWrapper(model).eval()
(ROOT / "models").mkdir(exist_ok=True)
onnx_path = ROOT / "models/brain_tumor_effnetb0.onnx"
torch.onnx.export(wrap, torch.zeros(1, 1, 224, 224), str(onnx_path), input_names=["image"], output_names=["logits", "cam"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}, "cam": {0: "batch"}}, opset_version=17,
                  dynamo=False)
shutil.copy(ROOT / "results/runs/clean/best.pt", ROOT / "models/brain_tumor_effnetb0.pt")
(ROOT / "docs/model").mkdir(parents=True, exist_ok=True)
shutil.copy(onnx_path, ROOT / "docs/model/brain_tumor_effnetb0.onnx")

# verification on 64 test images
groups = pd.read_csv(ROOT / "results/review/hashes_and_groups.csv")
X = np.load(ROOT.parent / "cache/all_224.npy")
idx = np.nonzero((groups.subset == "test").values)[0][::25][:64]
x = torch.from_numpy(X[idx]).float().div(255).unsqueeze(1)
with torch.no_grad():
    tl, tc = wrap(x)
sess = ort.InferenceSession(str(onnx_path))
ol, oc = sess.run(None, {"image": x.numpy()})
gm = GreyModel(model).eval()
cam_corr = []
for i in range(8):
    k = int(tl[i].argmax())
    g = gradcam_family(gm, x[i:i + 1], k)
    c = torch.relu(torch.tensor(oc[i, k]))
    c = torch.nn.functional.interpolate(c[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
    c = (c - c.min()) / max(c.max() - c.min(), 1e-9)
    cam_corr.append(float(np.abs(c - g).max()))
rep = {"max_abs_logit_diff_torch_vs_onnx": float(np.abs(tl.numpy() - ol).max()),
       "argmax_agreement": float((tl.numpy().argmax(1) == ol.argmax(1)).mean()),
       "max_abs_cam_diff_torch_vs_onnx": float(np.abs(tc.numpy() - oc).max()),
       "max_abs_diff_onnx_cam_vs_autograd_gradcam_normalised": max(cam_corr),
       "onnx_size_mb": onnx_path.stat().st_size / 1e6}
(ROOT / "results/onnx_verification.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
