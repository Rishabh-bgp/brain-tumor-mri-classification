"""Time the forward passes LIME would need in the browser demo (headless Chrome, ONNX Runtime Web, 1 WASM thread).

LIME with the evaluated setting needs 200 forward passes of the model (plus SLIC superpixels and a ridge fit, not timed
here). This script loads the demo page, creates an ONNX Runtime Web session exactly as docs/app.js does, and times 200
forward passes in batches of 4, 10 and 50. Result: results/xai/lime_browser_timing.json.

Usage (from v2/, with the demo served):  cd ../docs && python3 -m http.server 8765 &
                                         python scripts/time_lime_browser.py http://localhost:8765/
Needs Playwright and a Chrome/Chromium binary (CHROME env var, default /usr/bin/google-chrome)."""
import asyncio
import json
import os
import platform
import sys
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765/"
JS = """async () => {
  ort.env.wasm.numThreads = 1;
  const s = await ort.InferenceSession.create("model/brain_tumor_effnetb0.onnx", {executionProviders: ["wasm"]});
  const S = 224, out = {};
  for (const B of [4, 10, 50]) {
    const x = new Float32Array(B * S * S).fill(0.3);
    await s.run({image: new ort.Tensor("float32", x, [B, 1, S, S])});
    const t0 = performance.now(); let n = 0;
    while (n < 200) { await s.run({image: new ort.Tensor("float32", x, [B, 1, S, S])}); n += B; }
    out["batch" + B] = (performance.now() - t0) / 1000;
  }
  return out;
}"""


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=os.environ.get("CHROME", "/usr/bin/google-chrome"))
        pg = await b.new_page()
        await pg.goto(URL)
        await pg.wait_for_function("typeof ort !== 'undefined'")
        res = await pg.evaluate(JS)
        await b.close()
    res.update({"forward_passes": 200, "url": URL, "browser": "headless Chrome", "wasm_threads": 1,
                "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
                "note": "model forward passes only; SLIC superpixels and the ridge fit would add to this"})
    (ROOT / "results/xai/lime_browser_timing.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

asyncio.run(main())
