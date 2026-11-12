// NEURO//SCAN shared logic: preprocessing, calibration, maps and the templated rationale.
// Used by the browser app and by tests/test_logic.js (Node). Copyright (c) 2026 Rishabh Aryan, MIT License.
(function (root) {
  const CLASSES = ["glioma", "meningioma", "notumor", "pituitary"];
  const LABELS = { glioma: "Glioma", meningioma: "Meningioma", notumor: "No tumour", pituitary: "Pituitary" };

  function softmax(logits, T) {
    const z = logits.map(v => v / (T || 1));
    const m = Math.max(...z);
    const e = z.map(v => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
  }
  function normEntropy(p) {
    let h = 0;
    for (const v of p) if (v > 0) h -= v * Math.log(v);
    return h / Math.log(p.length);
  }
  // CAM for class k from cam tensor (4 x 7 x 7, row-major): ReLU then min-max -> Float32Array(49)
  function camFor(cam, k, h = 7, w = 7) {
    const out = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) out[i] = Math.max(0, cam[k * h * w + i]);
    return minmax(out);
  }
  function minmax(a) {
    let lo = Infinity, hi = -Infinity;
    for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const out = new Float32Array(a.length);
    const r = hi - lo;
    for (let i = 0; i < a.length; i++) out[i] = r > 0 ? (a[i] - lo) / r : 0;
    return out;
  }
  // bilinear upsample of an n x n map to size x size (align_corners = false, like PyTorch)
  function upsample(map, n, size) {
    const out = new Float32Array(size * size);
    const sc = n / size;
    for (let y = 0; y < size; y++) {
      let fy = (y + 0.5) * sc - 0.5; fy = Math.min(Math.max(fy, 0), n - 1);
      const y0 = Math.floor(fy), y1 = Math.min(y0 + 1, n - 1), wy = fy - y0;
      for (let x = 0; x < size; x++) {
        let fx = (x + 0.5) * sc - 0.5; fx = Math.min(Math.max(fx, 0), n - 1);
        const x0 = Math.floor(fx), x1 = Math.min(x0 + 1, n - 1), wx = fx - x0;
        out[y * size + x] = (1 - wy) * ((1 - wx) * map[y0 * n + x0] + wx * map[y0 * n + x1]) +
          wy * ((1 - wx) * map[y1 * n + x0] + wx * map[y1 * n + x1]);
      }
    }
    return out;
  }
  // foreground mask: 7x7 box blur of the [0,1] image > thr (mirrors btmri.xai.brain_mask)
  function foregroundMask(img, size = 224, thr = 0.08) {
    const I = new Float64Array((size + 1) * (size + 1));
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++)
      I[(y + 1) * (size + 1) + x + 1] = img[y * size + x] + I[y * (size + 1) + x + 1] + I[(y + 1) * (size + 1) + x] - I[y * (size + 1) + x];
    const m = new Uint8Array(size * size);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const y0 = Math.max(0, y - 3), y1 = Math.min(size, y + 4), x0 = Math.max(0, x - 3), x1 = Math.min(size, x + 4);
      const s = I[y1 * (size + 1) + x1] - I[y0 * (size + 1) + x1] - I[y1 * (size + 1) + x0] + I[y0 * (size + 1) + x0];
      m[y * size + x] = s / 49 > thr ? 1 : 0; // zero padding, like avg_pool2d with count_include_pad
    }
    return m;
  }
  function attention(sal, img, size = 224) {
    let tot = 0; for (const v of sal) tot += v;
    tot = tot || 1;
    const m = foregroundMask(img, size);
    let inside = 0; const q = { "upper-left": 0, "upper-right": 0, "lower-left": 0, "lower-right": 0 };
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const v = sal[y * size + x] / tot;
      if (m[y * size + x]) inside += v;
      q[(y < size / 2 ? "upper" : "lower") + "-" + (x < size / 2 ? "left" : "right")] += v;
    }
    const top = Object.keys(q).reduce((a, b) => (q[a] >= q[b] ? a : b));
    return { inside_foreground: inside, top_quadrant: top, top_quadrant_share: q[top] };
  }
  const pct = v => (v > 0.999 ? ">99.9%" : v < 0.001 ? "<0.1%" : (100 * v).toFixed(1) + "%");
  // Templated, factual rationale built only from model outputs and saved test statistics.
  function rationale(probs, stats, att, agreement) {
    const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    const [p1, k1] = order[0], [p2, k2] = order[1];
    const c1 = CLASSES[k1], c2 = CLASSES[k2], st = stats.per_class[c1];
    const H = normEntropy(probs), uncertain = p1 < stats.conf_threshold;
    const lines = [];
    lines.push({ k: "Prediction", v: `${LABELS[c1]}, calibrated confidence ${pct(p1)} (temperature T = ${stats.temperature.toFixed(2)}, fitted on the validation split).` });
    lines.push({ k: "Margin", v: `${(100 * (p1 - p2)).toFixed(1)} percentage points ahead of the runner-up, ${LABELS[c2]} (${pct(p2)}).` });
    lines.push({ k: "Certainty", flag: uncertain ? "warn" : "ok", v: uncertain
      ? `Uncertain: low confidence. ${pct(p1)} is below the ${pct(stats.conf_threshold)} threshold chosen on validation. Normalised entropy ${H.toFixed(2)}.`
      : `Above the ${pct(stats.conf_threshold)} confidence threshold chosen on validation. Normalised entropy ${H.toFixed(2)} (0 = certain, 1 = uniform).` });
    if (att) lines.push({ k: "Attention", v: `${pct(att.inside_foreground)} of the Grad-CAM mass lies on the bright head region of the slice; the strongest share (${pct(att.top_quadrant_share)}) is in the ${att.top_quadrant} quadrant of the image.` });
    if (agreement) lines.push({ k: "Method agreement", v: "Rank correlation with Grad-CAM: " + Object.entries(agreement).map(([m, r]) => `${m} ${r.toFixed(2)}`).join(", ") + "." });
    lines.push({ k: "Track record", v: `On the ${stats.test_n.toLocaleString("en-IN")}-image test set, "${LABELS[c1]}" predictions had precision ${pct(st.precision)} and recall ${pct(st.recall)}. ` +
      (st.missed_as_n > 0 ? `True "${LABELS[c1]}" images that were missed were most often called ${LABELS[st.missed_as]} (${st.missed_as_n} of ${st.support}). ` : `No true "${LABELS[c1]}" images were missed. `) +
      (st.false_alarm_from_n > 0 ? `Wrong "${LABELS[c1]}" calls most often came from true ${LABELS[st.false_alarm_from]} images (${st.false_alarm_from_n}).` : `There were no wrong "${LABELS[c1]}" calls.`) });
    return { cls: c1, conf: p1, runnerUp: c2, margin: p1 - p2, entropy: H, uncertain, lines };
  }
  const api = { CLASSES, LABELS, softmax, normEntropy, camFor, minmax, upsample, foregroundMask, attention, rationale };
  if (typeof module !== "undefined") module.exports = api; else root.NeuroLogic = api;
})(this);
