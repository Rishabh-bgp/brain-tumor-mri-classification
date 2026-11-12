// NEURO//SCAN browser app. Copyright (c) 2026 Rishabh Aryan, MIT License.
(() => {
  const L = window.NeuroLogic;
  const S = 224, VIEW = 448;
  const $ = id => document.getElementById(id);
  const METHODS = [
    { id: "gradcam", name: "Grad-CAM", label: "Grad-CAM — where the model looked", live: true },
    { id: "occlusion", name: "Occlusion", label: "Occlusion — where hiding a 32-px patch (stride 16) lowered the class score (logit) most", live: true },
    { id: "gradcampp", name: "Grad-CAM++", label: "Grad-CAM++ — where the model looked (precomputed, sample slices only)" },
    { id: "integrated_gradients", name: "Integrated Gradients", label: "Integrated Gradients — pixel-level attribution (precomputed, sample slices only)" },
    { id: "kernel_shap", name: "KernelSHAP", label: "KernelSHAP — superpixel contributions (precomputed, sample slices only)" },
  ];
  const AG_NAMES = { gradcampp: "Grad-CAM++", integrated_gradients: "Integrated Gradients", occlusion: "Occlusion", kernel_shap: "KernelSHAP" };
  let session, stats, presets = [], cur = null, method = "gradcam", busy = false;

  // ---------- preprocessing: greyscale (ITU-R 601-2 luma, like Pillow), zero-pad to square, bilinear resize ----------
  function toGraySquare(img) {
    const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height, s = Math.max(w, h);
    const c = document.createElement("canvas"); c.width = s; c.height = s;
    const ctx = c.getContext("2d", { willReadFrequently: true }); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, s, s);
    ctx.drawImage(img, Math.floor((s - w) / 2), Math.floor((s - h) / 2));
    const d = ctx.getImageData(0, 0, s, s).data, g = new Float32Array(s * s);
    for (let i = 0; i < s * s; i++) g[i] = (d[4 * i] * 19595 + d[4 * i + 1] * 38470 + d[4 * i + 2] * 7471 + 0x8000) >> 16;
    return resizePIL(g, s, S);
  }
  // Pillow-style separable bilinear (triangle) resampling with antialiasing support = scale
  function coeffs(inSize, outSize) {
    const scale = inSize / outSize, fs = Math.max(scale, 1), sup = fs, out = [];
    for (let i = 0; i < outSize; i++) {
      const c = (i + 0.5) * scale;
      const x0 = Math.max(Math.trunc(c - sup + 0.5), 0), x1 = Math.min(Math.trunc(c + sup + 0.5), inSize);
      const w = []; let t = 0;
      for (let x = x0; x < x1; x++) { const v = Math.max(0, 1 - Math.abs((x - c + 0.5) / fs)); w.push(v); t += v; }
      out.push({ x0, w: w.map(v => (t ? v / t : 0)) });
    }
    return out;
  }
  function resizePIL(src, n, m) {
    const cx = coeffs(n, m), tmp = new Float32Array(n * m), dst = new Float32Array(m * m);
    for (let y = 0; y < n; y++) for (let i = 0; i < m; i++) {
      let v = 0; const { x0, w } = cx[i];
      for (let k = 0; k < w.length; k++) v += src[y * n + x0 + k] * w[k];
      tmp[y * m + i] = Math.min(255, Math.max(0, Math.round(v)));
    }
    for (let j = 0; j < m; j++) { const { x0, w } = cx[j];
      for (let i = 0; i < m; i++) { let v = 0;
        for (let k = 0; k < w.length; k++) v += tmp[(x0 + k) * m + i] * w[k];
        dst[j * m + i] = Math.min(255, Math.max(0, Math.round(v))) / 255; } }
    return dst;
  }

  // ---------- rendering ----------
  function drawSlice(g) {
    const c = document.createElement("canvas"); c.width = S; c.height = S;
    const id = c.getContext("2d").createImageData(S, S);
    for (let i = 0; i < S * S; i++) { const v = g[i] * 255; id.data.set([v, v, v, 255], 4 * i); }
    c.getContext("2d").putImageData(id, 0, 0);
    const ctx = $("slice").getContext("2d"); ctx.imageSmoothingEnabled = true; ctx.clearRect(0, 0, VIEW, VIEW);
    ctx.drawImage(c, 0, 0, VIEW, VIEW);
  }
  const STOPS = [[0, [11, 29, 58]], [0.25, [14, 116, 144]], [0.45, [45, 212, 191]], [0.65, [250, 204, 21]], [0.82, [249, 115, 22]], [1, [239, 68, 68]]];
  function cmap(v) {
    for (let i = 1; i < STOPS.length; i++) if (v <= STOPS[i][0]) {
      const [a, ca] = STOPS[i - 1], [b, cb] = STOPS[i], t = (v - a) / (b - a);
      return ca.map((x, k) => x + t * (cb[k] - x));
    }
    return STOPS[STOPS.length - 1][1];
  }
  function drawHeat(map224) {
    const c = document.createElement("canvas"); c.width = S; c.height = S;
    const id = c.getContext("2d").createImageData(S, S);
    for (let i = 0; i < S * S; i++) { const v = map224[i], [r, g, b] = cmap(v); id.data.set([r, g, b, Math.round(255 * Math.min(1, v * 1.4))], 4 * i); }
    c.getContext("2d").putImageData(id, 0, 0);
    const ctx = $("heat").getContext("2d"); ctx.clearRect(0, 0, VIEW, VIEW); ctx.drawImage(c, 0, 0, VIEW, VIEW);
  }

  // ---------- inference ----------
  async function run(gray, n = 1) {
    const t = new ort.Tensor("float32", gray, [n, 1, S, S]);
    const out = await session.run({ image: t });
    return { logits: out.logits.data, cam: out.cam.data };
  }
  // Occlusion on the class logit, same setting as Captum's Occlusion in btmri.xai: 32-px black window, stride 16
  // (13 x 13 = 169 forward passes); each pixel gets the mean logit drop over the windows that cover it.
  async function occlusionMap(gray, k, base) {
    const W = 32, ST = 16, G = (S - W) / ST + 1, B = 13, n2 = G * G;
    const sum = new Float32Array(S * S), cnt = new Float32Array(S * S);
    for (let s = 0; s < n2; s += B) {
      const n = Math.min(B, n2 - s), batch = new Float32Array(n * S * S);
      for (let b = 0; b < n; b++) {
        batch.set(gray, b * S * S);
        const gy = Math.floor((s + b) / G) * ST, gx = ((s + b) % G) * ST;
        for (let y = gy; y < gy + W; y++) batch.fill(0, b * S * S + y * S + gx, b * S * S + y * S + gx + W);
      }
      const r = await run(batch, n);
      for (let b = 0; b < n; b++) {
        const d = base - r.logits[4 * b + k];
        const gy = Math.floor((s + b) / G) * ST, gx = ((s + b) % G) * ST;
        for (let y = gy; y < gy + W; y++) for (let x = gx; x < gx + W; x++) { sum[y * S + x] += d; cnt[y * S + x] += 1; }
      }
      $("status").textContent = `Occlusion ${Math.min(s + n, n2)}/${n2}…`;
    }
    const out = new Float32Array(S * S);
    for (let i = 0; i < S * S; i++) out[i] = cnt[i] ? Math.max(0, sum[i] / cnt[i]) : 0;
    return L.minmax(out);
  }
  function decode56(b64) {
    const bin = atob(b64), a = new Float32Array(bin.length);
    for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i) / 255;
    return L.upsample(a, 56, S);
  }

  async function analyse(img, name, preset) {
    if (!session) { $("status").textContent = "Model still loading — please wait…"; return; }
    if (busy) return; busy = true;
    $("placeholder").hidden = true; $("hud-file").textContent = name;
    document.querySelectorAll(".sample").forEach(el => el.classList.toggle("active", preset && el.dataset.file === preset.file));
    const gray = toGraySquare(img);
    drawSlice(gray); $("heat").getContext("2d").clearRect(0, 0, VIEW, VIEW);
    $("scan").hidden = false; $("status").textContent = "Inferring…";
    const t0 = performance.now();
    const [r] = await Promise.all([run(gray), new Promise(res => setTimeout(res, 900))]);
    const probs = L.softmax(Array.from(r.logits), stats.temperature);
    const k = probs.indexOf(Math.max(...probs));
    const gc = L.upsample(L.camFor(r.cam, k), 7, S);
    cur = { gray, probs, k, logitK: r.logits[k], maps: { gradcam: gc }, preset, att: L.attention(gc, gray) };
    if (preset) for (const m of ["gradcampp", "integrated_gradients", "kernel_shap"]) cur.maps[m] = decode56(preset.maps56[m]);
    $("scan").hidden = true;
    $("status").textContent = `Done in ${Math.round(performance.now() - t0)} ms · on-device`;
    renderResult(); setMethodButtons();
    busy = false;
    await showMethod(method in cur.maps || method === "occlusion" ? method : "gradcam");
  }
  async function showMethod(m) {
    method = m; setMethodButtons();
    $("heat-label").textContent = METHODS.find(x => x.id === m).label;
    if (!cur) return;
    if (m === "occlusion" && !cur.maps.occlusion) {
      busy = true; setMethodButtons(); $("scan").hidden = false;
      cur.maps.occlusion = await occlusionMap(cur.gray, cur.k, cur.logitK);
      $("scan").hidden = true; $("status").textContent = "Occlusion done · on-device"; busy = false; setMethodButtons();
    }
    drawHeat(cur.maps[m]);
  }
  function setMethodButtons() {
    const box = $("methods"); box.innerHTML = "";
    for (const m of METHODS) {
      const b = document.createElement("button"); b.type = "button"; b.textContent = m.name; b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", String(m.id === method));
      b.disabled = busy || (!m.live && !(cur && cur.preset));
      b.title = b.disabled ? "Available for the sample slices (precomputed in Python)" : m.label;
      b.onclick = () => !busy && showMethod(m.id);
      box.appendChild(b);
    }
  }
  function renderResult() {
    const { probs, preset } = cur;
    const agreement = preset ? Object.fromEntries(Object.entries(preset.agreement_with_gradcam).map(([m, v]) => [AG_NAMES[m], v])) : null;
    const R = L.rationale(probs, stats, cur.att, agreement);
    $("verdict").className = "verdict";
    $("verdict").innerHTML = `${L.LABELS[R.cls]} <span class="badge ${R.uncertain ? "warn" : "ok"}">${R.uncertain ? "uncertain — low confidence" : "confident"}</span>` +
      (preset ? `<small>True label of this test image: ${L.LABELS[preset.true]}${preset.true === R.cls ? " ✓ match" : " ✗ mismatch"}</small>` : `<small>Your image · no ground truth available</small>`);
    const bars = $("bars"); bars.innerHTML = "";
    L.CLASSES.map((c, i) => [c, probs[i], i]).sort((a, b) => b[1] - a[1]).forEach(([c, p, i]) => {
      const d = document.createElement("div"); d.className = "bar" + (i === cur.k ? " top" : "");
      d.innerHTML = `<span>${L.LABELS[c]}</span><div class="track"><div class="fill" style="background:var(--c-${c})"></div></div><span class="val">${(100 * p).toFixed(1)}%</span>`;
      bars.appendChild(d); requestAnimationFrame(() => requestAnimationFrame(() => { d.querySelector(".fill").style.width = (100 * p).toFixed(2) + "%"; }));
    });
    $("why").innerHTML = R.lines.map(l => `<dt>${l.k}</dt><dd class="${l.flag === "warn" ? "warn" : ""}">${l.v}</dd>`).join("") +
      `<dt>Note</dt><dd>Generated from the model's outputs and test statistics only. Research demo — not for diagnosis.</dd>`;
  }

  // ---------- wiring ----------
  function loadFile(f) {
    if (!f) return;
    if (!/^image\/(jpeg|png)$/.test(f.type)) { $("status").textContent = "Please choose a JPG or PNG image"; return; }
    const img = new Image(); img.onload = () => analyse(img, f.name, null); img.src = URL.createObjectURL(f);
  }
  const drop = $("drop");
  drop.onclick = () => $("file").click();
  drop.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } };
  $("file").onchange = e => loadFile(e.target.files[0]);
  drop.ondragover = e => { e.preventDefault(); drop.classList.add("drag"); };
  drop.ondragleave = () => drop.classList.remove("drag");
  drop.ondrop = e => { e.preventDefault(); drop.classList.remove("drag"); loadFile(e.dataTransfer.files[0]); };
  $("opacity").oninput = e => { $("heat").style.opacity = e.target.value / 100; $("opv").textContent = e.target.value + "%"; };
  setMethodButtons();

  (async () => {
    try {
      [stats, presets] = await Promise.all([fetch("stats.json").then(r => r.json()), fetch("presets.json").then(r => r.json())]);
      const box = $("samples");
      for (const p of presets) {
        const b = document.createElement("button"); b.className = "sample"; b.dataset.file = p.file; b.type = "button";
        b.disabled = true; b.title = "Model still loading…";
        b.innerHTML = `<img src="${p.file}" alt="Test-split MRI slice, true class ${L.LABELS[p.true]}" loading="lazy"><span>True: <b>${L.LABELS[p.true]}</b></span>`;
        b.onclick = () => { const img = new Image(); img.onload = () => analyse(img, p.file.split("/").pop(), p); img.src = p.file; };
        box.appendChild(b);
      }
      ort.env.wasm.numThreads = 1;
      session = await ort.InferenceSession.create("model/brain_tumor_effnetb0.onnx", { executionProviders: ["wasm"] });
      $("status").textContent = "Model ready · ONNX Runtime Web (WASM)";
      document.querySelectorAll(".sample").forEach(el => { el.disabled = false; el.title = ""; });
    } catch (err) { $("status").textContent = "Failed to load: " + err.message; console.error(err); }
  })();
  window.__neuro = { analyse, toGraySquare, run, get cur() { return cur; } };
})();
