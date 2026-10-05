// ChipQC Guardian browser demo. The model runs on this device; no image leaves the page.
import * as ort from "./vendor/ort.webgpu.min.mjs";      // ONNX Runtime Web 1.30.0 (MIT), shipped with the page: no third-party server is contacted
import { FRAME_W, FRAME_H, GRID, GREY_W, GREY_H, toFramePlanes, toGrey, acquisitionDescriptors, acquisitionFlags, tileTensor, assemble, decide, overlayPixels } from "./core.js";

ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
const $ = (id) => document.getElementById(id);
const BADGE = { PASS: ["pass", "may be used without a person looking"], REVIEW: ["review", "a person decides"], REACQUIRE: ["reacquire", "not a usable observation"] };
const state = { meta: null, session: null, backend: "", target: "0.1", last: null, busy: false };

async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get("Content-Length")) || 0, reader = response.body.getReader(), chunks = [];
  let done = 0;
  for (;;) {
    const { value, done: end } = await reader.read();
    if (end) break;
    chunks.push(value); done += value.length; onProgress(done, total);
  }
  const out = new Uint8Array(done);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

async function loadModel() {
  const want = new URLSearchParams(location.search).get("backend");
  const bytes = await fetchWithProgress("model/guardian_tiles.onnx", (d, t) => { $("status").textContent = `Downloading the model: ${(d / 1e6).toFixed(0)}${t ? " of " + (t / 1e6).toFixed(0) : ""} MB (once; your browser caches it)`; });
  const providers = want ? [want] : navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
  for (const p of providers) {
    try {
      $("status").textContent = `Preparing the model (${p === "webgpu" ? "WebGPU" : "WebAssembly"})…`;
      state.session = await ort.InferenceSession.create(bytes, { executionProviders: [p], graphOptimizationLevel: "all" });
      state.backend = p;
      break;
    } catch (e) { console.warn(p, "unavailable:", e); }
  }
  if (!state.session) throw new Error("Neither WebGPU nor WebAssembly could run the model in this browser.");
  $("status").textContent = state.backend === "webgpu" ? "Model ready (running on your GPU through WebGPU)." : "Model ready (running on your CPU through WebAssembly: about ten seconds per frame).";
  $("status").classList.add("ready");
}

async function decodeRGBA(blob) {
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height), ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  return { rgba: ctx.getImageData(0, 0, bitmap.width, bitmap.height).data, w: bitmap.width, h: bitmap.height };
}

async function assess(blob, onTile) {
  const { rgba, w, h } = await decodeRGBA(blob);
  const planes = toFramePlanes(rgba, w, h);
  const acquisition = acquisitionDescriptors(toGrey(planes));
  const flags = acquisitionFlags(acquisition, state.meta.acquisition_limits);
  const t0 = performance.now(), tiles = [];
  for (let t = 0; t < GRID * GRID; t++) {
    onTile(t);
    await new Promise((r) => setTimeout(r));                      // let the page repaint between tiles
    const input = new ort.Tensor("float32", tileTensor(planes, Math.floor(t / GRID), t % GRID), [1, 3, 336, 448]);
    const out = await state.session.run({ tiles: input });
    tiles.push(Float32Array.from(await out.evidence.getData()));
  }
  const { map, pGood } = assemble(tiles);
  return { planes, acquisition, flags, map, pGood, seconds: (performance.now() - t0) / 1000 };
}

function draw(result) {
  const W = 960, H = 720, grey = new Uint8ClampedArray(W * H), frame = new Uint8ClampedArray(W * H * 4);
  const [r, g, b] = result.planes;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {              // nearest sample of the 1344 x 1008 frame, for display only
    const i = Math.floor((y + 0.5) * FRAME_H / H) * FRAME_W + Math.floor((x + 0.5) * FRAME_W / W), o = y * W + x;
    grey[o] = (r[i] * 19595 + g[i] * 38470 + b[i] * 7471 + 0x8000) >> 16;
    frame[o * 4] = r[i]; frame[o * 4 + 1] = g[i]; frame[o * 4 + 2] = b[i]; frame[o * 4 + 3] = 255;
  }
  $("frame").getContext("2d").putImageData(new ImageData(frame, W, H), 0, 0);
  $("evidence").getContext("2d").putImageData(new ImageData(overlayPixels(grey, result.map, W, H), W, H), 0, 0);
}

function render() {
  const res = state.last;
  if (!res) return;
  const target = Number(state.target), tPass = state.meta.thresholds[state.target].t_pass;
  const { decision, reasons } = decide(res.pGood, res.flags, tPass, target);
  const [cls, meaning] = BADGE[decision];
  $("decision").innerHTML = `<span class="badge ${cls}">${decision}</span> <span class="meaning">${meaning}</span> <span class="p">P(good) <b>${res.pGood.toFixed(2)}</b></span>`;
  $("reasons").textContent = reasons.join(" · ");
  const L = state.meta.acquisition_limits, a = res.acquisition;
  const rows = [["Blacked-out share of the field", a.occluded_block_frac, `re-acquire above ${L.occluded_block_frac_max.toFixed(2)}`, a.occluded_block_frac > L.occluded_block_frac_max],
    ["Motion streak (anisotropy)", a.streak_anisotropy, `re-acquire above ${L.streak_anisotropy_max.toFixed(2)}`, a.streak_anisotropy > L.streak_anisotropy_max],
    ["Local sharpness (log10)", a.log_sharpness, `re-acquire below ${L.log_sharpness_min.toFixed(2)}`, a.log_sharpness < L.log_sharpness_min],
    ["Median brightness", a.median_brightness, `re-acquire below ${L.median_brightness_min.toFixed(2)}`, a.median_brightness < L.median_brightness_min],
    ["Clipped highlights", a.saturated_frac, "reported only", false]];
  $("descriptors").innerHTML = rows.map(([n, v, lim, hit]) => `<tr class="${hit ? "hit" : ""}"><td>${n}</td><td class="num">${v.toFixed(3)}</td><td>${lim}</td></tr>`).join("");
  let note = `Computed in this browser in ${res.seconds.toFixed(1)} s (${state.backend === "webgpu" ? "WebGPU" : "WebAssembly"}).`;
  if (res.example) {
    const ref = res.example.reference;
    note += ` Python reference for this frame: P(good) ${ref.p_good.toFixed(3)}, ${ref.decision} at a 10% target; difference in P(good) ${Math.abs(ref.p_good - res.pGood).toFixed(4)}.`;
    $("truth").textContent = `Experts labelled this frame ${res.example.expert_label}. Cell line ${res.example.cell_line}, ${res.example.culture_day_bin} days after seeding. The model in this page was fitted without this frame's acquisition date.`;
  } else {
    $("truth").textContent = "Your own frame: the result is exploratory. The model was fitted on one laboratory's chips and cameras (see the limits below).";
  }
  $("parity").textContent = note;
  const s = state.meta.evaluation.system?.[state.target]?.summary;
  if (s) $("measured").textContent = `Measured on dates the model never saw, at this target: ${(s.passed_automatically * 100).toFixed(0)}% of frames passed automatically and ${(s.bad_among_passed * 100).toFixed(1)}% of those were bad; ${(s.sent_to_reacquire * 100).toFixed(0)}% were sent back for re-acquisition.`;
  $("result").hidden = false;
}

async function run(blob, example) {
  if (state.busy || !state.session) return;
  state.busy = true; document.body.classList.add("busy");
  try {
    const res = await assess(blob, (t) => { $("progress").textContent = `Reading tile ${t + 1} of 9…`; });
    res.example = example || null;
    state.last = res;
    draw(res); render();
    $("progress").textContent = "";
    return res;
  } catch (e) {
    $("progress").textContent = `Could not score this image: ${e.message}`;
  } finally { state.busy = false; document.body.classList.remove("busy"); }
}

async function main() {
  state.meta = await (await fetch("model/web_model.json")).json();
  const ev = state.meta.evaluation;
  if (ev.auroc) $("headline").textContent = `AUROC ${ev.auroc.value.toFixed(3)} (95% interval ${ev.auroc.ci95[0].toFixed(3)}–${ev.auroc.ci95[1].toFixed(3)}) on ${ev.frames.toLocaleString()} frames from ${ev.dates} acquisition dates, each scored by a model that never saw its date.`;
  $("gallery").innerHTML = state.meta.frames.map((f, i) => `<button class="thumb" data-i="${i}" title="${f.image_id}"><img src="examples/${f.thumb}" alt="frame ${f.image_id}" loading="lazy"><span>${f.cell_line} · experts: ${f.expert_label}</span></button>`).join("");
  $("gallery").addEventListener("click", async (e) => {
    const b = e.target.closest(".thumb");
    if (!b || state.busy) return;
    document.querySelectorAll(".thumb").forEach((t) => t.classList.toggle("on", t === b));
    const f = state.meta.frames[Number(b.dataset.i)];
    await run(await (await fetch(`examples/${f.file}`)).blob(), f);
  });
  $("upload").addEventListener("change", async (e) => { if (e.target.files[0]) { document.querySelectorAll(".thumb").forEach((t) => t.classList.remove("on")); await run(e.target.files[0], null); } });
  document.querySelectorAll('input[name="target"]').forEach((r) => r.addEventListener("change", () => { state.target = r.value; render(); }));
  try { await loadModel(); } catch (e) { $("status").textContent = e.message; return; }
  if (new URLSearchParams(location.search).has("selftest")) {            // used by the automated parity check
    const out = [];
    for (const f of state.meta.frames) {
      const res = await run(await (await fetch(`examples/${f.file}`)).blob(), f);
      out.push({ image_id: f.image_id, p_good: res.pGood, flags: res.flags, acquisition: res.acquisition, seconds: res.seconds, reference: f.reference });
    }
    window.__selftest = { backend: state.backend, frames: out };
  } else {
    document.querySelector(".thumb")?.click();
  }
}

main();
