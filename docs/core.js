// ChipQC Guardian, browser port of the frame pipeline. Pure functions, no DOM: the same code runs in the page
// and in the Node parity test. Everything mirrors src/chipqc/{frames,descriptors,guardian}.py.

export const FRAME_W = 1344, FRAME_H = 1008, TILE_W = 448, TILE_H = 336, GRID = 3, PATCH = 14;
export const GREY_W = 1024, GREY_H = 768;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const BLOCK = 32, DARK_BLOCK = 0.12;

// ---------------------------------------------------------------- Pillow-compatible resampling (8-bit planes)
const FILTERS = {
  bilinear: { support: 1, f: (x) => { x = Math.abs(x); return x < 1 ? 1 - x : 0; } },
  bicubic: { support: 2, f: (x) => { const a = -0.5; x = Math.abs(x); return x < 1 ? ((a + 2) * x - (a + 3)) * x * x + 1 : x < 2 ? (((x - 5) * x + 8) * x - 4) * a : 0; } },
};
const PRECISION = 22, HALF = 1 << (PRECISION - 1);

function coefficients(inSize, outSize, filter) {
  const scale = inSize / outSize, fscale = Math.max(scale, 1), support = filter.support * fscale;
  const bounds = new Int32Array(outSize * 2), weights = [];
  for (let xx = 0; xx < outSize; xx++) {
    const centre = (xx + 0.5) * scale;
    const lo = Math.max(0, Math.trunc(centre - support + 0.5));
    const n = Math.min(inSize, Math.trunc(centre + support + 0.5)) - lo;
    const k = new Float64Array(n);
    let sum = 0;
    for (let x = 0; x < n; x++) { k[x] = filter.f((x + lo - centre + 0.5) / fscale); sum += k[x]; }
    const ki = new Int32Array(n);
    for (let x = 0; x < n; x++) { const v = (k[x] / sum) * (1 << PRECISION); ki[x] = Math.trunc(v + (v < 0 ? -0.5 : 0.5)); }
    bounds[xx * 2] = lo; bounds[xx * 2 + 1] = n; weights.push(ki);
  }
  return { bounds, weights };
}

const clip8 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);

/** Resize one 8-bit plane the way Pillow does: horizontal pass, round to 8 bits, vertical pass. */
export function resizePlane(src, sw, sh, dw, dh, kind = "bicubic") {
  if (sw === dw && sh === dh) return src;
  const filter = FILTERS[kind];
  let cur = src, cw = sw;
  if (sw !== dw) {
    const { bounds, weights } = coefficients(sw, dw, filter);
    const out = new Uint8Array(dw * sh);
    for (let y = 0; y < sh; y++) {
      const row = y * sw;
      for (let xx = 0; xx < dw; xx++) {
        const lo = bounds[xx * 2], n = bounds[xx * 2 + 1], k = weights[xx];
        let s = HALF;
        for (let x = 0; x < n; x++) s += cur[row + lo + x] * k[x];
        out[y * dw + xx] = clip8(s >> PRECISION);
      }
    }
    cur = out; cw = dw;
  }
  if (sh !== dh) {
    const { bounds, weights } = coefficients(sh, dh, filter);
    const out = new Uint8Array(cw * dh);
    for (let yy = 0; yy < dh; yy++) {
      const lo = bounds[yy * 2], n = bounds[yy * 2 + 1], k = weights[yy];
      for (let x = 0; x < cw; x++) {
        let s = HALF;
        for (let y = 0; y < n; y++) s += cur[(lo + y) * cw + x] * k[y];
        out[yy * cw + x] = clip8(s >> PRECISION);
      }
    }
    cur = out;
  }
  return cur;
}

/** RGBA pixels of any size -> three 8-bit planes at the model's frame size (portrait frames are rotated first). */
export function toFramePlanes(rgba, w, h) {
  let planes = [0, 1, 2].map((c) => { const p = new Uint8Array(w * h); for (let i = 0; i < w * h; i++) p[i] = rgba[i * 4 + c]; return p; });
  if (h > w) {                                             // rotate 90 degrees counter-clockwise, like PIL's rotate(90, expand=True)
    planes = planes.map((p) => { const r = new Uint8Array(w * h); for (let y = 0; y < w; y++) for (let x = 0; x < h; x++) r[y * h + x] = p[x * w + (w - 1 - y)]; return r; });
    [w, h] = [h, w];
  }
  return planes.map((p) => resizePlane(p, w, h, FRAME_W, FRAME_H, "bicubic"));
}

/** Frame planes -> grey float frame in [0, 1] at 1024 x 768 (ITU-R 601 luma, then bilinear), as frames.to_grey. */
export function toGrey(planes) {
  const n = FRAME_W * FRAME_H, l = new Uint8Array(n), [r, g, b] = planes;
  for (let i = 0; i < n; i++) l[i] = (r[i] * 19595 + g[i] * 38470 + b[i] * 7471 + 0x8000) >> 16;
  const small = resizePlane(l, FRAME_W, FRAME_H, GREY_W, GREY_H, "bilinear");
  const out = new Float32Array(GREY_W * GREY_H);
  for (let i = 0; i < out.length; i++) out[i] = small[i] / 255;
  return out;
}

// ---------------------------------------------------------------- acquisition descriptors and gate
function median(values) {
  const a = Float64Array.from(values).sort();
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

export function acquisitionDescriptors(grey) {
  const W = GREY_W, H = GREY_H, rows = H / BLOCK, cols = W / BLOCK;
  const lit = new Uint8Array(rows * cols);
  let litCount = 0, saturated = 0;
  for (let br = 0; br < rows; br++) for (let bc = 0; bc < cols; bc++) {
    let s = 0;
    for (let y = 0; y < BLOCK; y++) { const o = (br * BLOCK + y) * W + bc * BLOCK; for (let x = 0; x < BLOCK; x++) s += grey[o + x]; }
    if (s / (BLOCK * BLOCK) > DARK_BLOCK) { lit[br * cols + bc] = 1; litCount++; }
  }
  const hist = new Uint32Array(256);
  for (let i = 0; i < W * H; i++) { if (grey[i] > 0.97) saturated++; hist[Math.round(grey[i] * 255)]++; }
  const half = (W * H) / 2;                                  // even count: the median is the mean of the two middle values
  let acc = 0, lowMid = -1, highMid = -1;
  for (let v = 0; v < 256; v++) { acc += hist[v]; if (lowMid < 0 && acc >= half) lowMid = v; if (acc >= half + 1) { highMid = v; break; } }
  let ex = 0, ey = 0;
  for (let y = 0; y < H - 1; y++) for (let x = 0; x < W - 1; x++) {
    const i = y * W + x, dx = grey[i + 1] - grey[i], dy = grey[i + W] - grey[i];
    ex += dx * dx; ey += dy * dy;
  }
  const cells = (H - 1) * (W - 1); ex /= cells; ey /= cells;
  const by = Math.floor((H - 2) / BLOCK), bx = Math.floor((W - 2) / BLOCK), vars = [];
  for (let br = 0; br < by; br++) for (let bc = 0; bc < bx; bc++) {
    if (!lit[br * cols + bc]) continue;
    let s = 0, s2 = 0;
    for (let y = 0; y < BLOCK; y++) for (let x = 0; x < BLOCK; x++) {
      const i = (br * BLOCK + y + 1) * W + (bc * BLOCK + x + 1);
      const v = 4 * grey[i] - grey[i - W] - grey[i + W] - grey[i - 1] - grey[i + 1];
      s += v; s2 += v * v;
    }
    const n = BLOCK * BLOCK, m = s / n;
    vars.push(s2 / n - m * m);
  }
  const sharp = vars.length ? median(vars) : 0;
  return {
    occluded_block_frac: 1 - litCount / (rows * cols),
    saturated_frac: saturated / (W * H),
    median_brightness: (lowMid + highMid) / 2 / 255,
    log_sharpness: Math.log10(sharp + 1e-9),
    streak_anisotropy: Math.abs(ex - ey) / (ex + ey + 1e-9),
  };
}

export function acquisitionFlags(d, limits) {
  const flags = [];
  if (d.occluded_block_frac > limits.occluded_block_frac_max) flags.push(`${Math.round(d.occluded_block_frac * 100)}% of the field is blacked out`);
  if (d.streak_anisotropy > limits.streak_anisotropy_max) flags.push("motion streak suspected (one-directional smear beyond the reference limit)");
  if (d.log_sharpness < limits.log_sharpness_min) flags.push("defocus suspected (local sharpness below the reference limit)");
  if (d.median_brightness < limits.median_brightness_min) flags.push("under-exposed (median brightness below the reference limit)");
  return flags;
}

// ---------------------------------------------------------------- model input, score and decision
/** Frame planes -> Float32Array for tile (row, col): 3 x 336 x 448, ImageNet-normalised. */
export function tileTensor(planes, row, col) {
  const out = new Float32Array(3 * TILE_H * TILE_W);
  for (let c = 0; c < 3; c++) {
    const p = planes[c], base = c * TILE_H * TILE_W;
    for (let y = 0; y < TILE_H; y++) {
      const src = (row * TILE_H + y) * FRAME_W + col * TILE_W, dst = base + y * TILE_W;
      for (let x = 0; x < TILE_W; x++) out[dst + x] = (p[src + x] / 255 - MEAN[c]) / STD[c];
    }
  }
  return out;
}

export const MAP_W = (GRID * TILE_W) / PATCH, MAP_H = (GRID * TILE_H) / PATCH;      // 96 x 72 patches

/** Nine tile outputs (24 x 32 each, row-major tiles) -> one 72 x 96 evidence map and the frame's P(good). */
export function assemble(tileEvidence) {
  const th = TILE_H / PATCH, tw = TILE_W / PATCH, map = new Float32Array(MAP_W * MAP_H);
  let sum = 0;
  tileEvidence.forEach((e, t) => {
    const row = Math.floor(t / GRID), col = t % GRID;
    for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) { const v = e[y * tw + x]; map[(row * th + y) * MAP_W + col * tw + x] = v; sum += v; }
  });
  return { map, pGood: 1 / (1 + Math.exp(-sum / map.length)) };
}

export function decide(pGood, flags, tPass, target) {
  if (flags.length) return { decision: "REACQUIRE", reasons: flags };
  const pct = `${Math.round(target * 100)}%`;
  if (pGood >= tPass) return { decision: "PASS", reasons: [`P(good) ${pGood.toFixed(2)} is at or above the pass threshold ${tPass.toFixed(2)} (target: at most ${pct} of passed frames bad)`] };
  return { decision: "REVIEW", reasons: [`P(good) ${pGood.toFixed(2)} is below the pass threshold ${tPass.toFixed(2)} (${pGood < 0.5 ? "likely bad: review first" : "uncertain"})`] };
}

/** Evidence map -> RGBA overlay at (w, h): blue pulls towards good, red towards bad; drawn over the grey frame. */
export function overlayPixels(greyBase, map, w, h, strength = 0.55) {
  const abs = Float32Array.from(map, Math.abs).sort();
  const scale = Math.max(1, abs[Math.floor(abs.length * 0.98)]);
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(MAP_H - 1, Math.max(0, ((y + 0.5) * MAP_H) / h - 0.5)), y0 = Math.floor(fy), y1 = Math.min(MAP_H - 1, y0 + 1), wy = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(MAP_W - 1, Math.max(0, ((x + 0.5) * MAP_W) / w - 0.5)), x0 = Math.floor(fx), x1 = Math.min(MAP_W - 1, x0 + 1), wx = fx - x0;
      const e = (map[y0 * MAP_W + x0] * (1 - wx) + map[y0 * MAP_W + x1] * wx) * (1 - wy) + (map[y1 * MAP_W + x0] * (1 - wx) + map[y1 * MAP_W + x1] * wx) * wy;
      const a = Math.min(1, Math.abs(e) / scale) * strength, g = greyBase[y * w + x], i = (y * w + x) * 4;
      const tint = e >= 0 ? [42, 120, 214] : [227, 73, 72];
      out[i] = g * (1 - a) + tint[0] * a; out[i + 1] = g * (1 - a) + tint[1] * a; out[i + 2] = g * (1 - a) + tint[2] * a; out[i + 3] = 255;
    }
  }
  return out;
}
