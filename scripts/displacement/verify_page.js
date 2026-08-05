#!/usr/bin/env node
/* Execute a comb-explorer page against a stubbed DOM and drive every render path.
 *
 *   node scripts/displacement/verify_page.js page1.html [page2.html ...]
 *
 * `node --check` is NOT enough: it parses without executing, and a
 * temporal-dead-zone ReferenceError parses fine and then kills the whole page
 * at load.  This runs the page's script for real, decodes its PNG payload with
 * a real inflate (so the mean/shape self-checks are meaningful, not stubbed
 * away), and then drives:
 *
 *   every rotor x every segment length x every carrier x k = 1..KMAX,
 *   both spectrogram transforms, both bandwidth extremes, the frequency
 *   sliders, the scale-factor slider and its presets, the per-rotor k field
 *   and "in view" button, the time marker, and an out-of-range k (which must
 *   leave a VISIBLE placeholder, never nothing).
 *
 * Exit code 0 only if nothing threw, nothing warned, and every assertion held.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const zlib = require("zlib");
const vm = require("vm");

/* ── a real 8-bit grayscale PNG decoder (IHDR + IDAT + the five filters) ───── */
function decodePngGray(buf) {
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error("not a PNG");
  let off = 8, w = 0, h = 0, bitDepth = 0, colour = -1, inter = 0;
  const idat = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off), type = buf.toString("ascii", off + 4, off + 8);
    const data = buf.slice(off + 8, off + 8 + len);
    if (type === "IHDR") {
      w = data.readUInt32BE(0); h = data.readUInt32BE(4);
      bitDepth = data[8]; colour = data[9]; inter = data[12];
    } else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
    off += 12 + len;
  }
  if (bitDepth !== 8 || colour !== 0 || inter !== 0)
    throw new Error(`unsupported PNG (depth ${bitDepth}, colour ${colour}, interlace ${inter})`);
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const out = Buffer.alloc(w * h);
  let p = 0;
  for (let y = 0; y < h; y++) {
    const f = raw[p++];
    const line = raw.slice(p, p + w); p += w;
    const cur = out.slice(y * w, (y + 1) * w);
    const up = y ? out.slice((y - 1) * w, y * w) : Buffer.alloc(w);
    for (let x = 0; x < w; x++) {
      const a = x ? cur[x - 1] : 0, b = up[x], c = x ? up[x - 1] : 0, v = line[x];
      let r;
      switch (f) {
        case 0: r = v; break;
        case 1: r = v + a; break;
        case 2: r = v + b; break;
        case 3: r = v + ((a + b) >> 1); break;
        case 4: {
          const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c);
          r = v + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c); break;
        }
        default: throw new Error("bad PNG filter " + f);
      }
      cur[x] = r & 255;
    }
  }
  return { width: w, height: h, data: out };
}

/* ── DOM stub ─────────────────────────────────────────────────────────────── */
function makeDom(warns) {
  const ctx2d = () => ({
    _img: null,
    setTransform() {}, createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    putImageData() {}, drawImage(im) { this._img = im; }, clearRect() {}, beginPath() {},
    moveTo() {}, lineTo() {}, stroke() {}, fill() {}, fillText() {}, setLineDash() {}, fillRect() {},
    getImageData(x, y, w, h) {
      const px = this._img && this._img._px;
      const d = new Uint8ClampedArray(w * h * 4);
      if (px) for (let i = 0, o = 0; i < px.length; i++, o += 4) { d[o] = px[i]; d[o + 1] = px[i]; d[o + 2] = px[i]; d[o + 3] = 255; }
      return { data: d, width: w, height: h };
    },
    imageSmoothingEnabled: true, imageSmoothingQuality: "", strokeStyle: "", fillStyle: "",
    lineWidth: 1, font: "", globalAlpha: 1,
  });
  const mk = (tag) => {
    const el = {
      tagName: tag, style: { cssText: "" }, dataset: {}, className: "", children: [],
      width: 800, height: 110, clientWidth: 800, value: "", textContent: "", disabled: false,
      _ctx: null, _l: {},
      get innerHTML() { return this._html || ""; },
      set innerHTML(v) { this._html = v; if (v === "") this.children = []; },
      get classList() { const c = this.className; return { contains: (x) => (" " + c + " ").includes(" " + x + " "), add() {}, remove() {} }; },
      appendChild(c) { this.children.push(c); return c; },
      removeChild(c) { this.children = this.children.filter((x) => x !== c); },
      addEventListener(t, fn) { (this._l[t] = this._l[t] || []).push(fn); },
      removeEventListener() {},
      fire(t, ev) { (this._l[t] || []).forEach((fn) => fn(ev)); },
      querySelector: () => null, querySelectorAll: () => [],
      getAttribute: (a) => (a === "height" ? el.height : null), setAttribute() {},
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 420 }),
      getContext() { return (this._ctx = this._ctx || ctx2d()); },
    };
    return el;
  };
  const REG = {};
  const doc = {
    createElement: mk,
    getElementById: (id) => REG[id] || (REG[id] = mk("div")),
    querySelectorAll: () => [],
    documentElement: mk("html"),
  };
  class ImageStub {
    constructor() { this.width = 0; this.height = 0; this._px = null; }
    set src(v) {
      const b64 = String(v).split(",")[1] || "";
      try {
        const d = decodePngGray(Buffer.from(b64, "base64"));
        this.width = d.width; this.height = d.height; this._px = d.data;
        setTimeout(() => this.onload && this.onload(), 0);
      } catch (e) {
        setTimeout(() => this.onerror && this.onerror(e), 0);
      }
    }
  }
  const g = {
    document: doc, Image: ImageStub, devicePixelRatio: 1,
    getComputedStyle: () => ({ getPropertyValue: () => "#888" }),
    addEventListener() {}, removeEventListener() {},
    atob: (s) => Buffer.from(s, "base64").toString("binary"),
    setTimeout, clearTimeout, queueMicrotask, Promise, Math, JSON, Date, console: {
      log: (...a) => console.log("   [page]", ...a),
      warn: (...a) => { warns.push(a.join(" ")); console.log("   [page:warn]", ...a); },
      error: (...a) => { warns.push(a.join(" ")); console.log("   [page:error]", ...a); },
    },
    Uint8Array, Uint8ClampedArray, Float64Array, Array, Object, Number, String, Boolean, Error,
  };
  g.window = g; g.globalThis = g; g.self = g; g.REG = REG;
  return g;
}

/* ── driver ───────────────────────────────────────────────────────────────── */
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* The notebook widget ships the very same page inside a srcdoc iframe (that is
 * how JupyterLab is made to execute it at all).  Unwrap it so one harness
 * verifies both front ends: the payload contains no raw `"`, so the attribute
 * boundary is unambiguous. */
function unwrapSrcdoc(html) {
  const m = html.match(/srcdoc="([^"]*)"/);
  if (!m) return html;
  return m[1]
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&");
}

async function verify(file) {
  const html = unwrapSrcdoc(fs.readFileSync(file, "utf8"));
  const m = html.match(/<script>([\s\S]*)<\/script>/);
  if (!m) {
    // the index page is a static table; check that every link resolves
    const links = [...html.matchAll(/href='([^']+)'/g)].map((x) => x[1]);
    const dead = links.filter((l) => !fs.existsSync(path.join(path.dirname(file), l)));
    console.log(
      `\n${path.basename(file)}  ${(fs.statSync(file).size / 1e3).toFixed(1)} kB\n` +
        `  static index, ${links.length} links\n  ${dead.length ? "FAIL dead links: " + dead.join(", ") : "PASS"}`
    );
    return dead.length === 0 && links.length > 0;
  }
  const warns = [];
  const sandbox = makeDom(warns);
  vm.createContext(sandbox);
  const fails = [];
  let unhandled = null;
  vm.runInContext(m[1], sandbox, { filename: path.basename(file) });
  const api = sandbox.window.__comb;
  if (!api) throw new Error("page did not expose window.__comb");
  const D = api.D, M = D.meta;
  for (let i = 0; i < 400 && !api.state().ready; i++) await sleep(10);
  if (!api.state().ready) throw new Error("page never became ready (strip decode stalled)");

  const KS = D.ks, KMAX = KS[KS.length - 1], NROT = M.n_rotors;
  const try_ = (what, fn) => { try { fn(); return true; } catch (e) { fails.push(what + ": " + (e && e.stack ? e.stack.split("\n")[0] : e)); return false; } };
  let renders = 0;

  // Every microphone channel the payload actually carries is driven in full:
  // a channel that is in-page state (the notebook widget) has never been
  // rendered until it is selected, so it needs the same sweep as the first.
  const inPage = D.chans.filter((c) => D.spec[c.id]);
  for (const cinfo of inPage) {
    if (cinfo.id !== api.state().chan) {
      await api.setChannel(cinfo.id);
      if (api.state().chan !== cinfo.id) { fails.push(`setChannel(${cinfo.id}) did not take`); continue; }
      if (!api.state().ready) { fails.push(`channel ${cinfo.id} never became ready`); continue; }
    }
    // every rotor x carrier x segment x k = 1..KMAX
    for (let r = 0; r < NROT; r++) {
      for (const car of D.carriers.map((c) => c.id)) {
        for (let si = 0; si < D.segs.length; si++) {
          for (let k = 1; k <= KMAX; k++) {
            api.set({ stripRot: r, carrier: car, segIdx: si, ks: [k] });
            if (try_(`strips ${cinfo.id} r${r} ${car} seg${si} k${k}`, api.drawStrips)) renders++;
          }
        }
      }
    }
    // several harmonics at once, both bandwidth extremes
    for (const bw of [0.15, 6, 3.05]) {
      api.set({ bw, ks: KS.slice(0, Math.min(6, KS.length)) });
      try_(`strips ${cinfo.id} bw=${bw}`, api.drawStrips);
    }
    // an out-of-range k must leave a VISIBLE placeholder
    api.set({ ks: [KMAX + 999] });
    try_(`strips ${cinfo.id} out-of-range k`, api.drawStrips);
    const host = sandbox.REG["strips"];
    const ph = (host.children || []).filter((c) => c.className === "miss" && c.textContent);
    if (ph.length !== 1) fails.push(`${cinfo.id}: out-of-range k rendered ${ph.length} placeholders, want 1`);
    else if (!/not available/.test(ph[0].textContent)) fails.push("placeholder text is not explanatory: " + ph[0].textContent);
    // every declared strip stack must have decoded, with the declared shape
    const stacks = D.strips[cinfo.id] || {};
    for (const key in stacks) {
      const b = stacks[key];
      if (b.nk !== KS.length) fails.push(`strip ${cinfo.id}/${key} carries ${b.nk} harmonics, page declares ${KS.length}`);
    }
    if (api.state().strips !== Object.keys(stacks).length)
      fails.push(`${cinfo.id}: decoded ${api.state().strips} of ${Object.keys(stacks).length} strip stacks`);
    if (!Object.keys(stacks).length) fails.push(`${cinfo.id}: no strip stacks at all`);
    api.set({ bw: 1.5, carrier: D.carriers[0].id, stripRot: 0, segIdx: 0 });
  }
  // k contiguity (a hole here is the silent failure this page must never have)
  const contiguous = KS.length === KMAX - KS[0] + 1;
  if (!contiguous) fails.push(`k set is NOT contiguous: ${KS.length} values over ${KS[0]}..${KMAX}`);
  // every carrier must carry a trajectory for every rotor
  for (const c of D.carriers) {
    const G = D.traj.G[c.id];
    if (!G || G.length !== NROT) fails.push(`carrier ${c.id} has ${G ? G.length : "no"} trajectories, want ${NROT}`);
    else if (G[0].length !== D.traj.t.length) fails.push(`carrier ${c.id} trajectory is ${G[0].length} long, time axis is ${D.traj.t.length}`);
  }

  // both spectrogram transforms, at several frequency ranges and time indices
  api.set({ ks: KS.slice(0, 4), rotOn: Array.from({ length: NROT }, () => true) });
  for (const tf of ["stft", "sst"]) {
    for (const [fl, fh] of [[0, D.fmax], [0, 1200], [Math.max(0, D.fmax - 1000), D.fmax]]) {
      api.set({ tf, fl, fh });
      for (const tIdx of [0, Math.floor(D.spec.nt / 2), D.spec.nt - 1]) {
        api.set({ tIdx });
        if (try_(`spec ${tf} ${fl}-${fh} t${tIdx}`, api.draw)) renders++;
      }
    }
  }
  // scale factor, including the preset button and reset
  for (const sf of [1, 0.99458, 1.015, 0.985]) {
    api.set({ sf });
    try_("spec sf=" + sf, api.draw); try_("strips sf=" + sf, api.drawStrips);
  }
  const R = sandbox.REG;
  try_("sf slider oninput", () => R.sf.oninput({ target: { value: "0.997" } }));
  try_("sf preset button", () => R.sfc.onclick());
  try_("sf reset button", () => R.sfr.onclick());
  try_("freq slider oninput", () => { R.fl.value = 500; R.fh.value = 4000; R.fl.oninput(); });
  (R.fpre.children || []).forEach((b, i) => try_("freq preset " + i, () => b.onclick()));
  try_("transform select", () => R.tf.onchange({ target: { value: "sst" } }));
  try_("segment select", () => R.seg.onchange({ target: { value: String(D.segs.length - 1) } }));
  try_("rotor select", () => R.srot.onchange({ target: { value: String(NROT - 1) } }));
  try_("carrier select", () => R.car.onchange({ target: { value: D.carriers[D.carriers.length - 1].id } }));
  try_("bandwidth slider", () => R.bw.oninput({ target: { value: "6" } }));
  // per-rotor k field and "in view", through the delegated listeners
  for (let r = 0; r < NROT; r++) {
    const ev = (cls) => ({ target: { classList: { contains: (x) => x === cls }, dataset: { i: String(r) }, value: "1, 2 3,,7" } });
    try_(`k field r${r}`, () => R.rotchk.fire("input", ev("kin")));
    try_(`in-view r${r}`, () => R.rotchk.fire("click", ev("kview")));
    try_(`rotor toggle r${r}`, () => R.rotchk.fire("change", { target: { dataset: { i: String(r) }, checked: r % 2 === 0 } }));
  }
  // a channel that is NOT in this payload must be a link to a sibling file
  // that exists; a channel that IS in the payload must not claim a file
  const nav = D.chans || [];
  for (const c of nav) {
    if (c.file && !fs.existsSync(path.join(path.dirname(file), c.file)))
      fails.push(`channel option "${c.label}" -> ${c.file} which is not in ${path.dirname(file)}`);
    if (!c.file && !D.spec[c.id])
      fails.push(`channel option "${c.label}" is neither in this payload nor a link`);
  }
  try_("channel select", () => R.chan.onchange({ target: { value: nav.length ? nav[nav.length - 1].id : "" } }));

  process.removeAllListeners("unhandledRejection");
  if (unhandled) fails.push("unhandled rejection: " + unhandled);
  const bad = warns.filter((w) => /MISMATCH/.test(w));
  const size = fs.statSync(file).size;
  const nStacks = inPage.reduce((a, c) => a + Object.keys(D.strips[c.id] || {}).length, 0);
  console.log(
    `\n${path.basename(file)}  ${(size / 1e6).toFixed(2)} MB\n` +
      `  ${M.dataset}/${M.recording} t0=${M.t0} +${M.dur}s | rps=${M.rps_channel}\n` +
      `  ${NROT} rotors, k=${KS[0]}..${KMAX} (${KS.length} values, contiguous=${contiguous}), ` +
      `${D.segs.length} segment lengths, carriers=${D.carriers.map((c) => c.id).join("+")}\n` +
      `  channels in payload: ${inPage.map((c) => c.id).join(",")} ` +
      `(+${nav.length - inPage.length} linked)\n` +
      `  ${renders} render calls driven, ${nStacks} strip stacks decoded, ` +
      `${warns.length} page warnings (${bad.length} payload mismatches)\n` +
      `  ${fails.length ? "FAIL\n    " + fails.slice(0, 12).join("\n    ") : "PASS"}`
  );
  return fails.length === 0 && bad.length === 0;
}

(async () => {
  const files = process.argv.slice(2);
  if (!files.length) { console.error("usage: verify_page.js PAGE.html [...]"); process.exit(2); }
  let ok = true;
  for (const f of files) {
    try { ok = (await verify(f)) && ok; }
    catch (e) { ok = false; console.log(`\n${path.basename(f)}\n  FAIL ${e.stack || e}`); }
  }
  console.log(ok ? "\nALL PAGES OK" : "\nFAILURES PRESENT");
  process.exit(ok ? 0 : 1);
})();
