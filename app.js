import { Tokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs";
import { SAMPLES } from "./samples.js";

/* ---------- Models ---------- */
// mb = size of tokenizer.json, shown so visitors know what they are downloading.
const MODELS = [
  { id: "gpt-4o",   name: "GPT-4o",      org: "OpenAI · o200k",  repo: "Xenova/gpt-4o",               mb: 9.7,  def: true },
  { id: "gpt-4",    name: "GPT-4 / 3.5", org: "OpenAI · cl100k", repo: "Xenova/gpt-4",                mb: 4.2,  def: true },
  { id: "llama3",   name: "Llama 3",     org: "Meta",            repo: "Xenova/llama3-tokenizer-new", mb: 9.1,  def: true },
  { id: "gemma2",   name: "Gemma 2",     org: "Google",          repo: "unsloth/gemma-2-2b",          mb: 17.5 },
  { id: "qwen3",    name: "Qwen 3",      org: "Alibaba",         repo: "Qwen/Qwen3-0.6B",             mb: 11.4 },
  { id: "deepseek", name: "DeepSeek-V3", org: "DeepSeek",        repo: "deepseek-ai/DeepSeek-V3",     mb: 7.8 },
  { id: "mistral",  name: "Mistral 7B",  org: "Mistral AI",      repo: "Xenova/mistral-tokenizer-v3", mb: 2.0 },
  { id: "bloom",    name: "BLOOM",       org: "BigScience",      repo: "Xenova/bloom-560m",           mb: 14.5 },
];

const CACHE = "token-tax-v1";
const PIECE_LIMIT = 400;

const state = {
  splitId: null,
  sample: 0,
  bench: new Map(), // model id -> ratio on SAMPLES
};

const $ = (s) => document.querySelector(s);
const ready = () => MODELS.filter((m) => m.tok);

/* ---------- Loading ---------- */
async function fetchJson(url, onProgress) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch {}
  let res = null;
  if (cache) { try { res = await cache.match(url); } catch {} }
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} for ${url}`);
    if (cache) cache.put(url, res.clone()).catch(() => {});
  }

  const total = Number(res.headers.get("content-length")) || 0;
  if (!onProgress || !total || !res.body) return res.json();

  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(Math.min(got / total, 1));
  }
  const bytes = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { bytes.set(c, off); off += c.length; }
  return JSON.parse(new TextDecoder().decode(bytes));
}

const hub = (m, file) => `https://huggingface.co/${m.repo}/resolve/main/${file}`;

async function load(m) {
  if (m.tok || m.loading) return;
  m.loading = true;
  setChip(m, "loading", 0);
  try {
    const [json, cfg] = await Promise.all([
      fetchJson(hub(m, "tokenizer.json"), (p) => setChip(m, "loading", p)),
      fetchJson(hub(m, "tokenizer_config.json")),
    ]);
    m.tok = new Tokenizer(json, cfg);
    setChip(m, "ready", 1);
    if (!state.splitId) state.splitId = m.id;
    state.bench.set(m.id, benchmark(m));
    renderAll();
  } catch (err) {
    console.error(`Could not load ${m.name}`, err);
    setChip(m, "error", 0);
  } finally {
    m.loading = false;
  }
}

/* ---------- Tokenizing ---------- */
const encode = (m, text) => (text ? m.tok.encode(text, { add_special_tokens: false }).ids : []);

// Groups token ids into readable pieces. Byte-level tokenizers often split one
// Bangla letter across several tokens; those are merged until the text is valid.
// Decoding with the previous piece as context keeps SentencePiece spaces intact.
function pieces(m, ids, limit = Infinity) {
  const dec = (a) => m.tok.decode(a, { skip_special_tokens: true });
  const out = [];
  let buf = [], prev = [], prevText = "";
  for (const id of ids) {
    if (out.length >= limit) break;
    buf.push(id);
    const full = dec(prev.concat(buf));
    const text = full.startsWith(prevText) ? full.slice(prevText.length) : dec(buf);
    if (!text.includes("�")) {
      out.push({ text, n: buf.length });
      prev = buf; prevText = dec(prev); buf = [];
    }
  }
  if (buf.length && out.length < limit) out.push({ text: dec(buf), n: buf.length });
  return out;
}

function benchmark(m) {
  let bn = 0, en = 0;
  for (const s of SAMPLES) { bn += encode(m, s.bn).length; en += encode(m, s.en).length; }
  return bn / en;
}

/* ---------- Helpers ---------- */
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (x) => (x >= 10 ? x.toFixed(1) : x.toFixed(2)).replace(/\.?0+$/, "");
const ratioText = (x) => `${x.toFixed(x < 10 ? 2 : 1)}×`;

let segmenter = null;
try { segmenter = new Intl.Segmenter("bn", { granularity: "grapheme" }); } catch {}
const chars = (t) => (segmenter ? [...segmenter.segment(t)].length : [...t].length);
const words = (t) => t.trim().split(/\s+/).filter(Boolean).length;

/* ---------- Model chips ---------- */
function buildChips() {
  const box = $("#models");
  for (const m of MODELS) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "model";
    b.dataset.id = m.id;
    b.innerHTML = `<span class="model__state" aria-hidden="true"></span>${esc(m.name)} <small>${m.mb} MB</small><span class="model__bar"></span>`;
    b.addEventListener("click", () => load(m));
    m.chip = b;
    setChip(m, "idle", 0);
    box.append(b);
  }
}

function setChip(m, s, p) {
  const b = m.chip;
  if (!b) return;
  b.dataset.state = s;
  b.style.setProperty("--p", s === "loading" ? p : 0);
  b.disabled = s === "ready" || s === "loading";
  const label = {
    idle: `Load ${m.name} tokenizer (${m.mb} MB)`,
    loading: `Loading ${m.name}… ${Math.round(p * 100)}%`,
    ready: `${m.name} loaded`,
    error: `${m.name} failed to load. Tap to retry.`,
  }[s];
  b.title = label;
  b.setAttribute("aria-label", label);
}

/* ---------- Samples ---------- */
function buildSamples() {
  const box = $("#samples");
  SAMPLES.forEach((s, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "sample";
    b.textContent = s.topic;
    b.addEventListener("click", () => useSample(i));
    box.append(b);
  });
}

function useSample(i) {
  state.sample = i;
  $("#in-bn").value = SAMPLES[i].bn;
  $("#in-en").value = SAMPLES[i].en;
  markSample();
  renderAll();
}

function markSample() {
  document.querySelectorAll(".sample").forEach((b, i) => b.setAttribute("aria-pressed", String(i === state.sample)));
}

/* ---------- Render: compare ---------- */
function renderCompare(bnText, enText) {
  const rows = ready().map((m) => ({ m, bn: encode(m, bnText).length, en: encode(m, enText).length }));
  const tbody = $("#results tbody");

  if (!rows.length) {
    tbody.innerHTML = `<tr class="empty"><td colspan="4">Loading tokenizers…</td></tr>`;
    return;
  }

  const max = Math.max(1, ...rows.flatMap((r) => [r.bn, r.en]));
  tbody.innerHTML = rows.map(({ m, bn, en }) => {
    const ratio = bn && en ? bn / en : null;
    const ratioCell = ratio
      ? `<span class="ratio${ratio >= 2 ? " ratio--hi" : ""}">${ratioText(ratio)}</span>`
      : `<span class="muted">—</span>`;
    const takaCell = ratio
      ? `<span class="taka">৳${Math.round(100 * ratio).toLocaleString("en-IN")}<small>in Bangla</small></span>`
      : `<span class="muted">Needs both texts</span>`;
    return `<tr>
      <td><span class="m-name">${esc(m.name)}</span><span class="m-org">${esc(m.org)}</span></td>
      <td><div class="bars">
        <div class="bar bar--bn"><i style="--w:${(bn / max) * 100}%"></i><span>${bn} bn</span></div>
        <div class="bar bar--en"><i style="--w:${(en / max) * 100}%"></i><span>${en} en</span></div>
      </div></td>
      <td class="num">${ratioCell}</td>
      <td class="num">${takaCell}</td>
    </tr>`;
  }).join("");
}

/* ---------- Render: split ---------- */
function chipHtml(p) {
  let body = esc(p.text).replace(/ /g, '<span class="sp">·</span>').replace(/\n/g, '<span class="sp">↵</span>');
  if (!body) body = '<span class="sp">␣</span>';
  const t = `${p.n} token${p.n > 1 ? "s" : ""}`;
  return `<span class="tk" title="${t}">${body}${p.n > 1 ? `<sup>${p.n}</sup>` : ""}</span>`;
}

function renderTokens(el, m, text, ids) {
  if (!text) { el.innerHTML = `<span class="placeholder">Nothing to show yet.</span>`; return []; }
  const ps = pieces(m, ids, PIECE_LIMIT);
  const shown = ps.reduce((a, p) => a + p.n, 0);
  el.innerHTML = ps.map(chipHtml).join("") +
    (shown < ids.length ? `<span class="tokens__more">+${ids.length - shown} more tokens</span>` : "");
  return ps;
}

function renderSplit(bnText, enText) {
  const models = ready();
  const tabs = $("#split-tabs");
  tabs.innerHTML = models.map((m) =>
    `<button type="button" class="tab" role="tab" aria-selected="${m.id === state.splitId}" data-id="${m.id}">${esc(m.name)}</button>`
  ).join("");

  const m = models.find((x) => x.id === state.splitId);
  if (!m) {
    $("#split-stats").textContent = "";
    $("#split-bn").innerHTML = `<span class="placeholder">Loading tokenizers…</span>`;
    $("#split-en").innerHTML = "";
    return;
  }

  const bnIds = encode(m, bnText);
  const enIds = encode(m, enText);
  const bnPieces = renderTokens($("#split-bn"), m, bnText, bnIds);
  renderTokens($("#split-en"), m, enText, enIds);

  const broken = bnPieces.filter((p) => p.n > 1).length;
  const bw = words(bnText), ew = words(enText);
  $("#split-stats").innerHTML = [
    `Bangla: <b>${bnIds.length}</b> tokens${bw ? ` · <b>${fmt(bnIds.length / bw)}</b> per word` : ""}`,
    `English: <b>${enIds.length}</b> tokens${ew ? ` · <b>${fmt(enIds.length / ew)}</b> per word` : ""}`,
    `Bangla pieces split into bytes: <b>${broken}</b>`,
  ].map((s) => `<span>${s}</span>`).join("");
}

/* ---------- Render: benchmark + hero ---------- */
function renderBench() {
  const rows = ready().map((m) => ({ m, r: state.bench.get(m.id) })).filter((x) => x.r).sort((a, b) => b.r - a.r);
  const box = $("#bench");
  if (!rows.length) { box.innerHTML = `<p class="placeholder">Loading tokenizers…</p>`; return; }

  const max = Math.max(1, rows[0].r);
  box.innerHTML = rows.map(({ m, r }) => `
    <div class="brow">
      <div class="brow__name">${esc(m.name)}<small>${esc(m.org)}</small></div>
      <div class="brow__track" aria-hidden="true">
        <div class="brow__fill" style="--w:${(r / max) * 100}%"></div>
        <div class="brow__par" style="--par:${(1 / max) * 100}%" title="1.0× = parity with English"></div>
      </div>
      <div class="brow__val">${ratioText(r)}</div>
    </div>`).join("") +
    (rows.length < MODELS.length ? `<p class="placeholder">Load more models above to add them to the ranking.</p>` : "");

  const worst = rows[0], best = rows[rows.length - 1];
  $("#stat-worst").textContent = ratioText(worst.r);
  $("#stat-worst-label").textContent = `as many tokens for Bangla as for English in ${worst.m.name}, the worst tokenizer loaded`;
  if (rows.length > 1) {
    $("#stat-best").textContent = ratioText(best.r);
    $("#stat-best-label").textContent = `in ${best.m.name}, the fairest tokenizer loaded`;
  }
}

function renderMeta(bnText, enText) {
  const meta = (t) => (t ? `${chars(t)} characters · ${words(t)} words` : "");
  $("#meta-bn").textContent = meta(bnText);
  $("#meta-en").textContent = meta(enText);
}

function renderAll() {
  const bnText = $("#in-bn").value;
  const enText = $("#in-en").value;
  renderMeta(bnText, enText);
  renderCompare(bnText, enText);
  renderSplit(bnText, enText);
  renderBench();
}

/* ---------- Theme ---------- */
function setupTheme() {
  $("#theme").addEventListener("click", () => {
    const root = document.documentElement;
    const current = root.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    const next = current === "light" ? "dark" : "light";
    root.dataset.theme = next;
    try { localStorage.setItem("tt-theme", next); } catch {}
  });
}

/* ---------- Init ---------- */
function init() {
  setupTheme();
  buildChips();
  buildSamples();

  let timer;
  const onInput = () => {
    state.sample = -1;
    markSample();
    clearTimeout(timer);
    timer = setTimeout(renderAll, 120);
  };
  $("#in-bn").addEventListener("input", onInput);
  $("#in-en").addEventListener("input", onInput);

  $("#split-tabs").addEventListener("click", (e) => {
    const b = e.target.closest(".tab");
    if (!b) return;
    state.splitId = b.dataset.id;
    renderSplit($("#in-bn").value, $("#in-en").value);
  });

  $("#bench-n").textContent = SAMPLES.length;
  useSample(0);
  MODELS.filter((m) => m.def).forEach(load);
}

init();
