import { Tokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs";
import { SAMPLES } from "./samples.js";
import { PRECOMPUTED, WORDS } from "./precomputed.js";

/* ---------- Models ---------- */
// mb = size of tokenizer.json, shown so visitors know what they are downloading.
// def = downloaded automatically once the visitor types their own text.
// official = the repo belongs to the model's maker (otherwise a community copy).
const MODELS = [
  { id: "gpt-4",    name: "GPT-4 / 3.5", org: "OpenAI · cl100k", repo: "Xenova/gpt-4",                mb: 4.2,  def: true },
  { id: "gpt-4o",   name: "GPT-4o",      org: "OpenAI · o200k",  repo: "Xenova/gpt-4o",               mb: 9.7,  def: true },
  { id: "llama3",   name: "Llama 3",     org: "Meta",            repo: "Xenova/llama3-tokenizer-new", mb: 9.1,  def: true },
  { id: "gemma2",   name: "Gemma 2",     org: "Google",          repo: "unsloth/gemma-2-2b",          mb: 17.5 },
  { id: "qwen3",    name: "Qwen 3",      org: "Alibaba",         repo: "Qwen/Qwen3-0.6B",             mb: 11.4, official: true },
  { id: "deepseek", name: "DeepSeek-V3", org: "DeepSeek",        repo: "deepseek-ai/DeepSeek-V3",     mb: 7.8,  official: true },
  { id: "mistral",  name: "Mistral 7B",  org: "Mistral AI",      repo: "Xenova/mistral-tokenizer-v3", mb: 2.0 },
  { id: "bloom",    name: "BLOOM",       org: "BigScience",      repo: "Xenova/bloom-560m",           mb: 14.5 },
];
// Counts for these were cross-checked against OpenAI's published encodings.
const VERIFIED = new Set(["gpt-4", "gpt-4o"]);
const byId = Object.fromEntries(MODELS.map((m) => [m.id, m]));

const CACHE = "token-tax-v1";
const PIECE_LIMIT = 400;
const sampleIndex = {
  bn: new Map(SAMPLES.map((s, i) => [s.bn, i])),
  en: new Map(SAMPLES.map((s, i) => [s.en, i])),
};

const state = { sample: 0, cutId: "gpt-4" };

const $ = (s) => document.querySelector(s);
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const times = (x) => `${x.toFixed(x < 10 ? 2 : 1)}×`;
const taka = (n) => `৳${Math.round(n).toLocaleString("en-IN")}`;
const sum = (ps) => ps.reduce((a, p) => a + p.n, 0);

/* ---------- Results (pre-computed or live) ---------- */
const toPieces = (arr) => arr.map(([text, ids]) => ({ text, ids, n: ids.length }));

// Benchmark ratio per model over every sample pair, from the pre-computed data.
const RANKING = MODELS.map((m) => {
  let bn = 0, en = 0;
  for (const r of PRECOMPUTED[m.id]) { bn += sum(toPieces(r.bn)); en += sum(toPieces(r.en)); }
  return { m, r: bn / en };
}).sort((a, b) => b.r - a.r);

function result(m, lang, text) {
  if (!text) return { count: 0, pieces: [] };
  if (m.tok) {
    const ids = m.tok.encode(text, { add_special_tokens: false }).ids;
    return { count: ids.length, ids };
  }
  const i = sampleIndex[lang].get(text);
  if (i === undefined) return null;
  const pieces = toPieces(PRECOMPUTED[m.id][i][lang]);
  return { count: sum(pieces), pieces };
}

// Groups token ids into readable pieces. Byte-level tokenizers often split one
// Bangla letter across several tokens; those are merged until the text is valid.
// Decoding with the previous piece as context keeps SentencePiece spaces intact.
function piecesFromIds(m, ids, limit) {
  const dec = (a) => m.tok.decode(a, { skip_special_tokens: true });
  const out = [];
  let buf = [], prev = [], prevText = "";
  for (const id of ids) {
    if (out.length >= limit) break;
    buf.push(id);
    const full = dec(prev.concat(buf));
    const text = full.startsWith(prevText) ? full.slice(prevText.length) : dec(buf);
    if (!text.includes("�")) {
      out.push({ text, ids: buf, n: buf.length });
      prev = buf; prevText = dec(prev); buf = [];
    }
  }
  if (buf.length && out.length < limit) out.push({ text: dec(buf), ids: buf, n: buf.length });
  return out;
}

const isCustom = (bn, en) => (bn && !sampleIndex.bn.has(bn)) || (en && !sampleIndex.en.has(en));

/* ---------- Loading tokenizers ---------- */
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
  m.error = false;
  m.p = 0;
  scheduleRender();
  try {
    const [json, cfg] = await Promise.all([
      fetchJson(hub(m, "tokenizer.json"), (p) => { m.p = p; scheduleProgress(); }),
      fetchJson(hub(m, "tokenizer_config.json")),
    ]);
    m.tok = new Tokenizer(json, cfg);
  } catch (err) {
    console.error(`Could not load ${m.name}`, err);
    m.error = true;
  } finally {
    m.loading = false;
    scheduleRender();
  }
}

const loadDefaults = () => MODELS.filter((m) => m.def).forEach(load);

/* ---------- Render: status ---------- */
function renderStatus(custom) {
  const el = $("#status");
  const loading = MODELS.filter((m) => m.loading);
  const failed = MODELS.filter((m) => m.error && !m.tok);
  const live = MODELS.filter((m) => m.tok);

  let cls = "", html = "";
  if (failed.length) {
    cls = "status--err";
    html = `Couldn't download ${failed.map((m) => esc(m.name)).join(", ")}. Check your connection, then tap <b>Retry</b> in the table.`;
  } else if (loading.length) {
    const total = loading.reduce((a, m) => a + m.mb, 0);
    const done = loading.reduce((a, m) => a + m.mb * (m.p || 0), 0);
    const p = total ? done / total : 0;
    cls = "status--wait";
    html = `Downloading ${loading.length} tokenizer${loading.length > 1 ? "s" : ""} to measure your text · ${Math.round(p * 100)}% of ${total.toFixed(1)} MB
      <span class="status__bar" style="--p:${p}"><i></i></span>`;
  } else if (custom) {
    html = `Live: your text is measured in your browser with ${live.length} tokenizer${live.length === 1 ? "" : "s"}. Load more from the table below.`;
  } else {
    html = `Saved results for this sample, all ${MODELS.length} models. Edit the text to measure your own, live.`;
  }
  el.className = `status ${cls}`;
  el.innerHTML = `<span class="status__dot"></span><span>${html}</span>`;
}

/* ---------- Render: ledger ---------- */
function loadButton(m) {
  if (m.loading) return `<button class="load" type="button" disabled data-progress="${m.id}">Downloading… ${Math.round((m.p || 0) * 100)}%</button>`;
  const label = m.error ? "Retry" : `Load tokenizer · ${m.mb} MB`;
  return `<button class="load" type="button" data-load="${m.id}">${label}</button>`;
}

function renderLedger(bnText, enText) {
  const rows = MODELS.map((m) => ({ m, bn: result(m, "bn", bnText), en: result(m, "en", enText) }));
  const max = Math.max(1, ...rows.flatMap((r) => [r.bn?.count || 0, r.en?.count || 0]));

  $("#ledger-rows").innerHTML = rows.map(({ m, bn, en }) => {
    const name = `<div class="lrow__name">${esc(m.name)}<small>${esc(m.org)}</small></div>`;
    if (!bn || !en) {
      return `<div class="lrow" role="row">${name}
        <div role="cell">${loadButton(m)}</div>
        <div class="r" role="cell"><span class="na">—</span></div>
        <div class="r taka-cell" role="cell"><span class="na">—</span></div>
      </div>`;
    }
    const ratio = bn.count && en.count ? bn.count / en.count : null;
    return `<div class="lrow" role="row">${name}
      <div class="bars" role="cell">
        <div class="bar bar--bn"><i style="--w:${(bn.count / max) * 100}%"></i>${bn.count}</div>
        <div class="bar bar--en"><i style="--w:${(en.count / max) * 100}%"></i>${en.count}</div>
      </div>
      <div class="r" role="cell">${ratio ? `<span class="mult${ratio >= 2 ? " mult--hi" : ""}">${times(ratio)}</span>` : `<span class="na">Needs both texts</span>`}</div>
      <div class="r taka-cell" role="cell">${ratio ? `<span class="taka">${taka(100 * ratio)}<small>in Bangla</small></span>` : `<span class="na">—</span>`}</div>
    </div>`;
  }).join("");
}

/* ---------- Render: under the knife ---------- */
function pieceHtml(p) {
  let body = esc(p.text).replace(/ /g, '<span class="sp">·</span>').replace(/\n/g, '<span class="sp">↵</span>');
  if (!body) body = '<span class="sp">␣</span>';
  const title = `${p.n} token${p.n > 1 ? "s" : ""}: ${p.ids.join(", ")}`;
  return `<span class="pc${p.n > 1 ? " multi" : ""}" title="${title}">${body}${p.n > 1 ? `<sup>${p.n}</sup>` : ""}<span class="pc__ids">${p.ids.join(" ")}</span></span>`;
}

function renderPieces(el, m, res, text) {
  if (!text) { el.innerHTML = `<span class="pieces__empty">Nothing to show yet.</span>`; return []; }
  if (!res) {
    el.innerHTML = `<span class="pieces__empty">Load this tokenizer to see how it cuts your text. ${loadButton(m)}</span>`;
    return null;
  }
  const ps = res.pieces ? res.pieces.slice(0, PIECE_LIMIT) : piecesFromIds(m, res.ids, PIECE_LIMIT);
  const shown = sum(ps);
  el.innerHTML = ps.map(pieceHtml).join("") +
    (shown < res.count ? `<span class="pieces__more">+${res.count - shown} more tokens</span>` : "");
  return ps;
}

function renderCut(bnText, enText) {
  $("#tabs").innerHTML = MODELS.map((m) =>
    `<button type="button" class="tab" role="tab" aria-selected="${m.id === state.cutId}" data-id="${m.id}">${esc(m.name)}</button>`
  ).join("");

  const m = byId[state.cutId];
  const bn = result(m, "bn", bnText);
  const en = result(m, "en", enText);
  const bnPieces = renderPieces($("#cut-bn"), m, bn, bnText);
  renderPieces($("#cut-en"), m, en, enText);

  const words = (t) => t.trim().split(/\s+/).filter(Boolean).length;
  const per = (r, t) => (r && words(t) ? ` · <b>${(r.count / words(t)).toFixed(1)}</b> per word` : "");
  const stats = [];
  if (bn) stats.push(`বাংলা <b>${bn.count}</b> tokens${per(bn, bnText)}`);
  if (en) stats.push(`English <b>${en.count}</b> tokens${per(en, enText)}`);
  if (bnPieces?.length) stats.push(`<span class="hl">Letters chopped into bytes: <b>${bnPieces.filter((p) => p.n > 1).length}</b></span>`);
  $("#cut-stats").innerHTML = stats.length ? stats.map((s) => `<span>${s}</span>`).join("") : `<span>${esc(m.name)} · ${esc(m.org)}</span>`;
}

/* ---------- Render: ranking, hero, receipt ---------- */
function renderRanking() {
  const max = RANKING[0].r;
  $("#rank").innerHTML = RANKING.map(({ m, r }, i) => `
    <div class="rk">
      <span class="rk__pos">${String(i + 1).padStart(2, "0")}</span>
      <div class="rk__name">${esc(m.name)}<small>${esc(m.org)}</small></div>
      <div class="rk__track" aria-hidden="true">
        <div class="rk__fill" style="--w:${(r / max) * 100}%"></div>
        <div class="rk__par" style="--par:${(1 / max) * 100}%"><span>fair · 1×</span></div>
      </div>
      <div class="rk__val">${times(r)}</div>
    </div>`).join("");
  $("#n-samples").textContent = SAMPLES.length;
}

function renderHero() {
  $("#hero-x").textContent = `${Math.round(RANKING[0].r)}×`;

  // Receipt: the first sample, billed by the GPT-4 tokenizer.
  const m = byId["gpt-4"];
  const bn = result(m, "bn", SAMPLES[0].bn).count;
  const en = result(m, "en", SAMPLES[0].en).count;
  const bill = (100 * bn) / en;
  $("#rc-model").textContent = m.name.split(" ")[0];
  $("#rc-text").textContent = SAMPLES[0].bn;
  $("#rc-en-n").textContent = `${en} tokens`;
  $("#rc-bn-n").textContent = `${bn} tokens`;
  $("#rc-bn-t").textContent = taka(bill);
  $("#rc-tax").textContent = `+${taka(bill - 100)}`;
}

/* ---------- Render: 3 steps ---------- */
function renderSteps() {
  const en = toPieces(WORDS.en), bn = toPieces(WORDS.bn);
  const enN = sum(en), bnN = sum(bn);
  const word = (ps, n, cls = "") => `<span class="pieces ${cls}">${ps.map(pieceHtml).join("")}</span><span class="step__count">${n} token${n > 1 ? "s" : ""}</span>`;
  $("#step-en").innerHTML = word(en, enN);
  $("#step-bn").innerHTML = word(bn, bnN, "bn");
  $("#step-bill").innerHTML = `
    <span class="step__line"><span>${esc(WORDS.en.map((p) => p[0]).join(""))}</span><b>${taka(enN)}</b></span>
    <span class="step__line bn"><span>${esc(WORDS.bn.map((p) => p[0]).join(""))}</span><b>${taka(bnN)}</b></span>
    <span class="step__x">${bnN}× the price</span>`;
  $("#step-note").textContent = `Measured with ${byId[WORDS.model].name}'s tokenizer, at an example rate of ৳1 per token.`;
}

/* ---------- Render: check it yourself ---------- */
function renderVerify(bnText, enText) {
  const parts = ["gpt-4", "gpt-4o"].map((id) => {
    const m = byId[id];
    const bn = result(m, "bn", bnText), en = result(m, "en", enText);
    if (!bn || !en) return `<span class="na">load ${esc(m.name)} above to get its numbers</span>`;
    return `<b>${esc(m.name)}</b> বাংলা <b>${bn.count}</b> · English <b>${en.count}</b>`;
  });
  $("#verify-nums").innerHTML = parts.join(" &nbsp;/&nbsp; ");
}

/* ---------- Render: tokenizer sources ---------- */
function renderSources() {
  $("#sources").innerHTML = `
    <div class="src__row src__head"><span>Model</span><span>Tokenizer file</span><span>Source</span><span>Cross-checked</span></div>` +
    MODELS.map((m) => `
    <div class="src__row">
      <span class="src__name">${esc(m.name)}<small>${esc(m.org)}</small></span>
      <a href="https://huggingface.co/${m.repo}" target="_blank" rel="noopener">${esc(m.repo)}</a>
      <span>${m.official ? "Official" : "Community copy"}</span>
      <span>${VERIFIED.has(m.id) ? `<b class="ok">✓ matches OpenAI</b>` : "—"}</span>
    </div>`).join("");
}

/* ---------- Downloads + copy ---------- */
function toCsv(rows) {
  const cell = (v) => { const s = String(v); return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  return rows.map((r) => r.map(cell).join(",")).join("\r\n");
}

function download(name, text) {
  // BOM so Excel opens the Bangla text as UTF-8.
  const url = URL.createObjectURL(new Blob(["﻿" + text], { type: "text/csv;charset=utf-8" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadAll() {
  const rows = [["sample", "topic", "model", "tokenizer", "bangla_text", "english_text", "bangla_tokens", "english_tokens", "ratio"]];
  SAMPLES.forEach((s, i) => MODELS.forEach((m) => {
    const bn = sum(toPieces(PRECOMPUTED[m.id][i].bn)), en = sum(toPieces(PRECOMPUTED[m.id][i].en));
    rows.push([i + 1, s.topic, m.name, m.repo, s.bn, s.en, bn, en, (bn / en).toFixed(3)]);
  }));
  download("token-tax-samples.csv", toCsv(rows));
}

function downloadCurrent() {
  const bnText = $("#in-bn").value, enText = $("#in-en").value;
  const rows = [["model", "tokenizer", "source", "bangla_text", "english_text", "bangla_tokens", "english_tokens", "ratio"]];
  for (const m of MODELS) {
    const bn = result(m, "bn", bnText), en = result(m, "en", enText);
    if (!bn || !en) continue;
    rows.push([m.name, m.repo, m.tok ? "live" : "saved", bnText, enText, bn.count, en.count, en.count ? (bn.count / en.count).toFixed(3) : ""]);
  }
  download("token-tax-result.csv", toCsv(rows));
}

async function copyText(btn, text) {
  const label = btn.textContent;
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = "Copied ✓";
  } catch {
    btn.textContent = "Select and copy manually";
  }
  setTimeout(() => { btn.textContent = label; }, 1600);
}

/* ---------- Render loop ---------- */
const graphemes = (() => {
  try { const s = new Intl.Segmenter("bn", { granularity: "grapheme" }); return (t) => [...s.segment(t)].length; }
  catch { return (t) => [...t].length; }
})();

function renderAll() {
  const bnText = $("#in-bn").value;
  const enText = $("#in-en").value;
  const meta = (t) => (t ? `${graphemes(t)} characters · ${t.trim().split(/\s+/).filter(Boolean).length} words` : "");
  $("#meta-bn").textContent = meta(bnText);
  $("#meta-en").textContent = meta(enText);

  const custom = isCustom(bnText, enText);
  renderStatus(custom);
  renderLedger(bnText, enText);
  renderVerify(bnText, enText);
  renderCut(bnText, enText);
}

let frame = 0;
function scheduleRender() {
  cancelAnimationFrame(frame);
  frame = requestAnimationFrame(renderAll);
}
// Progress ticks only touch the status line and the load buttons.
let progressFrame = 0;
function scheduleProgress() {
  if (progressFrame) return;
  progressFrame = requestAnimationFrame(() => {
    progressFrame = 0;
    renderStatus(isCustom($("#in-bn").value, $("#in-en").value));
    document.querySelectorAll("[data-progress]").forEach((b) => {
      b.textContent = `Downloading… ${Math.round((byId[b.dataset.progress].p || 0) * 100)}%`;
    });
  });
}

/* ---------- Samples ---------- */
function buildSamples() {
  $("#samples").innerHTML = SAMPLES.map((s, i) =>
    `<button type="button" class="pill" data-i="${i}" aria-pressed="${i === state.sample}">${esc(s.topic)}</button>`
  ).join("");
  $("#samples").addEventListener("click", (e) => {
    const b = e.target.closest(".pill");
    if (b) useSample(Number(b.dataset.i));
  });
}

function useSample(i) {
  state.sample = i;
  $("#in-bn").value = SAMPLES[i].bn;
  $("#in-en").value = SAMPLES[i].en;
  document.querySelectorAll(".pill").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.i) === i)));
  renderAll();
}

/* ---------- Theme ---------- */
function setupTheme() {
  $("#theme").addEventListener("click", () => {
    const root = document.documentElement;
    const current = root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("tt-theme", next); } catch {}
  });
}

/* ---------- Init ---------- */
function init() {
  setupTheme();
  buildSamples();
  renderHero();
  renderSteps();
  renderRanking();
  renderSources();
  $("#m-n").textContent = SAMPLES.length;

  $("#show-ids").addEventListener("change", (e) => $("#cut").classList.toggle("show-ids", e.target.checked));
  $("#dl-all").addEventListener("click", downloadAll);
  $("#dl-current").addEventListener("click", downloadCurrent);
  document.querySelectorAll("[data-copy]").forEach((b) =>
    b.addEventListener("click", () => copyText(b, $(`#in-${b.dataset.copy}`).value)));

  let timer;
  const onInput = () => {
    state.sample = -1;
    document.querySelectorAll(".pill").forEach((b) => b.setAttribute("aria-pressed", "false"));
    if (isCustom($("#in-bn").value, $("#in-en").value)) loadDefaults();
    clearTimeout(timer);
    timer = setTimeout(renderAll, 120);
  };
  $("#in-bn").addEventListener("input", onInput);
  $("#in-en").addEventListener("input", onInput);

  document.addEventListener("click", (e) => {
    const lb = e.target.closest("[data-load]");
    if (lb) load(byId[lb.dataset.load]);
    const tab = e.target.closest(".tab");
    if (tab) { state.cutId = tab.dataset.id; renderCut($("#in-bn").value, $("#in-en").value); }
  });

  useSample(0);
}

init();
