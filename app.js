import { Tokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs";
import { SAMPLES } from "./samples.js";
import { PRECOMPUTED, WORDS } from "./precomputed.js";
import { BN_STATIC, DYN } from "./i18n.js";

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

const state = { sample: 0, cutId: "gpt-4", lang: "en", cutPieces: { bn: [], en: [] }, inspect: null, flores: null };

const $ = (s) => document.querySelector(s);
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const times = (x) => `${x.toFixed(x < 10 ? 2 : 1)}×`;
const taka = (n) => `৳${Math.round(n).toLocaleString("en-IN")}`;
const sum = (ps) => ps.reduce((a, p) => a + p.n, 0);
const utf8 = new TextEncoder();

/* ---------- Language ---------- */
// t("key", ...args): runtime text in the current language.
function t(key, ...args) {
  const v = DYN[key][state.lang === "bn" ? 1 : 0];
  return typeof v === "function" ? v(...args) : v;
}
const topic = (s) => (state.lang === "bn" && DYN.topics[1][s.topic]) || s.topic;

// English for static text comes from the page itself, captured once.
const EN_STATIC = { html: new Map(), ph: new Map(), aria: new Map() };
function captureEnglish() {
  document.querySelectorAll("[data-i18n]").forEach((el) => EN_STATIC.html.set(el.dataset.i18n, el.innerHTML));
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => EN_STATIC.ph.set(el.dataset.i18nPh, el.placeholder));
  document.querySelectorAll("[data-i18n-aria]").forEach((el) => EN_STATIC.aria.set(el.dataset.i18nAria, el.getAttribute("aria-label")));
}

function applyLanguage() {
  const bn = state.lang === "bn";
  const pick = (map, key) => (bn && BN_STATIC[key] !== undefined ? BN_STATIC[key] : map.get(key));
  document.documentElement.lang = state.lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.innerHTML = pick(EN_STATIC.html, el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.placeholder = pick(EN_STATIC.ph, el.dataset.i18nPh); });
  document.querySelectorAll("[data-i18n-aria]").forEach((el) => el.setAttribute("aria-label", pick(EN_STATIC.aria, el.dataset.i18nAria)));
  const btn = $("#lang");
  btn.textContent = t("langBtn");
  btn.lang = bn ? "en" : "bn";
  btn.setAttribute("aria-label", t("langAria"));

  // Translated blocks replace elements that held live values; fill them again.
  renderHero();
  renderSteps();
  renderRanking();
  renderHeat();
  renderFlores();
  renderSources();
  $("#m-n").textContent = SAMPLES.length;
  document.querySelectorAll(".pill").forEach((b) => { b.textContent = topic(SAMPLES[Number(b.dataset.i)]); });
  renderWord();
  renderAll();
}

function setLanguage(lang) {
  state.lang = lang;
  try { localStorage.setItem("tt-lang", lang); } catch {}
  applyLanguage();
}

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
    html = t("statusErr", failed.map((m) => esc(m.name)).join(", "));
  } else if (loading.length) {
    const total = loading.reduce((a, m) => a + m.mb, 0);
    const done = loading.reduce((a, m) => a + m.mb * (m.p || 0), 0);
    const p = total ? done / total : 0;
    cls = "status--wait";
    html = `${t("statusWait", loading.length, Math.round(p * 100), total.toFixed(1))}
      <span class="status__bar" style="--p:${p}"><i></i></span>`;
  } else if (custom) {
    html = t("statusLive", live.length);
  } else {
    html = t("statusSaved", MODELS.length);
  }
  el.className = `status ${cls}`;
  el.innerHTML = `<span class="status__dot"></span><span>${html}</span>`;
}

/* ---------- Render: ledger ---------- */
function loadButton(m) {
  if (m.loading) return `<button class="load" type="button" disabled data-progress="${m.id}">${t("downloading", Math.round((m.p || 0) * 100))}</button>`;
  const label = m.error ? t("retry") : t("loadTok", m.mb);
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
      <div class="r" role="cell">${ratio ? `<span class="mult${ratio >= 2 ? " mult--hi" : ""}">${times(ratio)}</span>` : `<span class="na">${t("needsBoth")}</span>`}</div>
      <div class="r taka-cell" role="cell">${ratio ? `<span class="taka">${taka(100 * ratio)}<small>${t("inBangla")}</small></span>` : `<span class="na">—</span>`}</div>
    </div>`;
  }).join("");
}

/* ---------- Render: under the knife ---------- */
// lang + i make a piece clickable for the inspector.
function pieceHtml(p, lang, i) {
  let body = esc(p.text).replace(/ /g, '<span class="sp">·</span>').replace(/\n/g, '<span class="sp">↵</span>');
  if (!body) body = '<span class="sp">␣</span>';
  const pick = lang ? ` tabindex="0" role="button" data-lang="${lang}" data-i="${i}"` : "";
  return `<span class="pc${p.n > 1 ? " multi" : ""}" title="${esc(t("pieceTitle", p.n, p.ids.join(", ")))}"${pick}>${body}${p.n > 1 ? `<sup>${p.n}</sup>` : ""}<span class="pc__ids">${p.ids.join(" ")}</span></span>`;
}

function renderPieces(el, lang, m, res, text) {
  state.cutPieces[lang] = [];
  if (!text) { el.innerHTML = `<span class="pieces__empty">${t("nothing")}</span>`; return []; }
  if (!res) {
    el.innerHTML = `<span class="pieces__empty">${t("loadToSee")} ${loadButton(m)}</span>`;
    return null;
  }
  const ps = res.pieces ? res.pieces.slice(0, PIECE_LIMIT) : piecesFromIds(m, res.ids, PIECE_LIMIT);
  state.cutPieces[lang] = ps;
  const shown = sum(ps);
  el.innerHTML = ps.map((p, i) => pieceHtml(p, lang, i)).join("") +
    (shown < res.count ? `<span class="pieces__more">${t("moreTokens", res.count - shown)}</span>` : "");
  return ps;
}

function renderCut(bnText, enText) {
  $("#tabs").innerHTML = MODELS.map((m) =>
    `<button type="button" class="tab" role="tab" aria-selected="${m.id === state.cutId}" data-id="${m.id}">${esc(m.name)}</button>`
  ).join("");

  const m = byId[state.cutId];
  const bn = result(m, "bn", bnText);
  const en = result(m, "en", enText);
  const bnPieces = renderPieces($("#cut-bn"), "bn", m, bn, bnText);
  renderPieces($("#cut-en"), "en", m, en, enText);

  const words = (s) => s.trim().split(/\s+/).filter(Boolean).length;
  const per = (r, s) => (r && words(s) ? t("perWord", (r.count / words(s)).toFixed(1)) : "");
  const stats = [];
  if (bn) stats.push(t("cutTokens", "বাংলা", bn.count) + per(bn, bnText));
  if (en) stats.push(t("cutTokens", "English", en.count) + per(en, enText));
  if (bnPieces?.length) stats.push(`<span class="hl">${t("chopped", bnPieces.filter((p) => p.n > 1).length)}</span>`);
  $("#cut-stats").innerHTML = stats.length ? stats.map((s) => `<span>${s}</span>`).join("") : `<span>${esc(m.name)} · ${esc(m.org)}</span>`;

  state.inspect = null;
  renderInspect();
}

/* ---------- Render: piece inspector ---------- */
function renderInspect() {
  const box = $("#inspect");
  const sel = state.inspect && state.cutPieces[state.inspect.lang][state.inspect.i];
  document.querySelectorAll(".cut .pc.is-on").forEach((el) => el.classList.remove("is-on"));
  if (!sel) {
    box.innerHTML = `<p class="inspect__hint">${t("inspectHint")}</p>`;
    return;
  }
  $(`.cut .pc[data-lang="${state.inspect.lang}"][data-i="${state.inspect.i}"]`)?.classList.add("is-on");

  const m = byId[state.cutId];
  const cols = t("inspectCols");
  const chars = [...sel.text].map((ch) => {
    const cp = ch.codePointAt(0).toString(16).toUpperCase().padStart(4, "0");
    const bytes = [...utf8.encode(ch)].map((b) => b.toString(16).toUpperCase().padStart(2, "0")).join(" ");
    const shown = ch === " " ? `<span class="sp">${t("space")}</span>` : ch === "\n" ? "↵" : esc(ch);
    return `<tr><td class="inspect__ch">${shown}</td><td>U+${cp}</td><td>${bytes}</td></tr>`;
  }).join("");
  const nBytes = utf8.encode(sel.text).length;
  const note = sel.n > 1
    ? t("inspectMulti", nBytes, esc(m.name), sel.n, sel.ids.join(", "))
    : t("inspectOne", nBytes, esc(m.name), sel.ids[0]);

  box.innerHTML = `
    <div class="inspect__head">
      <span class="inspect__title">${t("inspectTitle")}</span>
      <button class="inspect__close" type="button" aria-label="${t("close")}">×</button>
    </div>
    <div class="inspect__body">
      <span class="inspect__piece">${pieceHtml(sel)}</span>
      <table class="inspect__table">
        <thead><tr><th>${cols[0]}</th><th>${cols[1]}</th><th>${cols[2]}</th></tr></thead>
        <tbody>${chars}</tbody>
      </table>
    </div>
    <p class="inspect__note">${note}</p>`;
}

/* ---------- Render: ranking + per-sentence heatmap ---------- */
function renderRanking() {
  const max = RANKING[0].r;
  $("#rank").innerHTML = RANKING.map(({ m, r }, i) => `
    <div class="rk">
      <span class="rk__pos">${String(i + 1).padStart(2, "0")}</span>
      <div class="rk__name">${esc(m.name)}<small>${esc(m.org)}</small></div>
      <div class="rk__track" aria-hidden="true">
        <div class="rk__fill" style="--w:${(r / max) * 100}%"></div>
        <div class="rk__par" style="--par:${(1 / max) * 100}%"><span>${t("fair")}</span></div>
      </div>
      <div class="rk__val">${times(r)}</div>
    </div>`).join("");
  $("#n-samples").textContent = SAMPLES.length;
}

// FLORES-200 study results (research/results/summary.json), loaded at start-up.
function renderFlores() {
  const box = $("#flores");
  if (!state.flores) { box.hidden = true; return; }
  box.hidden = false;
  const demo = Object.fromEntries(RANKING.map(({ m, r }) => [m.id, r]));
  const rows = state.flores.filter((r) => r.split === "devtest").sort((a, b) => b.parity - a.parity);
  const max = Math.ceil(Math.max(...rows.map((r) => r.ci_high)));
  const pct = (v) => (v / max) * 100;
  box.innerHTML = rows.map((r, i) => `
    <div class="rk">
      <span class="rk__pos">${String(i + 1).padStart(2, "0")}</span>
      <div class="rk__name">${esc(r.model)}<small>${esc(byId[r.id].org)}</small></div>
      <div class="rk__track" aria-hidden="true">
        <div class="rk__fill" style="--w:${pct(r.parity)}%"></div>
        <div class="rk__ci" style="--l:${pct(r.ci_low)}%; --r:${pct(r.ci_high)}%"></div>
        <div class="rk__par" style="--par:${pct(1)}%"><span>${t("fair")}</span></div>
      </div>
      <div class="rk__val">${times(r.parity)}<small>95% CI ${r.ci_low.toFixed(2)}–${r.ci_high.toFixed(2)}</small><small>${t("floresDemo", times(demo[r.id]))}</small></div>
    </div>`).join("");
}

function renderHeat() {
  const head = `<thead><tr><th></th>${SAMPLES.map((s, i) => `<th scope="col" title="${esc(s.bn)}">${i + 1}<small>${esc(topic(s))}</small></th>`).join("")}</tr></thead>`;
  const body = RANKING.map(({ m }) => {
    const cells = PRECOMPUTED[m.id].map((row) => {
      const r = sum(toPieces(row.bn)) / sum(toPieces(row.en));
      // Shade from paper (fair) to full red (6× or more); at or below parity, green.
      const cls = r <= 1.05 ? " heat__fair" : "";
      const h = Math.round(Math.min(1, Math.max(0, (r - 1) / 5)) * 85);
      return `<td class="heat__c${cls}${h > 50 ? " heat__dark" : ""}" style="--h:${h}%">${r.toFixed(1)}×</td>`;
    }).join("");
    return `<tr><th scope="row">${esc(m.name)}</th>${cells}</tr>`;
  }).join("");
  $("#heat").innerHTML = head + `<tbody>${body}</tbody>`;
}

/* ---------- Render: hero receipt ---------- */
function renderHero() {
  // Rounded down: the headline says "over N×".
  $("#hero-x").textContent = `${Math.floor(RANKING[0].r)}×`;

  // Receipt: the first sample, billed by the GPT-4 tokenizer.
  const m = byId["gpt-4"];
  const bn = result(m, "bn", SAMPLES[0].bn).count;
  const en = result(m, "en", SAMPLES[0].en).count;
  const bill = (100 * bn) / en;
  $("#rc-model").textContent = m.name.split(" ")[0];
  $("#rc-text").textContent = SAMPLES[0].bn;
  $("#rc-en-n").textContent = t("receiptTokens", en);
  $("#rc-bn-n").textContent = t("receiptTokens", bn);
  $("#rc-bn-t").textContent = taka(bill);
  $("#rc-tax").textContent = `+${taka(bill - 100)}`;
}

/* ---------- Render: 3 steps + try a word ---------- */
function renderSteps() {
  const en = toPieces(WORDS.en), bn = toPieces(WORDS.bn);
  const enN = sum(en), bnN = sum(bn);
  const word = (ps, n, cls = "") => `<span class="pieces ${cls}">${ps.map((p) => pieceHtml(p)).join("")}</span><span class="step__count">${t("tokens", n)}</span>`;
  $("#step-en").innerHTML = word(en, enN);
  $("#step-bn").innerHTML = word(bn, bnN, "bn");
  $("#step-bill").innerHTML = `
    <span class="step__line"><span>${esc(WORDS.en.map((p) => p[0]).join(""))}</span><b>${taka(enN)}</b></span>
    <span class="step__line bn"><span>${esc(WORDS.bn.map((p) => p[0]).join(""))}</span><b>${taka(bnN)}</b></span>
    <span class="step__x">${t("timesPrice", bnN)}</span>`;
  $("#step-note").textContent = t("stepNote", byId[WORDS.model].name);
}

function renderWord() {
  const out = $("#word-out");
  const text = $("#word").value.trim();
  const m = byId["gpt-4"];
  if (!text) { out.innerHTML = ""; return; }
  if (!m.tok) {
    out.innerHTML = m.error && !m.loading
      ? `<span class="try__msg">${t("wordFailed")}</span>`
      : `<span class="try__msg">${t("wordLoading", Math.round((m.p || 0) * 100))}</span>`;
    return;
  }
  const ps = piecesFromIds(m, m.tok.encode(text, { add_special_tokens: false }).ids, 80);
  out.innerHTML = `<span class="pieces bn">${ps.map((p) => pieceHtml(p)).join("")}</span>
    <span class="step__count">${t("wordStats", sum(ps), utf8.encode(text).length)}</span>`;
}

/* ---------- Render: check it yourself ---------- */
function renderVerify(bnText, enText) {
  const parts = ["gpt-4", "gpt-4o"].map((id) => {
    const m = byId[id];
    const bn = result(m, "bn", bnText), en = result(m, "en", enText);
    if (!bn || !en) return `<span class="na">${t("verifyLoad", esc(m.name))}</span>`;
    return `<b>${esc(m.name)}</b> বাংলা <b>${bn.count}</b> · English <b>${en.count}</b>`;
  });
  const el = $("#verify-nums");
  if (el) el.innerHTML = parts.join(" &nbsp;/&nbsp; ");
}

/* ---------- Render: tokenizer sources ---------- */
function renderSources() {
  const h = t("srcHead");
  $("#sources").innerHTML = `
    <div class="src__row src__head"><span>${h[0]}</span><span>${h[1]}</span><span>${h[2]}</span><span>${h[3]}</span></div>` +
    MODELS.map((m) => `
    <div class="src__row">
      <span class="src__name">${esc(m.name)}<small>${esc(m.org)}</small></span>
      <a href="https://huggingface.co/${m.repo}" target="_blank" rel="noopener">${esc(m.repo)}</a>
      <span>${m.official ? t("official") : t("community")}</span>
      <span>${VERIFIED.has(m.id) ? `<b class="ok">${t("matches")}</b>` : "—"}</span>
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
    btn.textContent = t("copied");
  } catch {
    btn.textContent = t("copyFail");
  }
  setTimeout(() => { btn.textContent = label; }, 1600);
}

/* ---------- Render loop ---------- */
const graphemes = (() => {
  try { const s = new Intl.Segmenter("bn", { granularity: "grapheme" }); return (x) => [...s.segment(x)].length; }
  catch { return (x) => [...x].length; }
})();

function renderAll() {
  const bnText = $("#in-bn").value;
  const enText = $("#in-en").value;
  const meta = (x) => (x ? t("meta", graphemes(x), x.trim().split(/\s+/).filter(Boolean).length) : "");
  $("#meta-bn").textContent = meta(bnText);
  $("#meta-en").textContent = meta(enText);

  const custom = isCustom(bnText, enText);
  renderStatus(custom);
  renderLedger(bnText, enText);
  renderVerify(bnText, enText);
  renderCut(bnText, enText);
  renderWord();
}

let frame = 0;
function scheduleRender() {
  cancelAnimationFrame(frame);
  frame = requestAnimationFrame(renderAll);
}
// Progress ticks only touch the status line, the load buttons and the word box.
let progressFrame = 0;
function scheduleProgress() {
  if (progressFrame) return;
  progressFrame = requestAnimationFrame(() => {
    progressFrame = 0;
    renderStatus(isCustom($("#in-bn").value, $("#in-en").value));
    document.querySelectorAll("[data-progress]").forEach((b) => {
      b.textContent = t("downloading", Math.round((byId[b.dataset.progress].p || 0) * 100));
    });
    if (byId["gpt-4"].loading) renderWord();
  });
}

/* ---------- Samples ---------- */
function buildSamples() {
  $("#samples").innerHTML = SAMPLES.map((s, i) =>
    `<button type="button" class="pill" data-i="${i}" aria-pressed="${i === state.sample}">${esc(topic(s))}</button>`
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
  captureEnglish();
  setupTheme();
  buildSamples();

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

  let wordTimer;
  $("#word").addEventListener("input", () => {
    if ($("#word").value.trim()) load(byId["gpt-4"]);
    clearTimeout(wordTimer);
    wordTimer = setTimeout(renderWord, 120);
  });

  $("#lang").addEventListener("click", () => setLanguage(state.lang === "bn" ? "en" : "bn"));
  $("#show-ids").addEventListener("change", (e) => $("#cut").classList.toggle("show-ids", e.target.checked));
  $("#dl-all").addEventListener("click", downloadAll);
  $("#dl-current").addEventListener("click", downloadCurrent);
  document.querySelectorAll("[data-copy]").forEach((b) =>
    b.addEventListener("click", () => copyText(b, $(`#in-${b.dataset.copy}`).value)));

  const pickPiece = (el) => {
    state.inspect = { lang: el.dataset.lang, i: Number(el.dataset.i) };
    renderInspect();
  };
  document.addEventListener("click", (e) => {
    const lb = e.target.closest("[data-load]");
    if (lb) load(byId[lb.dataset.load]);
    const tab = e.target.closest(".tab");
    if (tab) { state.cutId = tab.dataset.id; renderCut($("#in-bn").value, $("#in-en").value); }
    const pc = e.target.closest(".cut .pc[data-i]");
    if (pc) pickPiece(pc);
    if (e.target.closest(".inspect__close")) { state.inspect = null; renderInspect(); }
  });
  $("#cut").addEventListener("keydown", (e) => {
    const pc = e.target.closest(".pc[data-i]");
    if (pc && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); pickPiece(pc); }
  });

  state.sample = 0;
  $("#in-bn").value = SAMPLES[0].bn;
  $("#in-en").value = SAMPLES[0].en;
  let saved = null;
  try { saved = localStorage.getItem("tt-lang"); } catch {}
  state.lang = saved === "bn" ? "bn" : "en";
  applyLanguage();

  fetch("research/results/summary.json")
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
    .then((j) => { state.flores = j.results; renderFlores(); })
    .catch((err) => console.warn("FLORES results unavailable", err));
}

init();
