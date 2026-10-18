// Token Tax study: how many more tokens does Bangla need than English for the
// same meaning? Uses the FLORES-200 benchmark (professionally translated,
// sentence-aligned) and 8 tokenizers pinned to exact Hugging Face revisions.
//
//   cd research
//   npm install
//   curl -LO https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
//   tar -xzf flores200_dataset.tar.gz
//   node run.mjs [path/to/flores200_dataset]
//
// Writes results/ : per-sentence counts (CSV), summary (CSV + JSON), chart (SVG).
import { Tokenizer } from "@huggingface/tokenizers";
import { getEncoding } from "js-tiktoken";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FLORES = path.resolve(process.argv[2] || path.join(HERE, "flores200_dataset"));
const OUT = path.join(HERE, "results");
const CACHE = path.join(HERE, ".cache");

// Tokenizers, pinned so the numbers can be reproduced exactly.
// tiktoken = OpenAI's own encoding, used to cross-check the Hugging Face copy.
const MODELS = [
  { id: "gpt-4",    name: "GPT-4 / 3.5", repo: "Xenova/gpt-4",                rev: "1d9f1f1b1fae88c0e4df1dab0a397f8de6229075", tiktoken: "cl100k_base" },
  { id: "gpt-4o",   name: "GPT-4o",      repo: "Xenova/gpt-4o",               rev: "7956d98f2a83b2751a98ea7136fdf7fe6cf54e69", tiktoken: "o200k_base" },
  { id: "llama3",   name: "Llama 3",     repo: "Xenova/llama3-tokenizer-new", rev: "566fd99122febe940eaa6518324e934fb0a06e5b" },
  { id: "gemma2",   name: "Gemma 2",     repo: "unsloth/gemma-2-2b",          rev: "25319945f7fd83b8b903e12081777b7eef2ba993" },
  { id: "qwen3",    name: "Qwen 3",      repo: "Qwen/Qwen3-0.6B",             rev: "c1899de289a04d12100db370d81485cdf75e47ca" },
  { id: "deepseek", name: "DeepSeek-V3", repo: "deepseek-ai/DeepSeek-V3",     rev: "e815299b0bcbac849fa540c768ef21845365c9eb" },
  { id: "mistral",  name: "Mistral 7B",  repo: "Xenova/mistral-tokenizer-v3", rev: "b251f946a5b6b16c36963ea8848611bce67dfbf1" },
  { id: "bloom",    name: "BLOOM",       repo: "Xenova/bloom-560m",           rev: "115f9242bc5d2dc3b6b3b78e53277e028cb90d65" },
];
const SPLITS = [
  { id: "devtest", file: (lang) => `devtest/${lang}.devtest` }, // primary
  { id: "dev",     file: (lang) => `dev/${lang}.dev` },         // replication
];
const BOOTSTRAP = 10000;
const SEED = 20261018;

/* ---------- helpers ---------- */
async function hubJson(m, file) {
  const p = path.join(CACHE, `${m.repo.replace("/", "__")}@${m.rev}__${file}`);
  if (!fs.existsSync(p)) {
    const res = await fetch(`https://huggingface.co/${m.repo}/resolve/${m.rev}/${file}`);
    if (!res.ok) throw new Error(`${res.status} for ${m.repo}/${file}`);
    fs.mkdirSync(CACHE, { recursive: true });
    fs.writeFileSync(p, await res.text());
  }
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

const readLines = (file) => fs.readFileSync(path.join(FLORES, file), "utf8").split("\n").filter((l, i, a) => i < a.length - 1 || l);

// Some FLORES-200 Bangla lines carry CSV-style quoting ("""text,"" more").
// Undo it so both languages are compared on the same plain text.
const isCsvQuoted = (l) => l.length > 1 && l.startsWith('"') && l.endsWith('"') && l.includes('""');
const unquote = (l) => (isCsvQuoted(l) ? l.slice(1, -1).replaceAll('""', '"') : l);

const seg = new Intl.Segmenter("bn", { granularity: "grapheme" });
const chars = (s) => [...seg.segment(s)].length;
const words = (s) => s.trim().split(/\s+/).filter(Boolean).length;
const sumOf = (a) => a.reduce((x, y) => x + y, 0);
const quantile = (sorted, q) => {
  const i = (sorted.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
};

// Deterministic RNG (mulberry32) so the confidence intervals are reproducible.
function rng(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// 95% bootstrap CI for the corpus-level ratio sum(bn) / sum(en), resampling sentences.
function bootstrapCI(bn, en, rand) {
  const n = bn.length, stats = new Float64Array(BOOTSTRAP);
  for (let b = 0; b < BOOTSTRAP; b++) {
    let sb = 0, se = 0;
    for (let k = 0; k < n; k++) { const j = (rand() * n) | 0; sb += bn[j]; se += en[j]; }
    stats[b] = sb / se;
  }
  stats.sort();
  return [quantile(stats, 0.025), quantile(stats, 0.975)];
}

const csv = (rows) => rows.map((r) => r.map((v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v)).join(",")).join("\n") + "\n";

/* ---------- run ---------- */
fs.mkdirSync(OUT, { recursive: true });
const tokenizers = {};
for (const m of MODELS) {
  tokenizers[m.id] = new Tokenizer(await hubJson(m, "tokenizer.json"), await hubJson(m, "tokenizer_config.json"));
  console.log(`loaded ${m.name}`);
}
const official = Object.fromEntries(MODELS.filter((m) => m.tiktoken).map((m) => [m.id, getEncoding(m.tiktoken)]));
const count = (id, text) => tokenizers[id].encode(text, { add_special_tokens: false }).ids.length;

const summary = [];
const meta = { generated: new Date().toISOString(), seed: SEED, bootstrap: BOOTSTRAP, splits: {}, crossCheck: {} };

for (const split of SPLITS) {
  const rawBn = readLines(split.file("ben_Beng"));
  const rawEn = readLines(split.file("eng_Latn"));
  if (rawBn.length !== rawEn.length) throw new Error(`${split.id}: line counts differ`);
  const bn = rawBn.map(unquote), en = rawEn.map(unquote);
  const n = bn.length;
  meta.splits[split.id] = { sentences: n, bnLinesUnquoted: rawBn.filter(isCsvQuoted).length, enLinesUnquoted: rawEn.filter(isCsvQuoted).length };
  console.log(`\n${split.id}: ${n} sentence pairs (${meta.splits[split.id].bnLinesUnquoted} Bangla lines unquoted)`);

  const rows = [["id", "model", "bangla_tokens", "english_tokens", "bangla_words", "english_words", "bangla_chars", "english_chars"]];
  const bnW = bn.map(words), enW = en.map(words), bnC = bn.map(chars), enC = en.map(chars);

  for (const m of MODELS) {
    const b = bn.map((s) => count(m.id, s));
    const e = en.map((s) => count(m.id, s));
    b.forEach((_, i) => rows.push([i + 1, m.id, b[i], e[i], bnW[i], enW[i], bnC[i], enC[i]]));

    if (official[m.id]) {
      const enc = official[m.id];
      let match = 0;
      for (let i = 0; i < n; i++) {
        if (enc.encode(bn[i]).length === b[i]) match++;
        if (enc.encode(en[i]).length === e[i]) match++;
      }
      meta.crossCheck[`${split.id}/${m.id}`] = { encoding: m.tiktoken, compared: 2 * n, matched: match };
    }

    const ratios = b.map((x, i) => x / e[i]).sort((x, y) => x - y);
    const [lo, hi] = bootstrapCI(b, e, rng(SEED));
    const rawParity = sumOf(rawBn.map((s) => count(m.id, s))) / sumOf(rawEn.map((s) => count(m.id, s)));
    const r = {
      split: split.id, model: m.name, id: m.id, sentences: n,
      bangla_tokens: sumOf(b), english_tokens: sumOf(e),
      parity: sumOf(b) / sumOf(e), ci_low: lo, ci_high: hi,
      median_ratio: quantile(ratios, 0.5), p25_ratio: quantile(ratios, 0.25), p75_ratio: quantile(ratios, 0.75),
      share_bangla_longer: b.filter((x, i) => x > e[i]).length / n,
      bangla_tokens_per_word: sumOf(b) / sumOf(bnW), english_tokens_per_word: sumOf(e) / sumOf(enW),
      bangla_tokens_per_char: sumOf(b) / sumOf(bnC),
      parity_raw_text: rawParity,
    };
    summary.push(r);
    console.log(`  ${m.name.padEnd(12)} parity ${r.parity.toFixed(3)}  95% CI [${lo.toFixed(3)}, ${hi.toFixed(3)}]  median ${r.median_ratio.toFixed(2)}  raw ${rawParity.toFixed(3)}`);
  }
  fs.writeFileSync(path.join(OUT, `flores_${split.id}_counts.csv`), csv(rows));
}

/* ---------- write summary ---------- */
const cols = Object.keys(summary[0]).filter((k) => k !== "id");
const fmt = (v) => (typeof v === "number" && !Number.isInteger(v) ? v.toFixed(4) : v);
fs.writeFileSync(path.join(OUT, "summary.csv"), csv([cols, ...summary.map((r) => cols.map((c) => fmt(r[c])))]));
meta.models = MODELS.map(({ id, name, repo, rev, tiktoken }) => ({ id, name, repo, revision: rev, officialEncoding: tiktoken || null }));
meta.libraries = {
  "@huggingface/tokenizers": JSON.parse(fs.readFileSync(path.join(HERE, "node_modules/@huggingface/tokenizers/package.json"), "utf8")).version,
  "js-tiktoken": JSON.parse(fs.readFileSync(path.join(HERE, "node_modules/js-tiktoken/package.json"), "utf8")).version,
  node: process.version,
};
fs.writeFileSync(path.join(OUT, "summary.json"), JSON.stringify({ meta, results: summary }, null, 2) + "\n");
console.log("\ncross-check:", JSON.stringify(meta.crossCheck));

/* ---------- chart: devtest parity with 95% CI ---------- */
const rows = summary.filter((r) => r.split === "devtest").sort((a, b) => b.parity - a.parity);
const W = 760, rowH = 40, top = 64, left = 150, right = 90, H = top + rows.length * rowH + 48;
const max = Math.ceil(Math.max(...rows.map((r) => r.ci_high)));
const x = (v) => left + (v / max) * (W - left - right);
const bars = rows.map((r, i) => {
  const y = top + i * rowH;
  const fill = r.parity < 2 ? "#0b6e4f" : "#d42a20";
  return `<text x="${left - 12}" y="${y + 20}" text-anchor="end" class="l">${r.model}</text>
  <rect x="${left}" y="${y + 8}" width="${x(r.parity) - left}" height="18" fill="${fill}"/>
  <line x1="${x(r.ci_low)}" x2="${x(r.ci_high)}" y1="${y + 17}" y2="${y + 17}" stroke="#16130f" stroke-width="2"/>
  <line x1="${x(r.ci_low)}" x2="${x(r.ci_low)}" y1="${y + 11}" y2="${y + 23}" stroke="#16130f" stroke-width="2"/>
  <line x1="${x(r.ci_high)}" x2="${x(r.ci_high)}" y1="${y + 11}" y2="${y + 23}" stroke="#16130f" stroke-width="2"/>
  <text x="${W - right + 10}" y="${y + 21}" class="v">${r.parity.toFixed(2)}×</text>`;
}).join("\n");
const ticks = Array.from({ length: max + 1 }, (_, v) => `<line x1="${x(v)}" x2="${x(v)}" y1="${top}" y2="${H - 40}" stroke="#16130f" stroke-opacity="${v === 1 ? 0.7 : 0.12}" ${v === 1 ? 'stroke-dasharray="4 3"' : ""}/>
  <text x="${x(v)}" y="${H - 22}" text-anchor="middle" class="t">${v}×</text>`).join("\n");
const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">
<style>.h{font-size:16px;font-weight:700}.s{font-size:12px;fill:#4a443c}.l{font-size:13px;font-weight:600}.v{font-size:13px;font-weight:700}.t{font-size:11px;fill:#4a443c}</style>
<rect width="100%" height="100%" fill="#fbf8f2"/>
<text x="20" y="28" class="h">Bangla tokens per English token, FLORES-200 devtest (n = ${rows[0].sentences})</text>
<text x="20" y="48" class="s">Bars: corpus ratio. Black lines: 95% bootstrap confidence interval. Dashed line: parity (1×).</text>
${ticks}
${bars}
</svg>
`;
fs.writeFileSync(path.join(OUT, "parity_devtest.svg"), svg);
console.log(`wrote ${OUT}`);
