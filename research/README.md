# The Bangla Token Tax: Tokenization Disparity Across 8 LLM Tokenizers on FLORES-200

**Lamia Islam Mim** · Dhaka, Bangladesh · October 2026
*Technical report, not peer-reviewed. Code and data in this folder.*

---

## Abstract

Large language models process text as tokens and most commercial APIs bill per token. We measure how many tokens eight widely used tokenizers need for Bangla compared with English when the content is the same, using the 1,012 professionally translated sentence pairs of the FLORES-200 devtest set. English-centric tokenizers need **4.9 to 5.8 times** as many tokens for Bangla (GPT-4/GPT-3.5: **5.80×**, 95% CI 5.73–5.86), splitting the average Bangla word into about 8 tokens. Newer tokenizers with larger vocabularies narrow the gap (GPT-4o: **1.69×**, DeepSeek-V3: 2.01×, Gemma 2: 2.68×), and BLOOM, whose training data deliberately included Bangla, comes closest to parity (**1.17×**). The effect is consistent: for six of the eight tokenizers, Bangla needs more tokens in at least 99.5% of sentence pairs. Results replicate on the separate FLORES-200 dev set, are unaffected by a data-cleaning step, and the GPT counts match OpenAI's official encodings in 8,036 of 8,036 cases.

![Bangla tokens per English token on FLORES-200 devtest, with 95% confidence intervals](results/parity_devtest.svg)

## 1. Question

If a Bangla speaker and an English speaker send the same message to an AI model, how many more tokens does the Bangla message use? Because tokens drive API price, generation time and how much text fits in a model's context window, a large and consistent gap would mean Bangla speakers pay a hidden "token tax".

## 2. Data

- **FLORES-200** ([NLLB Team, 2022](https://arxiv.org/abs/2207.04672); built on FLORES-101, [Goyal et al., 2022](https://arxiv.org/abs/2106.03193)): the same English sentences professionally translated into 200 languages, aligned line by line. Released by Meta under CC-BY-SA 4.0.
- **Primary set:** `devtest`, `ben_Beng` and `eng_Latn`, **1,012** sentence pairs.
- **Replication set:** `dev`, **997** sentence pairs.
- **Cleaning:** 67 Bangla devtest lines (65 in dev) carry CSV-style quoting, for example `"""আমাদের ...,"" তিনি যোগ করলেন।"`. English lines do not. We undo this quoting (strip the outer quotes, turn `""` into `"`) so both languages are compared on plain text. Section 5.3 shows this does not change the results.

The FLORES text is not redistributed here; the per-sentence results refer to sentences by their line number.

## 3. Tokenizers

Each tokenizer is loaded from the Hugging Face Hub at a fixed commit, so the numbers can be reproduced exactly.

| Model | Tokenizer source | Revision | Source type | Cross-checked |
|---|---|---|---|---|
| GPT-4 / GPT-3.5 | `Xenova/gpt-4` | `1d9f1f1` | Community copy of OpenAI `cl100k_base` | ✓ 4,018 / 4,018 |
| GPT-4o | `Xenova/gpt-4o` | `7956d98` | Community copy of OpenAI `o200k_base` | ✓ 4,018 / 4,018 |
| Llama 3 | `Xenova/llama3-tokenizer-new` | `566fd99` | Community copy | — |
| Gemma 2 | `unsloth/gemma-2-2b` | `2531994` | Community copy | — |
| Qwen 3 | `Qwen/Qwen3-0.6B` | `c1899de` | Official | — |
| DeepSeek-V3 | `deepseek-ai/DeepSeek-V3` | `e815299` | Official | — |
| Mistral 7B (v0.3) | `Xenova/mistral-tokenizer-v3` | `b251f94` | Community copy | — |
| BLOOM | `Xenova/bloom-560m` | `115f924` | Community copy | — |

Full commit hashes are in [`results/summary.json`](results/summary.json). **Cross-check:** every GPT-4 and GPT-4o count (both languages, both splits) was recomputed with OpenAI's published encodings via `js-tiktoken` 1.0.21. All 8,036 counts matched.

## 4. Method

- **Token count:** tokens produced for each sentence, without special start or end tokens (`add_special_tokens: false`), using `@huggingface/tokenizers` 0.2.0.
- **Parity ratio (primary metric):** total Bangla tokens ÷ total English tokens over all sentence pairs. 1.0 means equal treatment. This follows the "tokenization premium" framing of [Petrov et al. (2023)](https://arxiv.org/abs/2305.15425).
- **Uncertainty:** 95% confidence interval from a bootstrap over sentence pairs (10,000 resamples, fixed seed 20261018).
- **Spread:** median and interquartile range of the per-sentence ratio, and the share of pairs where Bangla needs more tokens.
- **Fertility:** tokens per whitespace-separated word, and Bangla tokens per character (grapheme cluster). A value above 1 token per character means characters are being split.

## 5. Results

### 5.1 Main result (devtest, n = 1,012)

| Model | Parity | 95% CI | Median (IQR) per sentence | Pairs where Bangla is longer | Tokens / word, Bangla | Tokens / word, English | Bangla tokens / character |
|---|---|---|---|---|---|---|---|
| GPT-4 / 3.5 | **5.80×** | 5.73–5.86 | 5.80 (5.13–6.52) | 100.0% | 8.08 | 1.24 | 1.94 |
| Llama 3 | 5.75× | 5.69–5.82 | 5.76 (5.09–6.48) | 100.0% | 8.02 | 1.24 | 1.92 |
| Qwen 3 | 5.02× | 4.96–5.08 | 5.04 (4.43–5.71) | 100.0% | 7.12 | 1.26 | 1.70 |
| Mistral 7B | 4.90× | 4.84–4.96 | 4.89 (4.30–5.56) | 100.0% | 7.54 | 1.37 | 1.81 |
| Gemma 2 | 2.68× | 2.65–2.71 | 2.68 (2.41–3.00) | 100.0% | 3.72 | 1.23 | 0.89 |
| DeepSeek-V3 | 2.01× | 1.99–2.03 | 2.00 (1.80–2.25) | 99.9% | 2.79 | 1.23 | 0.67 |
| GPT-4o | 1.69× | 1.68–1.71 | 1.69 (1.52–1.88) | 99.5% | 2.33 | 1.23 | 0.56 |
| BLOOM | **1.17×** | 1.16–1.18 | 1.17 (1.05–1.29) | 81.0% | 1.65 | 1.25 | 0.39 |

### 5.2 Replication (dev, n = 997)

| Model | devtest | dev |
|---|---|---|
| GPT-4 / 3.5 | 5.80× | 5.83× |
| Llama 3 | 5.75× | 5.79× |
| Qwen 3 | 5.02× | 5.05× |
| Mistral 7B | 4.90× | 4.93× |
| Gemma 2 | 2.68× | 2.69× |
| DeepSeek-V3 | 2.01× | 2.02× |
| GPT-4o | 1.69× | 1.71× |
| BLOOM | 1.17× | 1.18× |

Every model's dev ratio lies within 0.04 of its devtest ratio, and the ranking is identical.

### 5.3 Sensitivity to cleaning

Without undoing the CSV quoting, ratios change by at most 0.005 (GPT-4: 5.799× raw vs 5.795× cleaned). See `parity_raw_text` in [`results/summary.csv`](results/summary.csv).

### 5.4 Comparison with the website's demo sentences

The [Token Tax website](https://lamia-mim.github.io/token-tax/) uses 8 short hand-written sentence pairs. They produce the **same ranking** of the 8 models but slightly lower ratios (for example GPT-4: 5.22× demo vs 5.80× FLORES; BLOOM: 1.00× vs 1.17×). The demo understates the gap rather than exaggerating it.

## 6. Findings

1. **English-centric tokenizers impose a large, systematic premium.** GPT-4/3.5, Llama 3, Qwen 3 and Mistral need about 5 to 6 times as many tokens for Bangla. GPT-4 averages 8 tokens per Bangla word and almost 2 tokens per Bangla character, meaning single letters are regularly split into raw bytes. Llama 3's vocabulary extends OpenAI's 100K tiktoken vocabulary ([Llama Team, 2024](https://arxiv.org/abs/2407.21783)), which is consistent with its numbers being close to GPT-4's.
2. **Larger, more multilingual vocabularies close much of the gap.** GPT-4o's tokenizer cuts the premium from 5.80× to 1.69×. DeepSeek-V3 (2.01×) and Gemma 2 (2.68×) also do far better than the English-centric group.
3. **Training data matters most.** BLOOM, whose tokenizer was trained on data that deliberately included Bangla and other Indic languages, is closest to parity (1.17×) and is the only tokenizer where Bangla is *not* longer for a fifth of the sentences.
4. **The effect is not driven by a few sentences.** For six of eight tokenizers, Bangla needs more tokens in ≥ 99.5% of pairs, and the confidence intervals are narrow (about ±1%).

**What this implies (not measured here).** With per-token pricing, the input cost of the same Bangla content is roughly the parity ratio times the English cost. Because models generate one token at a time and have token-based context limits, Bangla output can also be expected to be slower and Bangla documents to fill the context window faster. These consequences follow from the counts but were not measured in this study.

## 7. Limitations

- **Tokens, not price.** Real cost also depends on output length, per-model prices and caching. The parity ratio describes input text only.
- **Domain.** FLORES-200 sentences come from English Wikimedia sources (news, travel, science) and are formal; conversational or social-media Bangla may behave differently.
- **Translation direction.** All Bangla sentences are translations from English, which can differ from text originally written in Bangla.
- **Word counts** use whitespace splitting for both languages.
- **Tokenizer copies.** Six tokenizers come from community copies or official repos on Hugging Face; only the two GPT tokenizers were cross-checked against the company's own encoding.
- **Tokenization only.** This study does not measure how well models understand or generate Bangla.

## 8. Reproduce

```bash
cd research
npm install
curl -LO https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
tar -xzf flores200_dataset.tar.gz
node run.mjs
```

The script downloads each tokenizer at its pinned revision (about 80 MB, cached in `.cache/`) and writes:

| File | Contents |
|---|---|
| [`results/summary.csv`](results/summary.csv) | One row per split × model with every metric above |
| [`results/summary.json`](results/summary.json) | Same results plus metadata: revisions, library versions, seed, cleaning counts, cross-check |
| [`results/flores_devtest_counts.csv`](results/flores_devtest_counts.csv) | Per sentence × model: Bangla and English tokens, words and characters |
| [`results/flores_dev_counts.csv`](results/flores_dev_counts.csv) | Same for the dev split |
| [`results/parity_devtest.svg`](results/parity_devtest.svg) | The chart above |

Re-running produces identical numbers (the bootstrap uses a fixed seed).

## 9. References

- Ahia, O., Kumar, S., Gonen, H., et al. (2023). *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models.* EMNLP. [arXiv:2305.13707](https://arxiv.org/abs/2305.13707)
- Goyal, N., Gao, C., Chaudhary, V., et al. (2022). *The Flores-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation.* TACL. [arXiv:2106.03193](https://arxiv.org/abs/2106.03193)
- Llama Team, AI @ Meta (2024). *The Llama 3 Herd of Models.* [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- NLLB Team (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation.* [arXiv:2207.04672](https://arxiv.org/abs/2207.04672)
- Petrov, A., La Malfa, E., Torr, P. H. S., & Bibi, A. (2023). *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS. [arXiv:2305.15425](https://arxiv.org/abs/2305.15425)
- Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL. [arXiv:2012.15613](https://arxiv.org/abs/2012.15613)

## Cite

```
Lamia Islam Mim (2026). The Bangla Token Tax: Tokenization Disparity Across 8 LLM Tokenizers on FLORES-200.
Technical report. https://github.com/Lamia-Mim/token-tax/tree/main/research
```
