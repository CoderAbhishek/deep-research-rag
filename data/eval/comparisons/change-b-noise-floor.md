# Comparison: change-b-noise-floor

- Baseline runs: 20261030-154958_dev__rescored_v2.2, 20261030-162610_dev__rescored_v2.2
- Candidate runs: 20261030-165023_dev, 20261030-170601_dev
- Question set version(s): v2.2

## Metrics

n = questions with a value in every compared run (quality metrics are computed on that same set). Spread = max minus min of the baseline run averages. Flips = mean per-question change between repeat runs (for yes/no metrics, the share of questions that flipped). Noise floor = largest of spread, flips (baseline), flips (candidate) and 1/n. A change counts only if |delta| > noise floor. Summary metrics (latency, tokens, cost) use the spread only. Moves = questions better / worse in every candidate run than in every baseline run (in brackets: better / worse on average, runs overlap).

| Metric | n | Baseline runs | Baseline mean | Candidate runs | Candidate mean | Delta | Spread | Flips base / cand | Noise floor | Verdict | Moves |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Behaviour accuracy | 24 | 0.708 / 0.708 | 0.708 | 0.667 / 0.708 | 0.688 | -0.021 | 0.000 | 0.083 / 0.125 | 0.125 | within noise | +0 (+0) / -0 (-1) |
| Answer accuracy (answerable) | 21 | 0.667 / 0.667 | 0.667 | 0.619 / 0.667 | 0.643 | -0.024 | 0.000 | 0.095 / 0.143 | 0.143 | within noise | +0 (+0) / -0 (-1) |
| False refusal rate | 21 | 0.238 / 0.238 | 0.238 | 0.238 / 0.238 | 0.238 | 0.000 | 0.000 | 0.095 / 0.095 | 0.095 | within noise | +0 (+0) / -0 (-0) |
| Correct refusal rate | 2 | 1 / 1 | 1 | 1 / 1 | 1 | 0.000 | 0.000 | 0.000 / 0.000 | 0.500 | within noise | +0 (+0) / -0 (-0) |
| Hit@20 dense | 22 | 0.727 / 0.727 | 0.727 | 0.682 / 0.682 | 0.682 | -0.045 | 0.000 | 0.091 / 0.000 | 0.091 | within noise | +0 (+0) / -0 (-2) |
| Hit@20 BM25 | 22 | 0.636 / 0.636 | 0.636 | 0.636 / 0.636 | 0.636 | 0.000 | 0.000 | 0.000 / 0.000 | 0.045 | within noise | +0 (+0) / -0 (-0) |
| Hit@20 RRF | 22 | 0.864 / 0.909 | 0.886 | 0.864 / 0.909 | 0.886 | 0.000 | 0.045 | 0.045 / 0.045 | 0.045 | within noise | +0 (+1) / -0 (-1) |
| Hit@5 RRF | 22 | 0.636 / 0.591 | 0.614 | 0.682 / 0.727 | 0.705 | 0.091 | 0.045 | 0.227 / 0.136 | 0.227 | within noise | +1 (+4) / -0 (-2) |
| Hit@5 rerank | 22 | 0.682 / 0.727 | 0.705 | 0.773 / 0.727 | 0.750 | 0.045 | 0.045 | 0.136 / 0.045 | 0.136 | within noise | +0 (+2) / -0 (-0) |
| Hit@5 final | 22 | 0.682 / 0.727 | 0.705 | 0.818 / 0.818 | 0.818 | 0.114 | 0.045 | 0.136 / 0.000 | 0.136 | within noise | +1 (+3) / -0 (-0) |
| MRR@5 RRF | 22 | 0.425 / 0.310 | 0.367 | 0.330 / 0.378 | 0.354 | -0.013 | 0.115 | 0.206 / 0.078 | 0.206 | within noise | +1 (+4) / -1 (-7) |
| MRR@5 rerank | 22 | 0.538 / 0.526 | 0.532 | 0.602 / 0.515 | 0.559 | 0.027 | 0.012 | 0.083 / 0.087 | 0.087 | within noise | +0 (+4) / -0 (-1) |
| Citation precision | 15 | 0.600 / 0.667 | 0.633 | 0.500 / 0.567 | 0.533 | -0.100 | 0.067 | 0.067 / 0.133 | 0.133 | within noise | +0 (+0) / -2 (-1) |
| HyDE cut off (count) | - | 0.000 / 0.000 | 0.000 | 0.000 / 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | within noise | - |
| Generation cut off (count) | - | 0.000 / 0.000 | 0.000 | 0.000 / 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | within noise | - |
| HyDE fallback used (count) | - | 0.000 / 0.000 | 0.000 | 0.000 / 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | within noise | - |
| Questions with < 5 chunks to the model | - | 7 / 8 | 7.500 | 0.000 / 0.000 | 0.000 | -7.500 | 1 | - | 1 | better | - |
| Latency p50 (s) | - | 1.894 / 1.869 | 1.881 | 2.381 / 2.309 | 2.345 | 0.463 | 0.025 | - | 0.025 | worse | - |
| Latency p95 (s) | - | 2.631 / 3.154 | 2.893 | 3.136 / 3.025 | 3.080 | 0.188 | 0.523 | - | 0.523 | within noise | - |
| Tokens per question | - | 1,472.9 / 1,463.2 | 1,468.1 | 1,506.6 / 1,503.8 | 1,505.2 | 37.1 | 9.625 | - | 9.625 | worse | - |
| Cost per run (USD) | - | 0.004 / 0.004 | 0.004 | 0.004 / 0.004 | 0.004 | 0.000 | 0.000 | - | 0.000 | worse | - |
| p50 HyDE stage (s) | - | - / - | - | - / - | - | - | - | - | - | - | - |
| p50 HyDE LLM call (s) | - | - / - | - | - / - | - | - | - | - | - | - | - |
| p50 dense (s) | - | - / - | - | - / - | - | - | - | - | - | - | - |
| p50 BM25 (s) | - | - / - | - | - / - | - | - | - | - | - | - | - |
| p50 rerank (s) | - | - / - | - | - / - | - | - | - | - | - | - | - |
| p50 dedup (s) | - | - / - | - | - / - | - | - | - | - | - | - | - |
| p50 generation stage (s) | - | - / - | - | - / - | - | - | - | - | - | - | - |
| p50 generation LLM call (s) | - | - / - | - | - / - | - | - | - | - | - | - | - |
| p50 outside LLM calls (s) | - | - / - | - | - / - | - | - | - | - | - | - | - |

## Control metrics

Stages the change does not touch. Each should be within noise.
- Hit@20 dense (`dense_hit20`): delta -0.045, floor 0.091: within noise
- Hit@20 BM25 (`bm25_hit20`): delta 0.000, floor 0.045: within noise

## Per-question status

**partly_regressed**: 1, **still_fail**: 6, **still_pass**: 15, **unstable_baseline**: 2

| ID | Status | Baseline pass | Candidate pass | Baseline lost_at | Candidate lost_at | Question |
|---|---|---|---|---|---|---|
| q22 | partly_regressed | Y Y | N Y | ok, ok | generation, ok | What was Infosys's basic earnings per share for FY2026? |
| q11 | unstable_baseline | Y N | Y N | ok, rerank | ok, generation | What was Infosys's free cash flow in FY26? |
| q21 | unstable_baseline | N Y | N Y | fusion, ok | generation, ok | What was Infosys's return on equity in fiscal 2026? |
| q02 | still_fail | N N | N N | rerank, rerank | fusion, rerank | What was Infosys's revenue in US dollars for fiscal 2026? |
| q03 | still_fail | N N | N N | rerank, rerank | rerank, rerank | What was Infosys's net profit for FY26? |
| q04 | still_fail | N N | N N | generation, generation | generation, generation | How much revenue did Infosys earn from North America in FY2026? |
| q25 | still_fail | N N | N N | not_retrieved, not_retrieved | not_retrieved, not_retrieved | How does Infosys classify its services? |
| q31 | still_fail | N N | N N | rerank, rerank | rerank, rerank | How much did Infosys spend on CSR globally in fiscal 2026? |
| q32 | still_fail | N N | N N | not_retrieved, not_retrieved | not_retrieved, not_retrieved | Which Infosys CSR project focuses on lakes, and in which state is it c |
| q09 | still_pass | Y Y | Y Y | ok, ok | ok, ok | What is Infosys's largest business segment by revenue? |
| q10 | still_pass | Y Y | Y Y | ok, ok | ok, ok | What was Infosys's adjusted operating margin for fiscal 2026? |
| q13 | still_pass | Y Y | Y Y | ok, ok | ok, ok | What total dividend per share did Infosys declare for fiscal 2026? |
| q14 | still_pass | Y Y | Y Y | ok, ok | ok, ok | How large was the share buyback Infosys completed, and at what price p |
| q15 | still_pass | Y Y | Y Y | ok, ok | ok, ok | What was Infosys's large deal TCV in fiscal 2026? |
| q17 | still_pass | Y Y | Y Y | ok, ok | ok, ok | How many US$50 million+ clients did Infosys have in fiscal 2026? |
| q18 | still_pass | Y Y | Y Y | ok, ok | ok, ok | How many employees did Infosys have at the end of fiscal 2026? |
| q19 | still_pass | Y Y | Y Y | ok, ok | ok, ok | How many people work at Infosys? |
| q23 | still_pass | Y Y | Y Y | ok, ok | ok, ok | What is Infosys Cobalt? |
| q26 | still_pass | Y Y | Y Y | ok, ok | ok, ok | Who is the CEO of Infosys? |
| q28 | still_pass | Y Y | Y Y | ok, ok | ok, ok | Who was appointed Vice Chairman of Infosys, and from what date? |
| q36 | still_pass | Y Y | Y Y | ok, ok | ok, ok | What share of Infosys's revenue comes from exports? |
| q38 | still_pass | Y Y | Y Y | n/a, n/a | n/a, n/a | What was TCS's revenue in fiscal 2026? |
| q39 | still_pass | Y Y | Y Y | rerank, generation | generation, rerank | Why did Infosys's revenue fall by 20% in fiscal 2026? |
| q41 | still_pass | Y Y | Y Y | n/a, n/a | n/a, n/a | What is the capital of France? |

## Regression list check

- **q22 failed** in candidate runs: [False, True]
