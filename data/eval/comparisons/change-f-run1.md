# Comparison: change-f-run1

- Baseline runs: 20261030-165023_dev, 20261030-170601_dev
- Candidate runs: 20261031-114854_dev
- Question set version(s): v2.2

## Metrics

n = questions with a value in every compared run (quality metrics are computed on that same set). Spread = max minus min of the baseline run averages. Flips = mean per-question change between repeat runs (for yes/no metrics, the share of questions that flipped). Noise floor = largest of spread, flips (baseline), flips (candidate) and 1/n. A change counts only if |delta| > noise floor. Summary metrics (latency, tokens, cost) use the spread only. Moves = questions better / worse in every candidate run than in every baseline run (in brackets: better / worse on average, runs overlap).

| Metric | n | Baseline runs | Baseline mean | Candidate runs | Candidate mean | Delta | Spread | Flips base / cand | Noise floor | Verdict | Moves |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Behaviour accuracy | 24 | 0.667 / 0.708 | 0.688 | 0.667 | 0.667 | -0.021 | 0.042 | 0.125 / - | 0.125 | within noise | +1 (+1) / -1 (-2) |
| Answer accuracy (answerable) | 21 | 0.619 / 0.667 | 0.643 | 0.619 | 0.619 | -0.024 | 0.048 | 0.143 / - | 0.143 | within noise | +1 (+1) / -1 (-2) |
| False refusal rate | 21 | 0.238 / 0.238 | 0.238 | 0.238 | 0.238 | 0.000 | 0.000 | 0.095 / - | 0.095 | within noise | +1 (+0) / -0 (-2) |
| Correct refusal rate | 2 | 1 / 1 | 1 | 1 | 1 | 0.000 | 0.000 | 0.000 / - | 0.500 | within noise | +0 (+0) / -0 (-0) |
| Hit@20 dense | 22 | 0.682 / 0.682 | 0.682 | 0.682 | 0.682 | 0.000 | 0.000 | 0.000 / - | 0.045 | within noise | +0 (+0) / -0 (-0) |
| Hit@20 BM25 | 22 | 0.636 / 0.636 | 0.636 | 0.636 | 0.636 | 0.000 | 0.000 | 0.000 / - | 0.045 | within noise | +0 (+0) / -0 (-0) |
| Hit@20 RRF | 22 | 0.864 / 0.909 | 0.886 | 0.864 | 0.864 | -0.023 | 0.045 | 0.045 / - | 0.045 | within noise | +0 (+1) / -1 (-0) |
| Hit@5 RRF | 22 | 0.682 / 0.727 | 0.705 | 0.591 | 0.591 | -0.114 | 0.045 | 0.136 / - | 0.136 | within noise | +0 (+1) / -2 (-2) |
| Hit@5 rerank | 22 | 0.773 / 0.727 | 0.750 | 0.636 | 0.636 | -0.114 | 0.045 | 0.045 / - | 0.045 | worse | +0 (+0) / -2 (-1) |
| Hit@5 final | 22 | 0.818 / 0.818 | 0.818 | 0.682 | 0.682 | -0.136 | 0.000 | 0.000 / - | 0.045 | worse | +0 (+0) / -3 (-0) |
| MRR@5 RRF | 22 | 0.330 / 0.378 | 0.354 | 0.361 | 0.361 | 0.007 | 0.048 | 0.078 / - | 0.078 | within noise | +2 (+1) / -2 (-4) |
| MRR@5 rerank | 22 | 0.602 / 0.515 | 0.559 | 0.496 | 0.496 | -0.063 | 0.087 | 0.087 / - | 0.087 | within noise | +0 (+2) / -2 (-2) |
| Citation precision | 15 | 0.500 / 0.567 | 0.533 | 0.500 | 0.500 | -0.033 | 0.067 | 0.133 / - | 0.133 | within noise | +2 (+1) / -2 (-1) |
| HyDE cut off (count) | - | 0.000 / 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | within noise | - |
| Generation cut off (count) | - | 0.000 / 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | within noise | - |
| HyDE fallback used (count) | - | 0.000 / 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | within noise | - |
| Questions with < 5 chunks to the model | - | 0.000 / 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | within noise | - |
| Latency p50 (s) | - | 2.381 / 2.309 | 2.345 | 12.2 | 12.2 | 9.843 | 0.072 | - | 0.072 | worse | - |
| Latency p95 (s) | - | 3.136 / 3.025 | 3.080 | 16.4 | 16.4 | 13.3 | 0.111 | - | 0.111 | worse | - |
| Tokens per question | - | 1,506.6 / 1,503.8 | 1,505.2 | 3,712.4 | 3,712.4 | 2,207.2 | 2.792 | - | 2.792 | worse | - |
| Cost per run (USD) | - | 0.004 / 0.004 | 0.004 | 0.008 | 0.008 | 0.004 | 0.000 | - | 0.000 | worse | - |
| Context chars per question | - | - / - | - | 11,047.2 | 11,047.2 | - | - | - | - | - | - |
| p50 HyDE stage (s) | - | - / - | - | 0.664 | 0.664 | - | - | - | - | - | - |
| p50 HyDE LLM call (s) | - | - / - | - | 0.659 | 0.659 | - | - | - | - | - | - |
| p50 dense (s) | - | - / - | - | 0.049 | 0.049 | - | - | - | - | - | - |
| p50 BM25 (s) | - | - / - | - | 0.013 | 0.013 | - | - | - | - | - | - |
| p50 rerank (s) | - | - / - | - | 0.592 | 0.592 | - | - | - | - | - | - |
| p50 dedup (s) | - | - / - | - | 0.001 | 0.001 | - | - | - | - | - | - |
| p50 page context (s) | - | - / - | - | 0.004 | 0.004 | - | - | - | - | - | - |
| p50 generation stage (s) | - | - / - | - | 10.8 | 10.8 | - | - | - | - | - | - |
| p50 generation LLM call (s) | - | - / - | - | 10.8 | 10.8 | - | - | - | - | - | - |
| p50 outside LLM calls (s) | - | - / - | - | 0.677 | 0.677 | - | - | - | - | - | - |

## Control metrics

Stages the change does not touch. Each should be within noise.
- Hit@20 dense (`dense_hit20`): delta 0.000, floor 0.045: within noise
- Hit@20 BM25 (`bm25_hit20`): delta 0.000, floor 0.045: within noise
- Hit@20 RRF (`rrf_hit20`): delta -0.023, floor 0.045: within noise
- Hit@5 rerank (`rerank_hit5`): delta -0.114, floor 0.045: **MOVED beyond the noise floor: distrust the noise estimate, or the claim that the change does not touch this stage**
- Hit@5 final (`final_hit5`): delta -0.136, floor 0.045: **MOVED beyond the noise floor: distrust the noise estimate, or the claim that the change does not touch this stage**

## Per-question status

**fixed**: 1, **regressed**: 1, **still_fail**: 5, **still_pass**: 14, **unstable_baseline**: 3

| ID | Status | Baseline pass | Candidate pass | Baseline lost_at | Candidate lost_at | Question |
|---|---|---|---|---|---|---|
| q18 | regressed | Y Y | N | ok, ok | generation | How many employees did Infosys have at the end of fiscal 2026? |
| q03 | fixed | N N | Y | rerank, rerank | ok_unlabelled_source | What was Infosys's net profit for FY26? |
| q11 | unstable_baseline | Y N | N | ok, generation | rerank | What was Infosys's free cash flow in FY26? |
| q21 | unstable_baseline | N Y | N | generation, ok | fusion | What was Infosys's return on equity in fiscal 2026? |
| q22 | unstable_baseline | N Y | Y | generation, ok | ok | What was Infosys's basic earnings per share for FY2026? |
| q02 | still_fail | N N | N | fusion, rerank | rerank | What was Infosys's revenue in US dollars for fiscal 2026? |
| q04 | still_fail | N N | N | generation, generation | generation | How much revenue did Infosys earn from North America in FY2026? |
| q25 | still_fail | N N | N | not_retrieved, not_retrieved | not_retrieved | How does Infosys classify its services? |
| q31 | still_fail | N N | N | rerank, rerank | rerank | How much did Infosys spend on CSR globally in fiscal 2026? |
| q32 | still_fail | N N | N | not_retrieved, not_retrieved | not_retrieved | Which Infosys CSR project focuses on lakes, and in which state is it c |
| q09 | still_pass | Y Y | Y | ok, ok | ok | What is Infosys's largest business segment by revenue? |
| q10 | still_pass | Y Y | Y | ok, ok | ok | What was Infosys's adjusted operating margin for fiscal 2026? |
| q13 | still_pass | Y Y | Y | ok, ok | ok | What total dividend per share did Infosys declare for fiscal 2026? |
| q14 | still_pass | Y Y | Y | ok, ok | ok | How large was the share buyback Infosys completed, and at what price p |
| q15 | still_pass | Y Y | Y | ok, ok | ok | What was Infosys's large deal TCV in fiscal 2026? |
| q17 | still_pass | Y Y | Y | ok, ok | ok | How many US$50 million+ clients did Infosys have in fiscal 2026? |
| q19 | still_pass | Y Y | Y | ok, ok | ok | How many people work at Infosys? |
| q23 | still_pass | Y Y | Y | ok, ok | ok | What is Infosys Cobalt? |
| q26 | still_pass | Y Y | Y | ok, ok | ok | Who is the CEO of Infosys? |
| q28 | still_pass | Y Y | Y | ok, ok | ok | Who was appointed Vice Chairman of Infosys, and from what date? |
| q36 | still_pass | Y Y | Y | ok, ok | ok | What share of Infosys's revenue comes from exports? |
| q38 | still_pass | Y Y | Y | n/a, n/a | n/a | What was TCS's revenue in fiscal 2026? |
| q39 | still_pass | Y Y | Y | generation, rerank | rerank | Why did Infosys's revenue fall by 20% in fiscal 2026? |
| q41 | still_pass | Y Y | Y | n/a, n/a | n/a | What is the capital of France? |

## Regression list check

- **q18 failed** in candidate runs: [False]
