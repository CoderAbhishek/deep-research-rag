# Comparison: regression-after-change-b

- Baseline runs: 20261030-154958_dev__rescored_v2.2, 20261030-162610_dev__rescored_v2.2, 20261030-165023_dev, 20261030-170601_dev
- Candidate runs: (none: baseline noise report)
- Question set version(s): v2.2

## Metrics

n = questions with a value in every compared run (quality metrics are computed on that same set). Spread = max minus min of the baseline run averages. Flips = mean per-question change between repeat runs (for yes/no metrics, the share of questions that flipped). Noise floor = largest of spread, flips (baseline), flips (candidate) and 1/n. A change counts only if |delta| > noise floor. Summary metrics (latency, tokens, cost) use the spread only. Moves = questions better / worse in every candidate run than in every baseline run (in brackets: better / worse on average, runs overlap).

| Metric | n | Baseline runs | Baseline mean | Candidate runs | Candidate mean | Delta | Spread | Flips base / cand | Noise floor | Verdict | Moves |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Behaviour accuracy | 24 | 0.708 / 0.708 / 0.667 / 0.708 | 0.698 | - | - | - | 0.042 | 0.125 / - | 0.125 | - | - |
| Answer accuracy (answerable) | 21 | 0.667 / 0.667 / 0.619 / 0.667 | 0.655 | - | - | - | 0.048 | 0.143 / - | 0.143 | - | - |
| False refusal rate | 21 | 0.238 / 0.238 / 0.238 / 0.238 | 0.238 | - | - | - | 0.000 | 0.095 / - | 0.095 | - | - |
| Correct refusal rate | 2 | 1 / 1 / 1 / 1 | 1 | - | - | - | 0.000 | 0.000 / - | 0.500 | - | - |
| Hit@20 dense | 22 | 0.727 / 0.727 / 0.682 / 0.682 | 0.705 | - | - | - | 0.045 | 0.091 / - | 0.091 | - | - |
| Hit@20 BM25 | 22 | 0.636 / 0.636 / 0.636 / 0.636 | 0.636 | - | - | - | 0.000 | 0.000 / - | 0.045 | - | - |
| Hit@20 RRF | 22 | 0.864 / 0.909 / 0.864 / 0.909 | 0.886 | - | - | - | 0.045 | 0.091 / - | 0.091 | - | - |
| Hit@5 RRF | 22 | 0.636 / 0.591 / 0.682 / 0.727 | 0.659 | - | - | - | 0.136 | 0.364 / - | 0.364 | - | - |
| Hit@5 rerank | 22 | 0.682 / 0.727 / 0.773 / 0.727 | 0.727 | - | - | - | 0.091 | 0.136 / - | 0.136 | - | - |
| Hit@5 final | 22 | 0.682 / 0.727 / 0.818 / 0.818 | 0.761 | - | - | - | 0.136 | 0.182 / - | 0.182 | - | - |
| MRR@5 RRF | 22 | 0.425 / 0.310 / 0.330 / 0.378 | 0.361 | - | - | - | 0.115 | 0.287 / - | 0.287 | - | - |
| MRR@5 rerank | 22 | 0.538 / 0.526 / 0.602 / 0.515 | 0.545 | - | - | - | 0.087 | 0.135 / - | 0.135 | - | - |
| Citation precision | 15 | 0.600 / 0.667 / 0.500 / 0.567 | 0.583 | - | - | - | 0.167 | 0.200 / - | 0.200 | - | - |
| HyDE cut off (count) | - | 0.000 / 0.000 / 0.000 / 0.000 | 0.000 | - | - | - | 0.000 | - | 0.000 | - | - |
| Generation cut off (count) | - | 0.000 / 0.000 / 0.000 / 0.000 | 0.000 | - | - | - | 0.000 | - | 0.000 | - | - |
| HyDE fallback used (count) | - | 0.000 / 0.000 / 0.000 / 0.000 | 0.000 | - | - | - | 0.000 | - | 0.000 | - | - |
| Questions with < 5 chunks to the model | - | 7 / 8 / 0.000 / 0.000 | 3.750 | - | - | - | 8 | - | 8 | - | - |
| Latency p50 (s) | - | 1.894 / 1.869 / 2.381 / 2.309 | 2.113 | - | - | - | 0.512 | - | 0.512 | - | - |
| Latency p95 (s) | - | 2.631 / 3.154 / 3.136 / 3.025 | 2.986 | - | - | - | 0.523 | - | 0.523 | - | - |
| Tokens per question | - | 1,472.9 / 1,463.2 / 1,506.6 / 1,503.8 | 1,486.6 | - | - | - | 43.3 | - | 43.3 | - | - |
| Cost per run (USD) | - | 0.004 / 0.004 / 0.004 / 0.004 | 0.004 | - | - | - | 0.000 | - | 0.000 | - | - |
| p50 HyDE stage (s) | - | - / - / - / - | - | - | - | - | - | - | - | - | - |
| p50 HyDE LLM call (s) | - | - / - / - / - | - | - | - | - | - | - | - | - | - |
| p50 dense (s) | - | - / - / - / - | - | - | - | - | - | - | - | - | - |
| p50 BM25 (s) | - | - / - / - / - | - | - | - | - | - | - | - | - | - |
| p50 rerank (s) | - | - / - / - / - | - | - | - | - | - | - | - | - | - |
| p50 dedup (s) | - | - / - / - / - | - | - | - | - | - | - | - | - | - |
| p50 generation stage (s) | - | - / - / - / - | - | - | - | - | - | - | - | - | - |
| p50 generation LLM call (s) | - | - / - / - / - | - | - | - | - | - | - | - | - | - |
| p50 outside LLM calls (s) | - | - / - / - / - | - | - | - | - | - | - | - | - | - |

## Per-question status

**stable_fail**: 6, **stable_pass**: 15, **unstable**: 3

| ID | Status | Baseline pass | Candidate pass | Baseline lost_at | Candidate lost_at | Question |
|---|---|---|---|---|---|---|
| q11 | unstable | Y N Y N | - | ok, rerank, ok, generation | - | What was Infosys's free cash flow in FY26? |
| q21 | unstable | N Y N Y | - | fusion, ok, generation, ok | - | What was Infosys's return on equity in fiscal 2026? |
| q22 | unstable | Y Y N Y | - | ok, ok, generation, ok | - | What was Infosys's basic earnings per share for FY2026? |
| q02 | stable_fail | N N N N | - | rerank, rerank, fusion, rerank | - | What was Infosys's revenue in US dollars for fiscal 2026? |
| q03 | stable_fail | N N N N | - | rerank, rerank, rerank, rerank | - | What was Infosys's net profit for FY26? |
| q04 | stable_fail | N N N N | - | generation, generation, generation, generation | - | How much revenue did Infosys earn from North America in FY2026? |
| q25 | stable_fail | N N N N | - | not_retrieved, not_retrieved, not_retrieved, not_retrieved | - | How does Infosys classify its services? |
| q31 | stable_fail | N N N N | - | rerank, rerank, rerank, rerank | - | How much did Infosys spend on CSR globally in fiscal 2026? |
| q32 | stable_fail | N N N N | - | not_retrieved, not_retrieved, not_retrieved, not_retrieved | - | Which Infosys CSR project focuses on lakes, and in which state is it c |
| q09 | stable_pass | Y Y Y Y | - | ok, ok, ok, ok | - | What is Infosys's largest business segment by revenue? |
| q10 | stable_pass | Y Y Y Y | - | ok, ok, ok, ok | - | What was Infosys's adjusted operating margin for fiscal 2026? |
| q13 | stable_pass | Y Y Y Y | - | ok, ok, ok, ok | - | What total dividend per share did Infosys declare for fiscal 2026? |
| q14 | stable_pass | Y Y Y Y | - | ok, ok, ok, ok | - | How large was the share buyback Infosys completed, and at what price p |
| q15 | stable_pass | Y Y Y Y | - | ok, ok, ok, ok | - | What was Infosys's large deal TCV in fiscal 2026? |
| q17 | stable_pass | Y Y Y Y | - | ok, ok, ok, ok | - | How many US$50 million+ clients did Infosys have in fiscal 2026? |
| q18 | stable_pass | Y Y Y Y | - | ok, ok, ok, ok | - | How many employees did Infosys have at the end of fiscal 2026? |
| q19 | stable_pass | Y Y Y Y | - | ok, ok, ok, ok | - | How many people work at Infosys? |
| q23 | stable_pass | Y Y Y Y | - | ok, ok, ok, ok | - | What is Infosys Cobalt? |
| q26 | stable_pass | Y Y Y Y | - | ok, ok, ok, ok | - | Who is the CEO of Infosys? |
| q28 | stable_pass | Y Y Y Y | - | ok, ok, ok, ok | - | Who was appointed Vice Chairman of Infosys, and from what date? |
| q36 | stable_pass | Y Y Y Y | - | ok, ok, ok, ok | - | What share of Infosys's revenue comes from exports? |
| q38 | stable_pass | Y Y Y Y | - | n/a, n/a, n/a, n/a | - | What was TCS's revenue in fiscal 2026? |
| q39 | stable_pass | Y Y Y Y | - | rerank, generation, generation, rerank | - | Why did Infosys's revenue fall by 20% in fiscal 2026? |
| q41 | stable_pass | Y Y Y Y | - | n/a, n/a, n/a, n/a | - | What is the capital of France? |
