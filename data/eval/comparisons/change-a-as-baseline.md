# Comparison: change-a-as-baseline

- Baseline runs: 20261030-154958_dev__rescored_v2.2, 20261030-162610_dev__rescored_v2.2
- Candidate runs: (none: baseline noise report)
- Question set version(s): v2.2

## Metrics

Spread = max minus min across baseline runs (the noise). A change counts only if |delta| > spread. n = questions with a value in every compared run (quality metrics are computed on that same set).

| Metric | n | Baseline runs | Baseline mean | Spread | Candidate runs | Candidate mean | Delta | Verdict |
|---|---|---|---|---|---|---|---|---|
| Behaviour accuracy | 24 | 0.708 / 0.708 | 0.708 | 0.000 | - | - | - | - |
| Answer accuracy (answerable) | 21 | 0.667 / 0.667 | 0.667 | 0.000 | - | - | - | - |
| False refusal rate | 21 | 0.238 / 0.238 | 0.238 | 0.000 | - | - | - | - |
| Correct refusal rate | 2 | 1 / 1 | 1 | 0.000 | - | - | - | - |
| Hit@20 dense | 22 | 0.727 / 0.727 | 0.727 | 0.000 | - | - | - | - |
| Hit@20 BM25 | 22 | 0.636 / 0.636 | 0.636 | 0.000 | - | - | - | - |
| Hit@20 RRF | 22 | 0.864 / 0.909 | 0.886 | 0.045 | - | - | - | - |
| Hit@5 RRF | 22 | 0.636 / 0.591 | 0.614 | 0.045 | - | - | - | - |
| Hit@5 rerank | 22 | 0.682 / 0.727 | 0.705 | 0.045 | - | - | - | - |
| Hit@5 final | 22 | 0.682 / 0.727 | 0.705 | 0.045 | - | - | - | - |
| MRR@5 RRF | 22 | 0.425 / 0.310 | 0.367 | 0.115 | - | - | - | - |
| MRR@5 rerank | 22 | 0.538 / 0.526 | 0.532 | 0.012 | - | - | - | - |
| Citation precision | 15 | 0.600 / 0.667 | 0.633 | 0.067 | - | - | - | - |
| HyDE cut off (count) | - | 0.000 / 0.000 | 0.000 | 0.000 | - | - | - | - |
| Generation cut off (count) | - | 0.000 / 0.000 | 0.000 | 0.000 | - | - | - | - |
| HyDE fallback used (count) | - | 0.000 / 0.000 | 0.000 | 0.000 | - | - | - | - |
| Latency p50 (s) | - | 1.894 / 1.869 | 1.881 | 0.025 | - | - | - | - |
| Latency p95 (s) | - | 2.631 / 3.154 | 2.893 | 0.523 | - | - | - | - |
| Tokens per question | - | 1,472.9 / 1,463.2 | 1,468.1 | 9.625 | - | - | - | - |
| Cost per run (USD) | - | 0.004 / 0.004 | 0.004 | 0.000 | - | - | - | - |

## Per-question status

**stable_fail**: 6, **stable_pass**: 16, **unstable**: 2

| ID | Status | Baseline pass | Candidate pass | Baseline lost_at | Candidate lost_at | Question |
|---|---|---|---|---|---|---|
| q11 | unstable | Y N | - | ok, rerank | - | What was Infosys's free cash flow in FY26? |
| q21 | unstable | N Y | - | fusion, ok | - | What was Infosys's return on equity in fiscal 2026? |
| q02 | stable_fail | N N | - | rerank, rerank | - | What was Infosys's revenue in US dollars for fiscal 2026? |
| q03 | stable_fail | N N | - | rerank, rerank | - | What was Infosys's net profit for FY26? |
| q04 | stable_fail | N N | - | generation, generation | - | How much revenue did Infosys earn from North America in FY2026? |
| q25 | stable_fail | N N | - | not_retrieved, not_retrieved | - | How does Infosys classify its services? |
| q31 | stable_fail | N N | - | rerank, rerank | - | How much did Infosys spend on CSR globally in fiscal 2026? |
| q32 | stable_fail | N N | - | not_retrieved, not_retrieved | - | Which Infosys CSR project focuses on lakes, and in which state is it c |
| q09 | stable_pass | Y Y | - | ok, ok | - | What is Infosys's largest business segment by revenue? |
| q10 | stable_pass | Y Y | - | ok, ok | - | What was Infosys's adjusted operating margin for fiscal 2026? |
| q13 | stable_pass | Y Y | - | ok, ok | - | What total dividend per share did Infosys declare for fiscal 2026? |
| q14 | stable_pass | Y Y | - | ok, ok | - | How large was the share buyback Infosys completed, and at what price p |
| q15 | stable_pass | Y Y | - | ok, ok | - | What was Infosys's large deal TCV in fiscal 2026? |
| q17 | stable_pass | Y Y | - | ok, ok | - | How many US$50 million+ clients did Infosys have in fiscal 2026? |
| q18 | stable_pass | Y Y | - | ok, ok | - | How many employees did Infosys have at the end of fiscal 2026? |
| q19 | stable_pass | Y Y | - | ok, ok | - | How many people work at Infosys? |
| q22 | stable_pass | Y Y | - | ok, ok | - | What was Infosys's basic earnings per share for FY2026? |
| q23 | stable_pass | Y Y | - | ok, ok | - | What is Infosys Cobalt? |
| q26 | stable_pass | Y Y | - | ok, ok | - | Who is the CEO of Infosys? |
| q28 | stable_pass | Y Y | - | ok, ok | - | Who was appointed Vice Chairman of Infosys, and from what date? |
| q36 | stable_pass | Y Y | - | ok, ok | - | What share of Infosys's revenue comes from exports? |
| q38 | stable_pass | Y Y | - | n/a, n/a | - | What was TCS's revenue in fiscal 2026? |
| q39 | stable_pass | Y Y | - | rerank, generation | - | Why did Infosys's revenue fall by 20% in fiscal 2026? |
| q41 | stable_pass | Y Y | - | n/a, n/a | - | What is the capital of France? |
