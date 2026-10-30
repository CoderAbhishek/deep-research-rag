# Comparison: change-a-v2.2

- Baseline runs: 20261030-113403_dev__rescored_v2.2, 20261030-133204_dev__rescored_v2.2
- Candidate runs: 20261030-154958_dev__rescored_v2.2, 20261030-162610_dev__rescored_v2.2
- Question set version(s): v2.2

## Metrics

Spread = max minus min across baseline runs (the noise). A change counts only if |delta| > spread. n = questions with a value in every compared run (quality metrics are computed on that same set).

| Metric | n | Baseline runs | Baseline mean | Spread | Candidate runs | Candidate mean | Delta | Verdict |
|---|---|---|---|---|---|---|---|---|
| Behaviour accuracy | 24 | 0.583 / 0.625 | 0.604 | 0.042 | 0.708 / 0.708 | 0.708 | 0.104 | better |
| Answer accuracy (answerable) | 21 | 0.524 / 0.571 | 0.548 | 0.048 | 0.667 / 0.667 | 0.667 | 0.119 | better |
| False refusal rate | 21 | 0.286 / 0.286 | 0.286 | 0.000 | 0.238 / 0.238 | 0.238 | -0.048 | better |
| Correct refusal rate | 2 | 1 / 1 | 1 | 0.000 | 1 / 1 | 1 | 0.000 | within noise |
| Hit@20 dense | 17 | 0.412 / 0.471 | 0.441 | 0.059 | 0.647 / 0.647 | 0.647 | 0.206 | better |
| Hit@20 BM25 | 17 | 0.588 / 0.588 | 0.588 | 0.000 | 0.588 / 0.588 | 0.588 | 0.000 | within noise |
| Hit@20 RRF | 17 | 0.706 / 0.765 | 0.735 | 0.059 | 0.882 / 0.882 | 0.882 | 0.147 | better |
| Hit@5 RRF | 17 | 0.471 / 0.588 | 0.529 | 0.118 | 0.588 / 0.588 | 0.588 | 0.059 | within noise |
| Hit@5 rerank | 17 | 0.588 / 0.529 | 0.559 | 0.059 | 0.647 / 0.647 | 0.647 | 0.088 | better |
| Hit@5 final | 22 | 0.591 / 0.591 | 0.591 | 0.000 | 0.682 / 0.727 | 0.705 | 0.114 | better |
| MRR@5 RRF | 17 | 0.174 / 0.289 | 0.231 | 0.116 | 0.362 / 0.272 | 0.317 | 0.085 | within noise |
| MRR@5 rerank | 17 | 0.414 / 0.377 | 0.396 | 0.036 | 0.461 / 0.416 | 0.438 | 0.043 | better |
| Citation precision | 8 | 0.562 / 0.688 | 0.625 | 0.125 | 0.688 / 0.750 | 0.719 | 0.094 | within noise |
| HyDE cut off (count) | - | 14 / 15 | 14.5 | 1 | 0.000 / 0.000 | 0.000 | -14.5 | better |
| Generation cut off (count) | - | 3 / 2 | 2.500 | 1 | 0.000 / 0.000 | 0.000 | -2.500 | better |
| HyDE fallback used (count) | - | 0.000 / 0.000 | 0.000 | 0.000 | 0.000 / 0.000 | 0.000 | 0.000 | within noise |
| Latency p50 (s) | - | 2.253 / 2.354 | 2.304 | 0.101 | 1.894 / 1.869 | 1.881 | -0.422 | better |
| Latency p95 (s) | - | 2.814 / 3.257 | 3.035 | 0.443 | 2.631 / 3.154 | 2.893 | -0.143 | within noise |
| Tokens per question | - | 1,831.8 / 1,843.5 | 1,837.6 | 11.7 | 1,472.9 / 1,463.2 | 1,468.1 | -369.6 | better |
| Cost per run (USD) | - | 0.007 / 0.007 | 0.007 | 0.000 | 0.004 / 0.004 | 0.004 | -0.003 | better |

## Per-question status

**partly_fixed**: 1, **still_fail**: 5, **still_pass**: 11, **unstable_baseline**: 7

| ID | Status | Baseline pass | Candidate pass | Baseline lost_at | Candidate lost_at | Question |
|---|---|---|---|---|---|---|
| q21 | partly_fixed | N N | N Y | unknown_stage, unknown_stage | fusion, ok | What was Infosys's return on equity in fiscal 2026? |
| q04 | unstable_baseline | N Y | N N | generation, ok | generation, generation | How much revenue did Infosys earn from North America in FY2026? |
| q09 | unstable_baseline | N Y | Y Y | not_retrieved, ok | ok, ok | What is Infosys's largest business segment by revenue? |
| q10 | unstable_baseline | Y N | Y Y | ok, generation | ok, ok | What was Infosys's adjusted operating margin for fiscal 2026? |
| q11 | unstable_baseline | Y N | Y N | ok, rerank | ok, rerank | What was Infosys's free cash flow in FY26? |
| q22 | unstable_baseline | N Y | Y Y | generation, ok | ok, ok | What was Infosys's basic earnings per share for FY2026? |
| q26 | unstable_baseline | N Y | Y Y | unknown_stage, ok | ok, ok | Who is the CEO of Infosys? |
| q36 | unstable_baseline | Y N | Y Y | ok, not_retrieved | ok, ok | What share of Infosys's revenue comes from exports? |
| q02 | still_fail | N N | N N | rerank, rerank | rerank, rerank | What was Infosys's revenue in US dollars for fiscal 2026? |
| q03 | still_fail | N N | N N | not_retrieved, rerank | rerank, rerank | What was Infosys's net profit for FY26? |
| q25 | still_fail | N N | N N | not_retrieved, not_retrieved | not_retrieved, not_retrieved | How does Infosys classify its services? |
| q31 | still_fail | N N | N N | rerank, rerank | rerank, rerank | How much did Infosys spend on CSR globally in fiscal 2026? |
| q32 | still_fail | N N | N N | not_retrieved, not_retrieved | not_retrieved, not_retrieved | Which Infosys CSR project focuses on lakes, and in which state is it c |
| q13 | still_pass | Y Y | Y Y | ok, ok | ok, ok | What total dividend per share did Infosys declare for fiscal 2026? |
| q14 | still_pass | Y Y | Y Y | ok, ok | ok, ok | How large was the share buyback Infosys completed, and at what price p |
| q15 | still_pass | Y Y | Y Y | ok, ok | ok, ok | What was Infosys's large deal TCV in fiscal 2026? |
| q17 | still_pass | Y Y | Y Y | ok, ok | ok, ok | How many US$50 million+ clients did Infosys have in fiscal 2026? |
| q18 | still_pass | Y Y | Y Y | ok, ok | ok, ok | How many employees did Infosys have at the end of fiscal 2026? |
| q19 | still_pass | Y Y | Y Y | ok, ok | ok, ok | How many people work at Infosys? |
| q23 | still_pass | Y Y | Y Y | ok, ok | ok, ok | What is Infosys Cobalt? |
| q28 | still_pass | Y Y | Y Y | ok, ok | ok, ok | Who was appointed Vice Chairman of Infosys, and from what date? |
| q38 | still_pass | Y Y | Y Y | n/a, n/a | n/a, n/a | What was TCS's revenue in fiscal 2026? |
| q39 | still_pass | Y Y | Y Y | fusion, fusion | rerank, generation | Why did Infosys's revenue fall by 20% in fiscal 2026? |
| q41 | still_pass | Y Y | Y Y | n/a, n/a | n/a, n/a | What is the capital of France? |

## Regression list check

- All regression-list questions passed in every candidate run.
