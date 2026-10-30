# Comparison: change-a

- Baseline runs: 20261030-113403_dev__rescored_v2.1, 20261030-133204_dev__rescored_v2.1
- Candidate runs: 20261030-154958_dev, 20261030-162610_dev
- Question set version(s): v2.1

## Metrics

Spread = max minus min across baseline runs (the noise). A change counts only if |delta| > spread. n = questions with a value in every compared run (quality metrics are computed on that same set).

| Metric | n | Baseline runs | Baseline mean | Spread | Candidate runs | Candidate mean | Delta | Verdict |
|---|---|---|---|---|---|---|---|---|
| Behaviour accuracy | 24 | 0.542 / 0.583 | 0.562 | 0.042 | 0.625 / 0.667 | 0.646 | 0.083 | better |
| Answer accuracy (answerable) | 21 | 0.476 / 0.524 | 0.500 | 0.048 | 0.571 / 0.619 | 0.595 | 0.095 | better |
| False refusal rate | 21 | 0.286 / 0.286 | 0.286 | 0.000 | 0.238 / 0.238 | 0.238 | -0.048 | better |
| Correct refusal rate | 2 | 1 / 1 | 1 | 0.000 | 1 / 1 | 1 | 0.000 | within noise |
| Hit@20 dense | 18 | 0.389 / 0.444 | 0.417 | 0.056 | 0.667 / 0.667 | 0.667 | 0.250 | better |
| Hit@20 BM25 | 18 | 0.556 / 0.556 | 0.556 | 0.000 | 0.556 / 0.556 | 0.556 | 0.000 | within noise |
| Hit@20 RRF | 18 | 0.667 / 0.722 | 0.694 | 0.056 | 0.833 / 0.889 | 0.861 | 0.167 | better |
| Hit@5 RRF | 18 | 0.444 / 0.556 | 0.500 | 0.111 | 0.556 / 0.556 | 0.556 | 0.056 | within noise |
| Hit@5 rerank | 18 | 0.556 / 0.500 | 0.528 | 0.056 | 0.611 / 0.611 | 0.611 | 0.083 | better |
| Hit@5 final | 22 | 0.591 / 0.591 | 0.591 | 0.000 | 0.682 / 0.636 | 0.659 | 0.068 | better |
| MRR@5 RRF | 18 | 0.164 / 0.273 | 0.219 | 0.109 | 0.342 / 0.256 | 0.299 | 0.081 | within noise |
| MRR@5 rerank | 18 | 0.391 / 0.356 | 0.374 | 0.034 | 0.435 / 0.393 | 0.414 | 0.040 | better |
| Citation precision | 8 | 0.562 / 0.688 | 0.625 | 0.125 | 0.688 / 0.750 | 0.719 | 0.094 | within noise |
| HyDE cut off (count) | - | 14 / 15 | 14.5 | 1 | 0.000 / 0.000 | 0.000 | -14.5 | better |
| Generation cut off (count) | - | 3 / 2 | 2.500 | 1 | 0.000 / 0.000 | 0.000 | -2.500 | better |
| HyDE fallback used (count) | - | - / - | - | - | 0.000 / 0.000 | 0.000 | - | - |
| Latency p50 (s) | - | 2.253 / 2.354 | 2.304 | 0.101 | 1.894 / 1.869 | 1.881 | -0.422 | better |
| Latency p95 (s) | - | 2.814 / 3.257 | 3.035 | 0.443 | 2.631 / 3.154 | 2.893 | -0.143 | within noise |
| Tokens per question | - | 1,831.8 / 1,843.5 | 1,837.6 | 11.7 | 1,472.9 / 1,463.2 | 1,468.1 | -369.6 | better |
| Cost per run (USD) | - | 0.007 / 0.007 | 0.007 | 0.000 | 0.004 / 0.004 | 0.004 | -0.003 | better |

## Per-question status

**partly_fixed**: 2, **partly_regressed**: 1, **still_fail**: 5, **still_pass**: 9, **unstable_baseline**: 7

| ID | Status | Baseline pass | Candidate pass | Baseline lost_at | Candidate lost_at | Question |
|---|---|---|---|---|---|---|
| q28 | partly_regressed | Y Y | N Y | ok, ok | generation, ok | Who was appointed Vice Chairman of Infosys, and from what date? |
| q19 | partly_fixed | N N | Y N | generation, generation | ok, generation | How many people work at Infosys? |
| q21 | partly_fixed | N N | N Y | not_retrieved, not_retrieved | fusion, ok_unlabelled_source | What was Infosys's return on equity in fiscal 2026? |
| q04 | unstable_baseline | N Y | N N | generation, ok | generation, generation | How much revenue did Infosys earn from North America in FY2026? |
| q09 | unstable_baseline | N Y | Y Y | not_retrieved, ok | ok, ok | What is Infosys's largest business segment by revenue? |
| q10 | unstable_baseline | Y N | Y Y | ok, generation | ok, ok | What was Infosys's adjusted operating margin for fiscal 2026? |
| q11 | unstable_baseline | Y N | Y N | ok, rerank | ok, rerank | What was Infosys's free cash flow in FY26? |
| q22 | unstable_baseline | N Y | N Y | generation, ok | generation, ok | What was Infosys's basic earnings per share for FY2026? |
| q26 | unstable_baseline | N Y | Y Y | unknown_stage, ok | ok, ok_unlabelled_source | Who is the CEO of Infosys? |
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
| q23 | still_pass | Y Y | Y Y | ok, ok | ok, ok | What is Infosys Cobalt? |
| q38 | still_pass | Y Y | Y Y | n/a, n/a | n/a, n/a | What was TCS's revenue in fiscal 2026? |
| q39 | still_pass | Y Y | Y Y | fusion, fusion | rerank, generation | Why did Infosys's revenue fall by 20% in fiscal 2026? |
| q41 | still_pass | Y Y | Y Y | n/a, n/a | n/a, n/a | What is the capital of France? |

## Regression list check

- **q28 failed** in candidate runs: [False, True]
