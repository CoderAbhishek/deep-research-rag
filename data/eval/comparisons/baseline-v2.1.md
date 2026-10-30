# Comparison: baseline-v2.1

- Baseline runs: 20261030-113403_dev__rescored_v2.1, 20261030-133204_dev__rescored_v2.1
- Candidate runs: (none: baseline noise report)
- Question set version(s): v2.1

## Metrics

Spread = max minus min across baseline runs (the noise). A change counts only if |delta| > spread. n = questions with a value in every compared run (quality metrics are computed on that same set).

| Metric | n | Baseline runs | Baseline mean | Spread | Candidate runs | Candidate mean | Delta | Verdict |
|---|---|---|---|---|---|---|---|---|
| Behaviour accuracy | 24 | 0.542 / 0.583 | 0.562 | 0.042 | - | - | - | - |
| Answer accuracy (answerable) | 21 | 0.476 / 0.524 | 0.500 | 0.048 | - | - | - | - |
| False refusal rate | 21 | 0.286 / 0.286 | 0.286 | 0.000 | - | - | - | - |
| Correct refusal rate | 2 | 1 / 1 | 1 | 0.000 | - | - | - | - |
| Hit@20 dense | 18 | 0.389 / 0.444 | 0.417 | 0.056 | - | - | - | - |
| Hit@20 BM25 | 18 | 0.556 / 0.556 | 0.556 | 0.000 | - | - | - | - |
| Hit@20 RRF | 18 | 0.667 / 0.722 | 0.694 | 0.056 | - | - | - | - |
| Hit@5 RRF | 18 | 0.444 / 0.556 | 0.500 | 0.111 | - | - | - | - |
| Hit@5 rerank | 18 | 0.556 / 0.500 | 0.528 | 0.056 | - | - | - | - |
| Hit@5 final | 22 | 0.591 / 0.591 | 0.591 | 0.000 | - | - | - | - |
| MRR@5 RRF | 18 | 0.164 / 0.273 | 0.219 | 0.109 | - | - | - | - |
| MRR@5 rerank | 18 | 0.391 / 0.356 | 0.374 | 0.034 | - | - | - | - |
| Citation precision | 9 | 0.500 / 0.611 | 0.556 | 0.111 | - | - | - | - |
| HyDE cut off (count) | - | 14 / 15 | 14.5 | 1 | - | - | - | - |
| Generation cut off (count) | - | 3 / 2 | 2.500 | 1 | - | - | - | - |
| Latency p50 (s) | - | 2.253 / 2.354 | 2.304 | 0.101 | - | - | - | - |
| Latency p95 (s) | - | 2.814 / 3.257 | 3.035 | 0.443 | - | - | - | - |
| Tokens per question | - | 1,831.8 / 1,843.5 | 1,837.6 | 11.7 | - | - | - | - |
| Cost per run (USD) | - | 0.007 / 0.007 | 0.007 | 0.000 | - | - | - | - |

## Per-question status

**stable_fail**: 7, **stable_pass**: 10, **unstable**: 7

| ID | Status | Baseline pass | Candidate pass | Baseline lost_at | Candidate lost_at | Question |
|---|---|---|---|---|---|---|
| q04 | unstable | N Y | - | generation, ok | - | How much revenue did Infosys earn from North America in FY2026? |
| q09 | unstable | N Y | - | not_retrieved, ok | - | What is Infosys's largest business segment by revenue? |
| q10 | unstable | Y N | - | ok, generation | - | What was Infosys's adjusted operating margin for fiscal 2026? |
| q11 | unstable | Y N | - | ok, rerank | - | What was Infosys's free cash flow in FY26? |
| q22 | unstable | N Y | - | generation, ok | - | What was Infosys's basic earnings per share for FY2026? |
| q26 | unstable | N Y | - | unknown_stage, ok | - | Who is the CEO of Infosys? |
| q36 | unstable | Y N | - | ok, not_retrieved | - | What share of Infosys's revenue comes from exports? |
| q02 | stable_fail | N N | - | rerank, rerank | - | What was Infosys's revenue in US dollars for fiscal 2026? |
| q03 | stable_fail | N N | - | not_retrieved, rerank | - | What was Infosys's net profit for FY26? |
| q19 | stable_fail | N N | - | generation, generation | - | How many people work at Infosys? |
| q21 | stable_fail | N N | - | not_retrieved, not_retrieved | - | What was Infosys's return on equity in fiscal 2026? |
| q25 | stable_fail | N N | - | not_retrieved, not_retrieved | - | How does Infosys classify its services? |
| q31 | stable_fail | N N | - | rerank, rerank | - | How much did Infosys spend on CSR globally in fiscal 2026? |
| q32 | stable_fail | N N | - | not_retrieved, not_retrieved | - | Which Infosys CSR project focuses on lakes, and in which state is it c |
| q13 | stable_pass | Y Y | - | ok, ok | - | What total dividend per share did Infosys declare for fiscal 2026? |
| q14 | stable_pass | Y Y | - | ok, ok | - | How large was the share buyback Infosys completed, and at what price p |
| q15 | stable_pass | Y Y | - | ok, ok | - | What was Infosys's large deal TCV in fiscal 2026? |
| q17 | stable_pass | Y Y | - | ok, ok | - | How many US$50 million+ clients did Infosys have in fiscal 2026? |
| q18 | stable_pass | Y Y | - | ok, ok | - | How many employees did Infosys have at the end of fiscal 2026? |
| q23 | stable_pass | Y Y | - | ok, ok | - | What is Infosys Cobalt? |
| q28 | stable_pass | Y Y | - | ok, ok | - | Who was appointed Vice Chairman of Infosys, and from what date? |
| q38 | stable_pass | Y Y | - | n/a, n/a | - | What was TCS's revenue in fiscal 2026? |
| q39 | stable_pass | Y Y | - | fusion, fusion | - | Why did Infosys's revenue fall by 20% in fiscal 2026? |
| q41 | stable_pass | Y Y | - | n/a, n/a | - | What is the capital of France? |
