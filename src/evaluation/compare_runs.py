"""
compare_runs.py - compare a candidate version of the system with a baseline,
question by question, against the measured run-to-run noise.

Why: averages hide churn. Between two identical baseline runs, 7 of 24
questions flipped pass/fail while the headline moved by one question. A change
is judged by (1) which questions it fixed or broke, and (2) whether metric
changes are larger than the baseline's own run-to-run spread.

Usage (from the project root):
  Baseline only (noise report + regression list):
    python -m src.evaluation.compare_runs --baseline RUN1 RUN2 --name baseline \
        --write-regression data/eval/regression_ids.json

  Baseline vs a change:
    python -m src.evaluation.compare_runs --baseline RUN1 RUN2 --candidate RUN3 RUN4 \
        --name change-a --regression data/eval/regression_ids.json

RUN is a run folder (data/eval/runs/<run_id> or a __rescored_<version> folder).
All runs being compared should use the same question set version.

Output: a Markdown report printed and saved to data/eval/comparisons/<name>.md,
plus <name>.json with the same data.

Per-question status (behaviour_ok across runs)
----------------------------------------------
Baseline only:   stable_pass, stable_fail, unstable
With candidate:  fixed            failed in every baseline run, passed in every candidate run
                 partly_fixed     failed in every baseline run, passed in some candidate runs
                 regressed        passed in every baseline run, failed in every candidate run
                 partly_regressed passed in every baseline run, failed in some candidate runs
                 still_pass / still_fail / unstable_baseline
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Quality metrics are recomputed from the rows, over the questions that have a
# value in EVERY run being compared. This keeps baseline and candidate on the
# same questions (re-scored baselines have unknown stage hits for relabelled
# questions). (label, row column, row filter, higher_is_better)
ROW_METRICS = [
    ("Behaviour accuracy", "behaviour_ok", None, True),
    ("Answer accuracy (answerable)", "answer_ok", "answer", True),
    ("False refusal rate", "refused", "answer", False),
    ("Correct refusal rate", "refused", "refuse", True),
    ("Hit@20 dense", "dense_hit20", None, True),
    ("Hit@20 BM25", "bm25_hit20", None, True),
    ("Hit@20 RRF", "rrf_hit20", None, True),
    ("Hit@5 RRF", "rrf_hit5", None, True),
    ("Hit@5 rerank", "rerank_hit5", None, True),
    ("Hit@5 final", "final_hit5", None, True),
    ("MRR@5 RRF", "rrf_rr5", None, True),
    ("MRR@5 rerank", "rerank_rr5", None, True),
    ("Citation precision", "citation_precision", None, True),
]
# Operational metrics come from each run's summary. (label, path, higher_is_better)
SUMMARY_METRICS = [
    ("HyDE cut off (count)", ("hyde_cut_off",), False),
    ("Generation cut off (count)", ("generation_cut_off",), False),
    ("Latency p50 (s)", ("latency_s", "p50"), False),
    ("Latency p95 (s)", ("latency_s", "p95"), False),
    ("Tokens per question", ("tokens_mean",), False),
    ("Cost per run (USD)", ("cost_usd_total",), False),
]


def _value(row: Dict[str, str], col: str) -> Optional[float]:
    v = (row.get(col) or "").strip()
    if v.upper() == "TRUE":
        return 1.0
    if v.upper() == "FALSE":
        return 0.0
    try:
        return float(v)
    except ValueError:
        return None


def load_run(run_dir: str) -> Dict[str, Any]:
    path = Path(run_dir)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    with open(path / "results.csv", encoding="utf-8-sig") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    return {"name": path.name, "summary": summary, "rows": rows}


def _get(d: Dict[str, Any], keys: tuple) -> Optional[float]:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d if isinstance(d, (int, float)) else None


def _passes(run: Dict[str, Any], qid: str) -> Optional[bool]:
    row = run["rows"].get(qid)
    if row is None:
        return None
    return row["behaviour_ok"].strip().upper() == "TRUE"


def classify(base: List[Optional[bool]], cand: List[Optional[bool]]) -> str:
    base = [b for b in base if b is not None]
    cand = [c for c in cand if c is not None]
    if not cand:
        if all(base):
            return "stable_pass"
        if not any(base):
            return "stable_fail"
        return "unstable"
    if all(base):
        if all(cand):
            return "still_pass"
        return "regressed" if not any(cand) else "partly_regressed"
    if not any(base):
        if not any(cand):
            return "still_fail"
        return "fixed" if all(cand) else "partly_fixed"
    return "unstable_baseline"


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "-"
    if float(v).is_integer() and abs(v) >= 1:
        return f"{int(v):,}"
    return f"{v:.3f}" if abs(v) < 10 else f"{v:,.1f}"


def _verdict(delta: Optional[float], spread: Optional[float], higher_better: bool) -> str:
    if delta is None or spread is None:
        return "-"
    if abs(delta) <= spread:
        return "within noise"
    return "better" if (delta > 0) == higher_better else "worse"


def _row_metric(runs: List[Dict[str, Any]], col: str, expected: Optional[str]):
    """Per-run means over the questions that have a value in every run."""
    ids = None
    for r in runs:
        have = {qid for qid, row in r["rows"].items()
                if (expected is None or row.get("expected_behavior") == expected) and _value(row, col) is not None}
        ids = have if ids is None else ids & have
    ids = sorted(ids or [])
    means = [sum(_value(r["rows"][q], col) for q in ids) / len(ids) if ids else None for r in runs]
    return means, len(ids)


def _metric_entry(label, bv, cv, higher_better, n):
    bv_ok = [v for v in bv if v is not None]
    cv_ok = [v for v in cv if v is not None]
    b_mean = sum(bv_ok) / len(bv_ok) if bv_ok else None
    c_mean = sum(cv_ok) / len(cv_ok) if cv_ok else None
    spread = (max(bv_ok) - min(bv_ok)) if len(bv_ok) > 1 else None
    delta = (c_mean - b_mean) if (b_mean is not None and c_mean is not None) else None
    return {"metric": label, "n_questions": n, "baseline": bv, "baseline_mean": b_mean, "baseline_spread": spread,
            "candidate": cv, "candidate_mean": c_mean, "delta": delta,
            "verdict": _verdict(delta, spread, higher_better)}


def compare(baseline: List[Dict[str, Any]], candidate: List[Dict[str, Any]],
            regression_ids: Optional[List[str]]) -> Dict[str, Any]:
    versions = {r["summary"]["config"].get("question_set_version", "unknown") for r in baseline + candidate}
    qids = sorted(set().union(*(r["rows"].keys() for r in baseline + candidate)))

    per_question = []
    for qid in qids:
        b = [_passes(r, qid) for r in baseline]
        c = [_passes(r, qid) for r in candidate]
        any_row = next(r["rows"][qid] for r in baseline + candidate if qid in r["rows"])
        per_question.append({
            "id": qid, "question": any_row["question"], "status": classify(b, c),
            "baseline": b, "candidate": c,
            "baseline_lost_at": [r["rows"].get(qid, {}).get("lost_at", "") for r in baseline],
            "candidate_lost_at": [r["rows"].get(qid, {}).get("lost_at", "") for r in candidate],
        })

    metrics = []
    runs = baseline + candidate
    nb = len(baseline)
    for label, col, expected, higher_better in ROW_METRICS:
        means, n = _row_metric(runs, col, expected)
        metrics.append(_metric_entry(label, means[:nb], means[nb:], higher_better, n))
    for label, keys, higher_better in SUMMARY_METRICS:
        bv = [_get(r["summary"]["overall"], keys) for r in baseline]
        cv = [_get(r["summary"]["overall"], keys) for r in candidate]
        metrics.append(_metric_entry(label, bv, cv, higher_better, None))

    regression_failures = []
    if regression_ids and candidate:
        for qid in regression_ids:
            c = [_passes(r, qid) for r in candidate]
            if any(v is False for v in c):
                regression_failures.append({"id": qid, "candidate": c})

    return {"baseline_runs": [r["name"] for r in baseline], "candidate_runs": [r["name"] for r in candidate],
            "question_set_versions": sorted(versions), "per_question": per_question,
            "metrics": metrics, "regression_failures": regression_failures}


def to_markdown(name: str, res: Dict[str, Any]) -> str:
    lines = [f"# Comparison: {name}", "",
             f"- Baseline runs: {', '.join(res['baseline_runs'])}",
             f"- Candidate runs: {', '.join(res['candidate_runs']) or '(none: baseline noise report)'}",
             f"- Question set version(s): {', '.join(res['question_set_versions'])}"]
    if len(res["question_set_versions"]) > 1:
        lines.append("- **WARNING: runs use different question set versions; re-score them first.**")
    lines += ["", "## Metrics", "",
              "Spread = max minus min across baseline runs (the noise). A change counts only if |delta| > spread. "
              "n = questions with a value in every compared run (quality metrics are computed on that same set).", "",
              "| Metric | n | Baseline runs | Baseline mean | Spread | Candidate runs | Candidate mean | Delta | Verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
    for mt in res["metrics"]:
        n = "-" if mt["n_questions"] is None else str(mt["n_questions"])
        lines.append(f"| {mt['metric']} | {n} | {' / '.join(_fmt(v) for v in mt['baseline'])} | {_fmt(mt['baseline_mean'])} | "
                     f"{_fmt(mt['baseline_spread'])} | {' / '.join(_fmt(v) for v in mt['candidate']) or '-'} | "
                     f"{_fmt(mt['candidate_mean'])} | {_fmt(mt['delta'])} | {mt['verdict']} |")
    counts: Dict[str, int] = {}
    for q in res["per_question"]:
        counts[q["status"]] = counts.get(q["status"], 0) + 1
    lines += ["", "## Per-question status", "", ", ".join(f"**{k}**: {v}" for k, v in sorted(counts.items())), "",
              "| ID | Status | Baseline pass | Candidate pass | Baseline lost_at | Candidate lost_at | Question |",
              "|---|---|---|---|---|---|---|"]
    order = ["regressed", "partly_regressed", "fixed", "partly_fixed", "unstable_baseline", "unstable",
             "still_fail", "stable_fail", "still_pass", "stable_pass"]
    yn = lambda v: "-" if v is None else ("Y" if v else "N")
    for q in sorted(res["per_question"], key=lambda q: (order.index(q["status"]), q["id"])):
        lines.append(f"| {q['id']} | {q['status']} | {' '.join(yn(v) for v in q['baseline'])} | "
                     f"{' '.join(yn(v) for v in q['candidate']) or '-'} | {', '.join(q['baseline_lost_at'])} | "
                     f"{', '.join(q['candidate_lost_at']) or '-'} | {q['question'][:70]} |")
    if res["candidate_runs"]:
        lines += ["", "## Regression list check", ""]
        if res["regression_failures"]:
            for f in res["regression_failures"]:
                lines.append(f"- **{f['id']} failed** in candidate runs: {f['candidate']}")
        else:
            lines.append("- All regression-list questions passed in every candidate run.")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Compare eval runs question by question.")
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="*", default=[])
    parser.add_argument("--name", required=True, help="report name, e.g. baseline or change-a")
    parser.add_argument("--regression", help="regression list JSON to check candidate runs against")
    parser.add_argument("--write-regression", help="write stable baseline passes to this JSON file")
    parser.add_argument("--out", default="data/eval/comparisons")
    args = parser.parse_args(argv)

    baseline = [load_run(r) for r in args.baseline]
    candidate = [load_run(r) for r in args.candidate]
    regression_ids = None
    if args.regression:
        regression_ids = json.loads(Path(args.regression).read_text(encoding="utf-8"))["ids"]
    res = compare(baseline, candidate, regression_ids)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = to_markdown(args.name, res)
    (out / f"{args.name}.md").write_text(report, encoding="utf-8")
    (out / f"{args.name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(report)
    print(f"Saved {out / (args.name + '.md')}")

    if args.write_regression:
        stable = [q["id"] for q in res["per_question"] if q["status"] in ("stable_pass", "still_pass")]
        Path(args.write_regression).write_text(json.dumps({
            "description": "Questions that passed in every baseline run. After any change, each must still pass "
                           "or the failure must be explained.",
            "from_runs": res["baseline_runs"], "question_set_versions": res["question_set_versions"],
            "ids": stable}, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote regression list ({len(stable)} questions) to {args.write_regression}")
    return res


if __name__ == "__main__":
    main()
