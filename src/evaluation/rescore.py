"""
rescore.py - re-score a saved evaluation run against a newer question file,
without calling the LLM again.

Why: label errors are found by reading results (baseline q13, q14, q22, q26).
After fixing the labels, the old runs must be scored with the SAME labels as
new runs, or before/after comparisons mix two different rulers.

Usage (from the project root):
    python -m src.evaluation.rescore data/eval/runs/<run_id>
    python -m src.evaluation.rescore data/eval/runs/<run_id> --questions data/eval/questions_v2.json

Writes a sibling folder data/eval/runs/<run_id>__rescored_<version>/ with
results.csv and summary.json. The original run is never modified.

What can be recomputed from a saved row
---------------------------------------
- Answer checks: the answer text is saved, so answer_ok, missing, refused and
  behaviour_ok are recomputed exactly.
- Final-context hit and citations: final_pages and cited_pages are saved.
- Stage hits (dense, BM25, RRF, rerank):
    * exactly, if the row has the ranked page lists (runs made with v2.1 code);
    * otherwise copied from the original row when the question's labels did
      not change, and set to unknown (blank) when they did.
- lost_at: recomputed from the per-stage hits ("unknown_stage" if a needed
  stage is unknown).
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation import metrics_v2 as m
from src.evaluation.eval_v2 import GROUP, TIMING_COLUMNS, question_set_version, summarise, write_outputs
from src.observability.logging_config import setup_logging

logger = logging.getLogger(__name__)

STAGE_LISTS = ("dense", "bm25", "rrf", "rerank")


def parse_pages(text: Optional[str]) -> List[m.PageKey]:
    """'file.pdf p12; file.pdf p34' -> [('file.pdf', 12), ('file.pdf', 34)]"""
    pages = []
    for part in (text or "").split(";"):
        part = part.strip()
        if part:
            name, num = part.rsplit(" p", 1)
            pages.append((name.strip(), int(num)))
    return pages


def parse_bool(value: Any) -> Optional[bool]:
    return {"TRUE": True, "FALSE": False}.get(str(value).strip().upper())


def parse_num(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(run_dir: Path) -> List[Dict[str, str]]:
    with open(run_dir / "results.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def rescore_row(row: Dict[str, str], q: Dict[str, Any], version: str) -> Dict[str, Any]:
    """Return a typed row scored against question q (new labels)."""
    out: Dict[str, Any] = dict(row)
    out["question_set_version"] = version
    out["group"] = GROUP.get(q["question_type"], "other")
    for k in ("latency_s", "total_tokens", "cost_usd", "citation_precision", "dedup_removed",
              "dense_bm25_overlap", "hyde_chars", "hyde_completion_tokens", "hyde_reasoning_chars",
              "generation_completion_tokens", "generation_reasoning_chars", *TIMING_COLUMNS):
        out[k] = parse_num(row.get(k))
    if row.get("error"):
        out["behaviour_ok"] = False
        return out

    relevant = m.relevant_set(q)
    labels_changed = relevant != set(parse_pages(row.get("relevant_pages")))
    out["relevant_pages"] = "; ".join(f"{f} p{p}" for f, p in sorted(relevant))

    answer = row.get("answer", "")
    refused = m.is_refusal(answer)
    answer_ok, missing = m.check_must_include(answer, q["must_include"])
    out.update({
        "refused": refused,
        "answer_ok": answer_ok,
        "missing": json.dumps(missing, ensure_ascii=False) if missing else "",
        "behaviour_ok": m.behaviour_correct(q["expected_behavior"], refused, answer_ok),
    })

    ranked = {s: parse_pages(row.get(f"{s}_pages")) for s in STAGE_LISTS}
    have_ranked = all(ranked[s] for s in STAGE_LISTS)
    for s in STAGE_LISTS:
        for k, metric in ((20, "hit20"), (5, "hit5")):
            col = f"{s}_{metric}"
            if col not in out:
                continue
            if have_ranked:
                out[col] = m.hit_at_k(ranked[s], relevant, k)
            elif not labels_changed:
                out[col] = parse_bool(row.get(col))
            else:
                out[col] = None
        rr_col = f"{s}_rr5"
        if have_ranked:
            out[rr_col] = m.reciprocal_rank_at_k(ranked[s], relevant, 5)
        elif not labels_changed:
            out[rr_col] = parse_num(row.get(rr_col))
        else:
            out[rr_col] = None

    final = parse_pages(row.get("final_pages"))
    out["final_hit5"] = m.hit_at_k(final, relevant, 5)
    cites = m.citation_metrics(parse_pages(row.get("cited_pages")), relevant)
    out["citation_precision"] = cites["citation_precision"]
    out["any_citation_relevant"] = cites["any_citation_relevant"]

    if not relevant:
        out["lost_at"] = "n/a"
    else:
        hits = {"dense": out.get("dense_hit20"), "bm25": out.get("bm25_hit20"), "rrf": out.get("rrf_hit20"),
                "rerank": out.get("rerank_hit5"), "final": out["final_hit5"]}
        out["lost_at"] = m.locate_loss_from_hits(hits, answer_ok, refused)
    return out


def rescore_run(run_dir: Path, questions_path: str) -> Path:
    questions = {q["id"]: q for q in json.loads(Path(questions_path).read_text(encoding="utf-8"))["questions"]}
    version = question_set_version(questions_path)
    old_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [rescore_row(r, questions[r["id"]], version) for r in load_rows(run_dir)]

    changed = []
    for old, new in zip(load_rows(run_dir), rows):
        if parse_bool(old["behaviour_ok"]) != new["behaviour_ok"] or old["lost_at"] != new["lost_at"]:
            b_old, b_new = old["behaviour_ok"].upper(), "TRUE" if new["behaviour_ok"] else "FALSE"
            changed.append(f"{new['id']}: behaviour {b_old}->{b_new}, lost_at {old['lost_at']}->{new['lost_at']}")

    new_id = f"{run_dir.name}__rescored_{version}"
    groups = sorted({r["group"] for r in rows})
    summary = {
        "run_id": new_id,
        "rescored_from": run_dir.name,
        "config": dict(old_summary["config"], question_set_version=version, questions_file=questions_path),
        "rescore_changes": changed,
        "overall": summarise(rows),
        "by_group": {g: summarise([r for r in rows if r["group"] == g]) for g in groups},
    }
    out_dir = run_dir.parent / new_id
    write_outputs(out_dir, rows, summary)
    logger.info("Re-scored %s with question set %s -> %s", run_dir.name, version, out_dir)
    for line in changed:
        logger.info("  changed: %s", line)
    if not changed:
        logger.info("  no row changed")
    return out_dir


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Re-score saved eval runs with a newer question file.")
    parser.add_argument("runs", nargs="+", help="run folders under data/eval/runs/")
    parser.add_argument("--questions", default="data/eval/questions_v2.json")
    args = parser.parse_args(argv)
    setup_logging()
    for run in args.runs:
        rescore_run(Path(run), args.questions)


if __name__ == "__main__":
    main()
