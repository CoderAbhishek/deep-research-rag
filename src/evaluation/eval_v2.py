"""
eval_v2.py - run the page-labelled evaluation set through the full pipeline.

Usage (from the project root, with the venv active):
    python -m src.evaluation.eval_v2                         # dev split (default)
    python -m src.evaluation.eval_v2 --limit 3 --sleep 5     # quick smoke run
    python -m src.evaluation.eval_v2 --ids q04 q08           # specific questions
    python -m src.evaluation.eval_v2 --split test --allow-test   # reporting only

Why the test split needs --allow-test
-------------------------------------
"When a measure becomes a target, it ceases to be a good measure." Error
analysis and tuning use the dev split only. The test split is run once a
change is finished, to report numbers that were not tuned against. The flag
makes running it a deliberate act.

What one run produces (data/eval/runs/<run_id>/)
------------------------------------------------
results.csv   one row per question:
                - the answer and its checks (refused, answer_ok, behaviour_ok)
                - where the relevant page was found or lost, stage by stage
                  (dense / BM25 / RRF / rerank / final context)
                - citations, latency, tokens, cost, request_id, trace_id
                - EMPTY review columns for error analysis:
                  review_pass, review_note, failure_category
summary.json  run configuration (git commit, models, pool sizes) and
              aggregate metrics, overall and per answer group

Each question is traced in LangSmith with tags "evaluation" and "eval-v2"
and metadata question_id / split, so a failing row leads to its trace.
"""

import argparse
import csv
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation import metrics_v2 as m
from src.observability.logging_config import new_request_id, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS = "data/eval/questions_v2.json"
DEFAULT_OUT = "data/eval/runs"

GROUP = {"number": "numeric", "percentage": "numeric", "table_lookup": "numeric", "comparison": "numeric",
         "definition": "text", "person": "text", "policy": "text", "list": "text", "narrative": "text",
         "unanswerable": "decline", "out_of_scope": "decline", "false_premise": "decline",
         "adversarial": "decline", "off_topic": "decline"}

CSV_COLUMNS = [
    "id", "split", "question_set_version", "group", "question_type", "difficulty", "phrasing", "expected_behavior",
    "question", "reference_answer", "answer",
    "behaviour_ok", "answer_ok", "refused", "missing", "lost_at",
    "dense_hit20", "bm25_hit20", "rrf_hit20",
    "dense_hit5", "bm25_hit5", "rrf_hit5", "rerank_hit5", "final_hit5",
    "dense_rr5", "bm25_rr5", "rrf_rr5", "rerank_rr5",
    "relevant_pages", "final_pages", "cited_pages", "citation_precision", "any_citation_relevant",
    "dense_pages", "bm25_pages", "rrf_pages", "rerank_pages",
    "rrf_chunks", "rerank_chunks", "final_chunks", "generation_chunks", "dedup_removed_pool",
    "dedup_removed", "dense_bm25_overlap",
    "latency_s", "total_tokens", "cost_usd", "hyde_finish_reason", "generation_finish_reason",
    "hyde_chars", "hyde_completion_tokens", "hyde_reasoning_chars", "hyde_fallback",
    "generation_completion_tokens", "generation_reasoning_chars",
    "error", "request_id", "trace_id",
    "review_pass", "review_note", "failure_category",
]


def question_set_version(path: str) -> str:
    return json.loads(Path(path).read_text(encoding="utf-8")).get("version", "unknown")


def load_questions(path: str, split: str, ids: Optional[List[str]], limit: Optional[int]) -> List[Dict[str, Any]]:
    questions = json.loads(Path(path).read_text(encoding="utf-8"))["questions"]
    if ids:
        wanted = set(ids)
        questions = [q for q in questions if q["id"] in wanted]
    elif split != "all":
        questions = [q for q in questions if q["split"] == split]
    return questions[:limit] if limit else questions


def _pages(items: List[Dict[str, Any]]) -> List[m.PageKey]:
    return [m.page_key(i) for i in items]


def _fmt_pages(pages: List[m.PageKey]) -> str:
    """Unique pages, in order, as 'file pN; file pM' (used for display)."""
    return "; ".join(f"{f} p{p}" for f, p in dict.fromkeys(pages))


def _fmt_chunks(items: List[Dict[str, Any]]) -> str:
    """Chunk-level identity in rank order: 'file pN cK; ...' (diagnoses which chunk of a page was used)."""
    return "; ".join(f"{i.get('file_name')} p{i.get('page_number')} c{i.get('chunk_index')}" for i in items)


def _fmt_ranked(pages: List[m.PageKey]) -> str:
    """Every chunk's page, in rank order, duplicates kept (used for re-scoring)."""
    return "; ".join(f"{f} p{p}" for f, p in pages)


def evaluate_question(q: Dict[str, Any], resources: Dict[str, Any], run_id: str,
                      version: str = "unknown") -> Dict[str, Any]:
    """Run one question through the pipeline and score it. Never raises."""
    from src.pipeline.rag_pipeline import run_pipeline

    request_id = new_request_id()
    row: Dict[str, Any] = {k: "" for k in CSV_COLUMNS}
    row.update({
        "id": q["id"], "split": q["split"], "question_set_version": version,
        "group": GROUP.get(q["question_type"], "other"),
        "question_type": q["question_type"], "difficulty": q["difficulty"], "phrasing": q["phrasing"],
        "expected_behavior": q["expected_behavior"], "question": q["question"],
        "reference_answer": q["reference_answer"], "request_id": request_id,
    })
    relevant = m.relevant_set(q)
    row["relevant_pages"] = _fmt_pages(list(relevant))

    try:
        result = run_pipeline(
            q["question"], resources,
            langsmith_extra={
                "metadata": {"request_id": request_id, "entrypoint": "evaluation",
                             "question_id": q["id"], "split": q["split"], "eval_run": run_id},
                "tags": ["evaluation", "eval-v2", q["split"]],
            },
        )
    except Exception as exc:
        logger.exception("Pipeline failed for %s", q["id"])
        row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        row["behaviour_ok"] = False
        return row

    answer = result["answer"]
    debug = result["debug"]
    refused = m.is_refusal(answer)
    answer_ok, missing = m.check_must_include(answer, q["must_include"])

    stage_pages = {
        "dense": _pages(debug.get("dense", [])),
        "bm25": _pages(debug.get("bm25", [])),
        "rrf": _pages(debug["rrf_pool"]),
        "rerank": _pages(debug["reranked"]),
        "final": _pages(result.get("retrieved_sources", [])),
    }
    cited = _pages(result["sources"])
    hyde_llm = debug["llm"].get("hyde") or {}
    gen_llm = debug["llm"].get("generation") or {}
    cites = m.citation_metrics(cited, relevant)

    row.update({
        "answer": answer,
        "refused": refused,
        "answer_ok": answer_ok,
        "missing": json.dumps(missing, ensure_ascii=False) if missing else "",
        "behaviour_ok": m.behaviour_correct(q["expected_behavior"], refused, answer_ok),
        "lost_at": m.locate_loss(stage_pages, relevant, answer_ok, refused),
        "dense_hit20": m.hit_at_k(stage_pages["dense"], relevant, 20),
        "bm25_hit20": m.hit_at_k(stage_pages["bm25"], relevant, 20),
        "rrf_hit20": m.hit_at_k(stage_pages["rrf"], relevant, 20),
        "dense_hit5": m.hit_at_k(stage_pages["dense"], relevant, 5),
        "bm25_hit5": m.hit_at_k(stage_pages["bm25"], relevant, 5),
        "rrf_hit5": m.hit_at_k(stage_pages["rrf"], relevant, 5),
        "rerank_hit5": m.hit_at_k(stage_pages["rerank"], relevant, 5),
        "final_hit5": m.hit_at_k(stage_pages["final"], relevant, 5),
        "dense_rr5": m.reciprocal_rank_at_k(stage_pages["dense"], relevant, 5),
        "bm25_rr5": m.reciprocal_rank_at_k(stage_pages["bm25"], relevant, 5),
        "rrf_rr5": m.reciprocal_rank_at_k(stage_pages["rrf"], relevant, 5),
        "rerank_rr5": m.reciprocal_rank_at_k(stage_pages["rerank"], relevant, 5),
        "final_pages": _fmt_ranked(stage_pages["final"]),
        "cited_pages": _fmt_pages(cited),
        "citation_precision": cites["citation_precision"],
        "any_citation_relevant": cites["any_citation_relevant"],
        "dedup_removed": debug["counts"]["dedup_removed"],
        "dense_bm25_overlap": debug["counts"]["dense_bm25_overlap"],
        "latency_s": debug["timings_s"]["total_s"],
        "total_tokens": debug["total_tokens"],
        "cost_usd": debug["total_cost_usd"],
        "hyde_finish_reason": hyde_llm.get("finish_reason"),
        "generation_finish_reason": gen_llm.get("finish_reason"),
        # Per-call detail, used to set token ceilings from data (DR-26, DR-33).
        "hyde_chars": len(debug.get("hyde_passage") or ""),
        "hyde_completion_tokens": hyde_llm.get("completion_tokens"),
        "hyde_reasoning_chars": hyde_llm.get("reasoning_chars"),
        "hyde_fallback": debug.get("hyde_fallback"),
        "generation_completion_tokens": gen_llm.get("completion_tokens"),
        "generation_reasoning_chars": gen_llm.get("reasoning_chars"),
        # Full ranked page lists, so the run can be re-scored if labels change.
        "dense_pages": _fmt_ranked(stage_pages["dense"]),
        "bm25_pages": _fmt_ranked(stage_pages["bm25"]),
        "rrf_pages": _fmt_ranked(stage_pages["rrf"]),
        "rerank_pages": _fmt_ranked(stage_pages["rerank"]),
        # Chunk-level lists (change B): which chunk of each page was kept.
        "rrf_chunks": _fmt_chunks(debug["rrf_pool"]),
        "rerank_chunks": _fmt_chunks(debug.get("reranked_all") or debug["reranked"]),
        "final_chunks": _fmt_chunks(debug.get("final", [])),
        "generation_chunks": debug["counts"].get("generation_chunks"),
        "dedup_removed_pool": debug["counts"].get("dedup_removed_pool"),
        "trace_id": result.get("trace_id") or "",
    })
    return row


def _chunks_sent(r: Dict[str, Any]) -> Optional[float]:
    """
    Chunks sent to the model. Runs before change B have no generation_chunks
    column; there it is 5 minus the duplicates removed from the top 5
    (top_k_generate was 5 in every run so far).
    """
    for key, fn in (("generation_chunks", lambda v: v), ("dedup_removed", lambda v: 5 - v)):
        v = r.get(key)
        if v not in (None, ""):
            try:
                return fn(float(v))
            except (TypeError, ValueError):
                pass
    return None


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics for a list of rows (None values are skipped)."""
    ok = [r for r in rows if not r["error"]]
    answerable = [r for r in ok if r["expected_behavior"] == "answer"]
    decline = [r for r in ok if r["expected_behavior"] == "refuse"]
    lost: Dict[str, int] = {}
    for r in ok:
        lost[r["lost_at"]] = lost.get(r["lost_at"], 0) + 1
    return {
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "behaviour_accuracy": m.mean(r["behaviour_ok"] for r in rows),
        "answer_accuracy_answerable": m.mean(r["answer_ok"] for r in answerable),
        "false_refusal_rate_answerable": m.mean(r["refused"] for r in answerable),
        "correct_refusal_rate_should_decline": m.mean(r["refused"] for r in decline),
        "retrieval": {
            "hit@20": {s: m.mean(r[f"{s}_hit20"] for r in ok) for s in ("dense", "bm25", "rrf")},
            "hit@5": {s: m.mean(r[f"{s}_hit5"] for r in ok) for s in ("dense", "bm25", "rrf", "rerank", "final")},
            "mrr@5": {s: m.mean(r[f"{s}_rr5"] for r in ok) for s in ("dense", "bm25", "rrf", "rerank")},
        },
        "lost_at": dict(sorted(lost.items())),
        "citation_precision": m.mean(r["citation_precision"] for r in ok),
        "latency_s": {"p50": m.percentile([r["latency_s"] for r in ok], 50),
                      "p95": m.percentile([r["latency_s"] for r in ok], 95)},
        "tokens_mean": m.mean(r["total_tokens"] for r in ok),
        "hyde_cut_off": sum(1 for r in ok if r.get("hyde_finish_reason") == "length"),
        "generation_cut_off": sum(1 for r in ok if r.get("generation_finish_reason") == "length"),
        "hyde_fallback": sum(1 for r in ok if r.get("hyde_fallback") is True),
        "fewer_than_5_chunks": sum(1 for r in ok if (_chunks_sent(r) or 5) < 5),
        "cost_usd_total": round(sum(r["cost_usd"] for r in ok if r["cost_usd"] is not None), 6) if ok else None,
    }


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _csv_value(v: Any) -> Any:
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    return "" if v is None else v


def write_outputs(out_dir: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # utf-8-sig adds a BOM so Excel shows ₹ and other characters correctly.
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _csv_value(r.get(k)) for k in CSV_COLUMNS})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def log_summary(s: Dict[str, Any]) -> None:
    o = s["overall"]
    logger.info("=" * 70)
    logger.info("EVAL SUMMARY  run=%s  split=%s  questions=%d  errors=%d",
                s["run_id"], s["config"]["split"], o["questions"], o["errors"])
    logger.info("  behaviour accuracy (did the right thing)     %s", o["behaviour_accuracy"])
    logger.info("  answer accuracy (answerable questions)       %s", o["answer_accuracy_answerable"])
    logger.info("  false refusals (answerable questions)        %s", o["false_refusal_rate_answerable"])
    logger.info("  correct refusals (should-decline questions)  %s", o["correct_refusal_rate_should_decline"])
    logger.info("  hit@20 %s", o["retrieval"]["hit@20"])
    logger.info("  hit@5  %s", o["retrieval"]["hit@5"])
    logger.info("  mrr@5  %s", o["retrieval"]["mrr@5"])
    logger.info("  where answers were lost: %s", o["lost_at"])
    logger.info("  cut off at max_tokens: HyDE %s, generation %s   HyDE fallback used: %s",
                o["hyde_cut_off"], o["generation_cut_off"], o["hyde_fallback"])
    logger.info("  latency p50/p95 %s s   tokens/question %s   total cost USD %s",
                (o["latency_s"]["p50"], o["latency_s"]["p95"]), o["tokens_mean"], o["cost_usd_total"])
    for g, gs in s["by_group"].items():
        logger.info("  [%s] n=%d behaviour=%s answer=%s rerank_hit5=%s", g, gs["questions"],
                    gs["behaviour_accuracy"], gs["answer_accuracy_answerable"], gs["retrieval"]["hit@5"]["rerank"])
    logger.info("=" * 70)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run the page-labelled evaluation set (v2).")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--ids", nargs="*", help="run only these question IDs")
    parser.add_argument("--limit", type=int, help="run only the first N questions")
    parser.add_argument("--sleep", type=float, default=15.0, help="seconds between questions (Groq rate limits)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--allow-test", action="store_true", help="required to run the held-out test split")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv()
    setup_logging()

    if args.split in ("test", "all") and not args.ids and not args.allow_test:
        parser.error("the test split is for final reporting only; add --allow-test to run it deliberately")

    questions = load_questions(args.questions, args.split, args.ids, args.limit)
    if not questions:
        parser.error("no questions selected")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + ("ids" if args.ids else args.split)
    version = question_set_version(args.questions)

    from src.pipeline import rag_pipeline as rp
    from src.generation import generator as gen
    from src.retrieval import query as qmod

    from src.observability.tracing import log_tracing_status  # imported here so rescore/compare need no langsmith
    log_tracing_status(logger)
    logger.info("Eval run %s: %d questions (split=%s)", run_id, len(questions), args.split)
    resources = rp.load_resources()

    rows = []
    for i, q in enumerate(questions, start=1):
        row = evaluate_question(q, resources, run_id, version)
        rows.append(row)
        logger.info("[%d/%d] %s behaviour=%s answer=%s lost_at=%s rerank_hit5=%s cited=%s",
                    i, len(questions), q["id"], row["behaviour_ok"], row["answer_ok"],
                    row["lost_at"] or "-", row["rerank_hit5"], row["cited_pages"] or "none")
        if i < len(questions):
            time.sleep(args.sleep)

    groups = sorted({r["group"] for r in rows})
    summary = {
        "run_id": run_id,
        "config": {
            "split": args.split, "ids": args.ids, "limit": args.limit, "questions_file": args.questions,
            "question_set_version": version,
            "git_commit": _git_commit(), "hyde_model": qmod.DEFAULT_MODEL, "generation_model": gen.DEFAULT_MODEL,
            "hyde_max_tokens": getattr(qmod, "QUERY_MAX_TOKENS", 512),
            "hyde_reasoning_effort": getattr(qmod, "QUERY_REASONING_EFFORT", None),
            "generation_max_tokens": getattr(gen, "GENERATION_MAX_TOKENS", 512),
            "generation_reasoning_effort": getattr(gen, "GENERATION_REASONING_EFFORT", None),
            "embed_model": rp.EMBED_MODEL, "reranker_model": rp.RERANKER_MODEL,
            "pool_size": rp.POOL_SIZE, "top_k_rerank": rp.TOP_K_RERANK, "top_k_generate": rp.TOP_K_GENERATE,
        },
        "overall": summarise(rows),
        "by_group": {g: summarise([r for r in rows if r["group"] == g]) for g in groups},
    }
    out_dir = Path(args.out) / run_id
    n = 2
    while out_dir.exists():  # never overwrite an earlier run started in the same second
        out_dir = Path(args.out) / f"{run_id}-{n}"
        n += 1
    summary["run_id"] = out_dir.name
    write_outputs(out_dir, rows, summary)
    log_summary(summary)
    logger.info("Wrote %s and %s", out_dir / "results.csv", out_dir / "summary.json")
    return summary


if __name__ == "__main__":
    main()
