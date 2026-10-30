"""
Unit tests for src/evaluation/metrics_v2.py. No models, no network, no API key.

Run with pytest:   python -m pytest tests/test_metrics_v2.py -q
Or without it:     python -m tests.test_metrics_v2
"""

from src.evaluation import metrics_v2 as m

AR, AGM = "infosys-ar-26.pdf", "agm-2026-transcript.pdf"
REL = {(AR, 80)}


def test_normalize_ignores_commas_spaces_and_rupee_signs():
    assert m.normalize("₹ 1,78,650 Crore") == "178650crore"
    assert m.normalize("`1,78,650 cr") == "178650cr"
    assert m.normalize(None) == ""


def test_hit_and_reciprocal_rank():
    ranked = [(AR, 10), (AGM, 47), (AR, 80), (AR, 81)]
    assert m.hit_at_k(ranked, REL, 5) is True
    assert m.hit_at_k(ranked, REL, 2) is False
    assert abs(m.reciprocal_rank_at_k(ranked, REL, 5) - 1 / 3) < 1e-9
    assert m.reciprocal_rank_at_k(ranked, REL, 2) == 0.0


def test_no_relevant_pages_gives_none_not_zero():
    assert m.hit_at_k([(AR, 1)], set(), 5) is None
    assert m.reciprocal_rank_at_k([(AR, 1)], set(), 5) is None


def test_must_include_alternatives_and_missing_groups():
    ok, missing = m.check_must_include("Retail had 30.7% [SOURCE 1]", [["Retail"], ["30.7"]])
    assert ok is True and missing == []
    ok, missing = m.check_must_include("Revenue was ₹1,78,650 crore", [["1,78,650"], ["9.6"]])
    assert ok is False and missing == [["9.6"]]
    ok, _ = m.check_must_include("US$20.2 billion", [["20,158", "20.2"]])
    assert ok is True
    assert m.check_must_include("anything", []) == (None, [])


def test_refusal_detection():
    assert m.is_refusal("The provided documents do not contain sufficient information to answer this question.")
    assert not m.is_refusal("Revenue was ₹1,78,650 crore [SOURCE 1].")


def test_behaviour():
    assert m.behaviour_correct("refuse", refused=True, answer_ok=None)
    assert not m.behaviour_correct("refuse", refused=False, answer_ok=None)
    assert m.behaviour_correct("answer", refused=False, answer_ok=True)
    assert not m.behaviour_correct("answer", refused=True, answer_ok=None)
    assert not m.behaviour_correct("answer", refused=False, answer_ok=False)
    assert m.behaviour_correct("answer_or_refuse", refused=True, answer_ok=False)
    assert m.behaviour_correct("answer_or_refuse", refused=False, answer_ok=True)


def test_citation_metrics():
    c = m.citation_metrics([(AR, 80), (AR, 10), (AR, 80)], REL)
    assert c == {"cited_count": 2, "citation_precision": 0.5, "any_citation_relevant": True}
    assert m.citation_metrics([], REL)["citation_precision"] is None


def test_locate_loss_walks_the_pipeline_in_order():
    base = {"dense": [(AR, 80)], "bm25": [], "rrf": [(AR, 80)], "rerank": [(AR, 80)], "final": [(AR, 80)]}
    assert m.locate_loss(base, REL, answer_ok=True, refused=False) == "ok"
    assert m.locate_loss(base, REL, answer_ok=False, refused=False) == "generation"
    # A page lost before the final context, with a wrong answer or refusal:
    assert m.locate_loss(dict(base, final=[]), REL, answer_ok=False, refused=True) == "dedup"
    assert m.locate_loss(dict(base, rerank=[], final=[]), REL, False, True) == "rerank"
    assert m.locate_loss(dict(base, rrf=[], rerank=[], final=[]), REL, False, True) == "fusion"
    assert m.locate_loss({"dense": [(AR, 1)], "bm25": [(AR, 2)]}, REL, False, True) == "not_retrieved"
    assert m.locate_loss(base, set(), True, False) == "n/a"


def test_correct_answer_from_unlabelled_page_is_a_label_gap():
    # Relevant page never retrieved, yet the answer is correct: the model used a page
    # that is missing from relevant_pages (baseline q13, q22, q26).
    stages = {"dense": [(AR, 38)], "bm25": [], "rrf": [(AR, 38)], "rerank": [(AR, 38)], "final": [(AR, 38)]}
    assert m.locate_loss(stages, REL, answer_ok=True, refused=False) == "ok_unlabelled_source"
    # Same retrieval, but the answer is wrong: still a retrieval failure.
    assert m.locate_loss(stages, REL, answer_ok=False, refused=False) == "not_retrieved"


def test_locate_loss_from_hits_matches_and_handles_unknowns():
    hits = {"dense": True, "bm25": False, "rrf": True, "rerank": False, "final": False}
    assert m.locate_loss_from_hits(hits, answer_ok=False, refused=True) == "rerank"
    assert m.locate_loss_from_hits(hits, answer_ok=True, refused=False) == "ok_unlabelled_source"
    unknown = {"dense": None, "bm25": None, "rrf": None, "rerank": None, "final": True}
    assert m.locate_loss_from_hits(unknown, answer_ok=True, refused=False) == "ok"
    assert m.locate_loss_from_hits(dict(unknown, final=None), answer_ok=False, refused=True) == "unknown_stage"


def test_mean_and_percentile():
    assert m.mean([True, False, None, True]) == round(2 / 3, 4)
    assert m.mean([None]) is None
    assert m.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
