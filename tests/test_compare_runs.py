"""
Unit tests for the noise floor and per-question moves in
src/evaluation/compare_runs.py (DR-36). No models, no network, no API key.

Run with pytest:   python -m pytest tests/test_compare_runs.py -q
Or without it:     python -m tests.test_compare_runs
"""

from src.evaluation import compare_runs as c


def run(name, values, col="dense_hit20"):
    """A minimal run: {qid: value} for one column, plus an empty summary."""
    rows = {q: {"id": q, "question": q, "expected_behavior": "answer", "behaviour_ok": "TRUE", col: v}
            for q, v in values.items()}
    return {"name": name, "rows": rows, "summary": {"config": {}, "overall": {}}}


QIDS = [f"q{i:02d}" for i in range(1, 11)]


def test_flips_catch_churn_that_the_run_average_hides():
    # Two questions flip in opposite directions: averages equal, spread 0.
    a1 = run("a1", {q: ("TRUE" if q in ("q01", "q02") else "FALSE") for q in QIDS})
    a2 = run("a2", {q: ("TRUE" if q in ("q01", "q03") else "FALSE") for q in QIDS})
    means, ids = c._row_metric([a1, a2], "dense_hit20", None)
    assert means[0] == means[1]
    assert c.question_flips([a1, a2], "dense_hit20", ids) == 0.2


def test_noise_floor_is_the_largest_estimate_and_at_least_one_question():
    assert c.noise_floor(0.0, 0.2, 0.1, 10) == 0.2
    assert c.noise_floor(0.0, 0.0, 0.0, 10) == 0.1          # one question of ten
    assert c.noise_floor(0.3, None, None, None) == 0.3       # summary metrics: spread only
    assert c.noise_floor(None, None, None, None) is None


def test_one_question_difference_is_never_a_finding():
    assert c._verdict(0.1, c.noise_floor(0.0, 0.0, 0.0, 10), True) == "within noise"
    assert c._verdict(0.2, c.noise_floor(0.0, 0.0, 0.0, 10), True) == "better"
    assert c._verdict(0.2, c.noise_floor(0.0, 0.0, 0.0, 10), False) == "worse"


def test_moves_separate_consistent_from_partial_changes():
    base = [run("a1", {"q01": "FALSE", "q02": "TRUE", "q03": "TRUE", "q04": "FALSE"}),
            run("a2", {"q01": "FALSE", "q02": "FALSE", "q03": "TRUE", "q04": "FALSE"})]
    cand = [run("b1", {"q01": "TRUE", "q02": "TRUE", "q03": "FALSE", "q04": "FALSE"}),
            run("b2", {"q01": "TRUE", "q02": "TRUE", "q03": "TRUE", "q04": "FALSE"})]
    mv = c.question_moves(base, cand, "dense_hit20", ["q01", "q02", "q03", "q04"], higher_better=True)
    # q01: N N -> Y Y consistent gain; q02: Y N -> Y Y partial gain; q03: Y Y -> N Y partial loss.
    assert mv == {"better": 1, "better_partial": 1, "worse": 0, "worse_partial": 1}
    # For a lower-is-better metric the same values read the other way round.
    mv_low = c.question_moves(base, cand, "dense_hit20", ["q01", "q02", "q03", "q04"], higher_better=False)
    assert mv_low == {"better": 0, "better_partial": 1, "worse": 1, "worse_partial": 1}


def test_control_metric_that_moves_is_flagged():
    a = [run("a1", {q: "TRUE" for q in QIDS}), run("a2", {q: "TRUE" for q in QIDS})]
    b = [run("b1", {q: ("FALSE" if q in ("q01", "q02", "q03") else "TRUE") for q in QIDS}),
         run("b2", {q: ("FALSE" if q in ("q01", "q02", "q03") else "TRUE") for q in QIDS})]
    res = c.compare(a, b, None, control=["dense_hit20"])
    assert res["controls"][0]["ok"] is False
    stable = c.compare(a, a, None, control=["dense_hit20"])
    assert stable["controls"][0]["ok"] is True


def test_unknown_control_column_is_rejected():
    a = [run("a1", {"q01": "TRUE"})]
    try:
        c.compare(a, a, None, control=["no_such_column"])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
