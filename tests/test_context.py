"""
Unit tests for src/generation/context.py (change F). No models, no network.

Run with pytest:   python -m pytest tests/test_context.py -q
Or without it:     python -m tests.test_context
"""

from src.generation import context as ctx

F = "infosys-ar-26.pdf"


def chunk(page, idx, text, score=None):
    c = {"text": text, "metadata": {"file_name": F, "page_number": page, "chunk_index": idx}}
    if score is not None:
        c["rerank_score"] = score
    return c


# Page 80 in miniature: c0 table heading + rows, c1 rows, c2 the answer, c3 unrelated.
# Consecutive chunks share a 25-character overlap, as the splitter produces.
OV1 = "Segmental operating income"
OV2 = "partially offset by benefits"
P80 = [
    chunk(80, 0, "Business segments (In Rs crore) Financial Services ... " + OV1),
    chunk(80, 1, OV1 + " 2026 12,678 6,444 ... revenue was " + OV2),
    chunk(80, 2, OV2 + " Revenue by geography: North America 1,00,167"),
    chunk(80, 3, "Liquidity: our principal source of liquidity is cash"),
]
INDEX = ctx.build_page_index(list(reversed(P80)) + [chunk(81, 0, "other page")])


def test_index_groups_by_page_and_sorts_by_chunk_index():
    assert [c["metadata"]["chunk_index"] for c in INDEX[(F, 80)]] == [0, 1, 2, 3]
    assert len(INDEX[(F, 81)]) == 1


def test_overlap_is_removed_once():
    joined = ctx.join_overlapping(P80[0]["text"], P80[1]["text"])
    assert joined.count(OV1) == 1
    assert ctx.join_overlapping("abc", "xyz") == "abc\nxyz"   # no overlap: newline join


def test_q04_shape_answer_chunk_reaches_the_model():
    best = chunk(80, 0, P80[0]["text"], score=3.2)
    out = ctx.page_window(best, INDEX, max_chars=10_000)
    assert "North America 1,00,167" in out["text"]
    assert out["metadata"]["context_chunk_indices"] == [0, 1, 2, 3]
    assert out["metadata"]["page_number"] == 80 and out["metadata"]["chunk_index"] == 0
    assert out["rerank_score"] == 3.2


def test_q22_shape_previous_chunk_comes_first():
    # Best chunk c1 (numbers); the budget fits exactly one neighbour: the heading chunk c0 wins.
    best = chunk(80, 1, P80[1]["text"])
    one_neighbour = len(ctx._merged_text(INDEX[(F, 80)], 0, 1))
    out = ctx.page_window(best, INDEX, max_chars=one_neighbour)
    assert out["metadata"]["context_chunk_indices"] == [0, 1]
    assert out["text"].startswith("Business segments")


def test_budget_is_respected_and_zero_disables():
    best = chunk(80, 0, P80[0]["text"])
    small = ctx.page_window(best, INDEX, max_chars=len(P80[0]["text"]) + 5)
    assert small["metadata"]["context_chunk_indices"] == [0]
    assert len(small["text"]) <= len(P80[0]["text"]) + 5
    off = ctx.page_window(best, INDEX, max_chars=0)
    assert off["text"] == best["text"] and off["metadata"]["context_chunk_indices"] == [0]


def test_unknown_page_falls_back_to_the_chunk_and_input_is_not_mutated():
    stray = chunk(999, 4, "lonely chunk")
    out = ctx.page_window(stray, INDEX, max_chars=2500)
    assert out["text"] == "lonely chunk"
    assert "context_chunk_indices" not in stray["metadata"]


def test_assemble_keeps_order():
    sel = [chunk(81, 0, "other page"), chunk(80, 2, P80[2]["text"])]
    out = ctx.assemble_page_context(sel, INDEX, 2500)
    assert [c["metadata"]["page_number"] for c in out] == [81, 80]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
