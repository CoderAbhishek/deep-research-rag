"""
context.py - page-level context assembly (change F, DR-38).

The problem
-----------
Retrieval ranks chunks, then page deduplication keeps the single
highest-scoring chunk of each page. The model therefore sees one 512-character
slice of each page. Two failures in the change B runs came from that slice:

  q04  page 80 entered the pool as c0 (segment table) and c2 (revenue by
       geography, the answer). c0 scored higher, dedup kept c0, the answer
       chunk was dropped, the model refused.
  q22  the model used chunk c1 of page 16 (US$ table rows) without the
       "In US$ million" heading, which sits at the end of c0, and answered
       with a dollar EPS as if it were rupees.

The change
----------
Page selection is unchanged (same pages, same order, same [SOURCE X]
numbering). Only the TEXT sent for each selected page changes: instead of the
best chunk alone, a window of the page's chunks around it, in reading order,
up to a character budget per page.

  - Expand around the best chunk: previous chunk first, then next,
    alternating. Previous first because headings and table titles sit above
    the numbers they describe.
  - Stop when the next addition on either side would exceed the budget, or
    the page has no more chunks.
  - Neighbouring chunks overlap (RecursiveCharacterTextSplitter, overlap 64;
    measured ~62 characters on AR p80). The overlap is removed when joining,
    so the model does not read the same line twice.

Budget: PAGE_CONTEXT_CHARS = 2500 (set in rag_pipeline.py). Corpus pages have
a median of 7 chunks and ~3,160 characters (p90 ~5,900), so 2,500 covers the
best chunk plus about two neighbours on each side, or most of a typical page.
It is a design choice made from the page-length distribution, not tuned on
eval results. 0 disables the change (the best chunk alone, as before).
"""

from typing import Any, Dict, List, Tuple

PageKey = Tuple[str, int]

MIN_OVERLAP = 20     # shorter suffix/prefix matches are treated as coincidence
MAX_OVERLAP = 200    # splitter overlap is 64; leave room for whitespace differences


def _key(meta: Dict[str, Any]) -> PageKey:
    return meta.get("file_name"), meta.get("page_number")


def build_page_index(chunks: List[Dict[str, Any]]) -> Dict[PageKey, List[Dict[str, Any]]]:
    """
    Group every chunk of the corpus by page, sorted by chunk_index.

    `chunks` is the list load_resources() already holds for BM25 (all 3,439
    chunks with text and metadata), so no extra database call is needed.
    """
    index: Dict[PageKey, List[Dict[str, Any]]] = {}
    for c in chunks:
        index.setdefault(_key(c["metadata"]), []).append(c)
    for page_chunks in index.values():
        page_chunks.sort(key=lambda c: c["metadata"].get("chunk_index", 0))
    return index


def join_overlapping(a: str, b: str) -> str:
    """
    Join two consecutive chunks, removing the text b repeats from the end of a.

    The longest suffix of `a` that is also a prefix of `b` (MIN_OVERLAP to
    MAX_OVERLAP characters) is kept once. With no such overlap the chunks are
    joined with a newline.
    """
    for k in range(min(MAX_OVERLAP, len(a), len(b)), MIN_OVERLAP - 1, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return a + "\n" + b


def _merged_text(page_chunks: List[Dict[str, Any]], lo: int, hi: int) -> str:
    text = page_chunks[lo]["text"]
    for i in range(lo + 1, hi + 1):
        text = join_overlapping(text, page_chunks[i]["text"])
    return text


def page_window(best: Dict[str, Any], page_index: Dict[PageKey, List[Dict[str, Any]]],
                max_chars: int) -> Dict[str, Any]:
    """
    Return a copy of `best` whose text is a window of its page around it.

    The copy keeps the original metadata and scores (so citations, logs and
    evaluation still refer to the same page and best chunk) and adds:
      metadata["context_chunk_indices"]  chunk_index values included, in order
      metadata["context_chars"]          length of the merged text
    """
    page_chunks = page_index.get(_key(best["metadata"])) or []
    indices = [c["metadata"].get("chunk_index") for c in page_chunks]
    best_idx = best["metadata"].get("chunk_index")
    if max_chars <= 0 or best_idx not in indices:
        lo = hi = None
        text = best["text"]
        included = [best_idx]
    else:
        pos = indices.index(best_idx)
        lo = hi = pos
        text = page_chunks[pos]["text"]
        while True:
            grew = False
            for side in ("prev", "next"):
                if side == "prev" and lo > 0:
                    candidate = _merged_text(page_chunks, lo - 1, hi)
                    if len(candidate) <= max_chars:
                        lo, text, grew = lo - 1, candidate, True
                elif side == "next" and hi < len(page_chunks) - 1:
                    candidate = _merged_text(page_chunks, lo, hi + 1)
                    if len(candidate) <= max_chars:
                        hi, text, grew = hi + 1, candidate, True
            if not grew:
                break
        included = indices[lo:hi + 1]

    out = dict(best)
    out["metadata"] = dict(best["metadata"], context_chunk_indices=included, context_chars=len(text))
    out["text"] = text
    return out


def assemble_page_context(selected: List[Dict[str, Any]], page_index: Dict[PageKey, List[Dict[str, Any]]],
                          max_chars: int) -> List[Dict[str, Any]]:
    """Apply page_window to each selected chunk, keeping order (= [SOURCE X] numbering)."""
    return [page_window(c, page_index, max_chars) for c in selected]
