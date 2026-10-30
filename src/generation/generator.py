"""
generator.py — LLM answer generation from retrieved chunks.

This module is the generation layer of the RAG pipeline. It takes a user
query and a list of retrieved + reranked chunks, formats them into a
structured prompt, calls an LLM, and returns a grounded answer with
source citations.

Design principles:
1. Grounding discipline: the LLM is explicitly instructed to answer ONLY
   from the provided context. It must not use parametric knowledge (facts
   memorised from training data).
2. Citation enforcement: every factual claim in the answer must be tied
   to a [SOURCE X] label that maps back to a specific file + page number.
3. Graceful failure: if the context does not contain enough information
   to answer the question, the model must say so — not hallucinate a
   plausible-sounding answer.
4. Low temperature: temperature=0.1 near-deterministic output. For
   grounded factual retrieval-augmented generation, creativity is the
   enemy — we want the model to copy figures accurately, not rephrase them.
"""

import re
from typing import List, Dict, Any

from src.llm.groq_client import chat


DEFAULT_MODEL = "openai/gpt-oss-20b"

# Token budget for answer generation. max_tokens caps hidden reasoning PLUS the
# answer for a reasoning model; at 512 the baseline returned empty answers on
# 2-3 of 24 questions with finish_reason=length (DR-26). 2048 is a ceiling,
# not a target: you pay only for tokens used.
GENERATION_MAX_TOKENS = 2048
GENERATION_REASONING_EFFORT = "low"


SYSTEM_PROMPT = """You are a research analyst assistant. You answer questions about business documents.

STRICT RULES — follow these exactly:
1. Answer using ONLY the context documents provided below. Do not use any knowledge not present in the context.
2. Cite your sources. Every factual claim must be attributed to a [SOURCE X] label. Example: "North America revenue was ₹1,00,167 crore [SOURCE 1]."
3. If the context does not contain sufficient information to answer, respond with exactly: "The provided documents do not contain sufficient information to answer this question."
4. Copy numbers, names, and figures exactly as they appear in the sources. Do not round, convert, or paraphrase figures.
5. Do not infer, calculate, or extrapolate beyond what the sources explicitly state."""


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a labelled context block for the LLM.

    Each chunk gets a [SOURCE X] label followed by its file name and page
    number. The model will reference these labels in its citations.

    Example output for two chunks:
    ─────────────────────────────────────────────
    [SOURCE 1] File: infosys-ar-26.pdf | Page: 80
    Infosys Integrated Annual Report 2025-26 111 Business segments –
    Consolidated (In ₹ crore) ...

    [SOURCE 2] File: infosys-ar-26.pdf | Page: 81
    The Company has identified the following ratios as key financial ...
    ─────────────────────────────────────────────

    Design decisions:
    - [SOURCE X] numbering starts at 1 (human-readable, matches how the
      model naturally counts in prose: "source 1", "source 2").
    - Metadata (file_name, page_number) is included on the label line so
      citations are traceable back to the original document without the
      model needing to know the internal metadata structure.
    - .get() with defaults on metadata fields: defensive programming — the
      function does not crash if a chunk has incomplete metadata.
    - "\n\n".join() puts a blank line between sources: visually separates
      chunks so the model does not conflate text from different pages.

    Args:
        chunks: List of chunk dicts, each with 'text' and 'metadata'.
                metadata must contain 'file_name' and 'page_number'.

    Returns:
        A single string containing all chunks, labelled and separated.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        file_name   = meta.get("file_name",   "unknown")
        page_number = meta.get("page_number", "?")

        label = f"[SOURCE {i}] File: {file_name} | Page: {page_number}"
        parts.append(f"{label}\n{chunk['text']}")

    return "\n\n".join(parts)


# Matches citation labels such as "[SOURCE 2]" and "【SOURCE 2】".
# The model sometimes uses full-width brackets and sometimes groups labels
# ("[SOURCE 1, SOURCE 3]"), so we match the "SOURCE <n>" token itself
# rather than the surrounding brackets.
_CITATION_PATTERN = re.compile(r"SOURCE\s*(\d+)")


def extract_cited_source_numbers(answer: str, num_chunks: int) -> List[int]:
    """
    Return the sorted, de-duplicated [SOURCE X] numbers cited in an answer.

    Only numbers that match a chunk actually passed to the model
    (1..num_chunks) are kept. A refusal, or an answer with no citation
    labels, returns an empty list.
    """
    nums = {int(n) for n in _CITATION_PATTERN.findall(answer or "")}
    return sorted(n for n in nums if 1 <= n <= num_chunks)


def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Generate a grounded, cited answer from retrieved chunks.

    This function is the final step in the RAG pipeline. It does three things:
    1. Formats the retrieved chunks into a labelled context block.
    2. Constructs a two-message prompt (system + user) and calls the LLM.
    3. Returns the answer along with source metadata for downstream use
       (e.g., displaying citations to the user, evaluation, logging).

    Args:
        query:  The user's original question, passed through unchanged.
                We use the ORIGINAL query here (not the HyDE passage).
                The HyDE passage was used for retrieval — the user never
                sees it. The model should answer the actual question asked.
        chunks: Retrieved and reranked chunks. These are the chunks the
                cross-encoder selected as most relevant. Each must have
                'text' and 'metadata' keys. Typically 3-7 chunks.
        model:  Groq model identifier. Default is DEFAULT_MODEL.

    Returns:
        A dict with:
            'answer'     — the LLM's answer string
            'sources'    — only the sources the answer actually cites,
                           parsed from its [SOURCE X] labels. Empty when
                           the answer cites nothing (e.g. a refusal).
            'retrieved_sources' — every chunk passed to the model, in
                           [SOURCE X] order (for debugging and evaluation)
            'query'      — the original query (for logging/tracing)
            'num_chunks' — number of chunks fed to the model
            'model'      — the model used (for logging/tracing)
            'llm'        - token usage, latency, finish_reason and
                           estimated cost of the generation call

    Why we return metadata alongside the answer:
    In production, you need to log what model answered a query, with how
    many chunks, from which sources. Without this, debugging a wrong answer
    requires reconstructing the entire query session from logs.
    """
    context = format_context(chunks)

    user_message = f"""Context documents:

{context}

---

Question: {query}

Answer the question using only the context documents above. Cite your sources using the [SOURCE X] labels."""

    # The shared wrapper traces this call in LangSmith and logs model,
    # latency, token usage, finish_reason and estimated cost.
    llm = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        model=model,
        temperature=0.1,
        max_tokens=GENERATION_MAX_TOKENS,
        reasoning_effort=GENERATION_REASONING_EFFORT,
        purpose="generation",
    )
    answer = llm.text  # already "" if the model returned no content

    retrieved_sources = [
        {
            "source_num":   i + 1,
            "file_name":    chunk["metadata"].get("file_name",   "unknown"),
            "page_number":  chunk["metadata"].get("page_number", "?"),
        }
        for i, chunk in enumerate(chunks)
    ]

    # Show only what the answer cites, not everything that was retrieved.
    cited_nums = extract_cited_source_numbers(answer, num_chunks=len(chunks))
    sources = [s for s in retrieved_sources if s["source_num"] in cited_nums]

    return {
        "answer":     answer,
        "sources":    sources,
        "retrieved_sources": retrieved_sources,
        "query":      query,
        "num_chunks": len(chunks),
        "model":      model,
        "llm":        llm.stats(),
    }