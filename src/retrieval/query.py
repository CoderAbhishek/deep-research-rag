"""
query.py — Query engineering: rewriting, multi-query generation, and HyDE.

All LLM calls go through the shared wrapper src/llm/groq_client.py, which
reads GROQ_API_KEY from the environment, traces each call in LangSmith and
logs model, latency, token usage, finish_reason and estimated cost.
Never hardcode credentials.
"""

from typing import List

from dotenv import load_dotenv

from src.llm.groq_client import LLMResult, chat

DEFAULT_MODEL = "openai/gpt-oss-20b"

# Token budget for query-side LLM calls (HyDE, rewriting, multi-query).
# gpt-oss-20b is a reasoning model: max_tokens caps hidden reasoning PLUS the
# visible output. At 512 with default reasoning, HyDE hit the limit on 14-15 of
# 24 baseline questions (DR-26). HyDE needs ~150 visible tokens and almost no
# reasoning, so: a higher ceiling (a cap, not a target) and low effort.
QUERY_MAX_TOKENS = 1024
QUERY_REASONING_EFFORT = "low"

load_dotenv()


def _call_llm(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL,
              purpose: str = "query") -> LLMResult:
    """
    Single LLM call through the shared Groq wrapper.

    Returns an LLMResult: .text is the stripped output; the other fields
    (tokens, latency, finish_reason, cost) are used for observability.
    purpose labels the call in logs and in the LangSmith trace.
    temperature=0.3: focused, near-deterministic rewrites.
    max_tokens / reasoning_effort: QUERY_MAX_TOKENS / QUERY_REASONING_EFFORT
    (see the note at the top of this module).
    """
    result = chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=0.3,
        max_tokens=QUERY_MAX_TOKENS,
        reasoning_effort=QUERY_REASONING_EFFORT,
        purpose=purpose,
    )
    result.text = result.text.strip()
    return result


def rewrite_query(query: str, model: str = DEFAULT_MODEL) -> str:
    """
    Rewrite a conversational user query into a retrieval-optimised form.

    The rewritten query uses the vocabulary and phrasing likely to appear
    in the source documents (annual reports, research reports) rather than
    the natural language of a question.

    Returns a single string: the rewritten query.
    """
    system_prompt = (
        "You are a retrieval query optimiser for a financial document RAG system. "
        "Your task is to rewrite a user's question into a concise search query that "
        "uses the exact vocabulary likely to appear in annual reports, financial filings, "
        "and management discussions. "
        "Remove question words (what, is, how, why). "
        "Include domain-specific terms: segment, geography, breakdown, FY, crore, revenue, etc. "
        "Output ONLY the rewritten query. No explanation, no preamble."
    )
    user_prompt = f"Original question: {query}\n\nRewritten search query:"
    return _call_llm(system_prompt, user_prompt, model, purpose="rewrite").text


def generate_multi_query(query: str, n: int = 3, model: str = DEFAULT_MODEL) -> List[str]:
    """
    Generate N alternative phrasings of the user's query for multi-query retrieval.

    Each variant uses different vocabulary or framing. Running retrieval for each
    and taking the union increases the chance that the relevant chunk is retrieved
    by at least one variant.

    Returns a list of N strings (one per variant).
    """
    system_prompt = (
        "You are a retrieval query diversifier for a financial document RAG system. "
        "Generate alternative phrasings of a user's question. Each variant should use "
        "different vocabulary — synonyms, different term order, different level of specificity. "
        f"Generate exactly {n} variants, one per line. "
        "Output ONLY the variants, numbered 1. 2. 3. No other text."
    )
    user_prompt = f"Original question: {query}\n\n{n} alternative phrasings:"
    raw = _call_llm(system_prompt, user_prompt, model, purpose="multi_query").text

    variants = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit():
            line = line.lstrip("0123456789").lstrip(".):- ").strip()
        if line:
            variants.append(line)

    return variants[:n]   # Safety cap — return at most n variants


def generate_hyde_result(query: str, model: str = DEFAULT_MODEL) -> LLMResult:
    """
    Generate a Hypothetical Document Embedding (HyDE) passage.

    HyDE (Gao et al., 2022): ask the LLM to write a short passage that WOULD answer
    the question if it were from the source document. The passage uses document-native
    vocabulary. Embedding the passage instead of the query closes the query-document
    embedding gap.

    Returns an LLMResult: .text is the hypothetical passage (~80-120 words);
    the other fields carry token usage, latency and finish_reason.
    The passage may be factually wrong — that is expected and acceptable.
    It only needs to use the right vocabulary to improve retrieval.
    """
    system_prompt = (
        "You are simulating a passage from an Indian corporate annual report. "
        "Write a short 80-120 word excerpt that would directly answer the user's question. "
        "Use formal financial language: segment, geography, crore, FY, year-on-year, growth, etc. "
        "Include plausible-sounding (but not necessarily accurate) figures. "
        "The goal is NOT factual accuracy — it is to produce text that uses the same "
        "vocabulary as the real answer in the document, to improve embedding-based retrieval. "
        "Output ONLY the passage. No preamble, no 'here is a passage'."
    )
    user_prompt = f"Question: {query}\n\nHypothetical annual report excerpt:"
    return _call_llm(system_prompt, user_prompt, model, purpose="hyde")


def generate_hyde(query: str, model: str = DEFAULT_MODEL) -> str:
    """Return only the HyDE passage text (kept for the experiment scripts)."""
    return generate_hyde_result(query, model).text