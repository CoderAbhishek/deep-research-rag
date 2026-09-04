"""
query.py — Query engineering: rewriting, multi-query generation, and HyDE.

All LLM calls use Groq (hosted open-weight model) via the groq Python client.
API key is loaded from the GROQ_API_KEY environment variable in .env.
Never hardcode credentials.
"""

import os
from typing import List
from groq import Groq
from dotenv import load_dotenv
DEFAULT_MODEL = "groq/compound-mini"

load_dotenv()


def _get_groq_client() -> Groq:
    """
    Instantiate the Groq client.
    Reads GROQ_API_KEY from environment (loaded from .env by load_dotenv()).
    Raises a clear error if the key is missing rather than a cryptic auth failure.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to your .env file: GROQ_API_KEY=your_key_here"
        )
    return Groq(api_key=api_key)


def _call_llm(client: Groq, system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Single LLM call via Groq. Returns the text content of the first choice.

    model: DEFAULT_MODEL — free tier, fast (~500ms), sufficient for query rewriting.
    For higher quality at the cost of latency: "llama-3.3-70b-versatile".
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,       # Low temperature: we want focused, deterministic rewrites
        max_tokens=512,        # Query rewrites are short; no need for long output
    )
    return response.choices[0].message.content.strip()


def rewrite_query(query: str, model: str = DEFAULT_MODEL) -> str:
    """
    Rewrite a conversational user query into a retrieval-optimised form.

    The rewritten query uses the vocabulary and phrasing likely to appear
    in the source documents (annual reports, research reports) rather than
    the natural language of a question.

    Returns a single string: the rewritten query.
    """
    client = _get_groq_client()
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
    return _call_llm(client, system_prompt, user_prompt, model)


def generate_multi_query(query: str, n: int = 3, model: str = DEFAULT_MODEL) -> List[str]:
    """
    Generate N alternative phrasings of the user's query for multi-query retrieval.

    Each variant uses different vocabulary or framing. Running retrieval for each
    and taking the union increases the chance that the relevant chunk is retrieved
    by at least one variant.

    Returns a list of N strings (one per variant).
    """
    client = _get_groq_client()
    system_prompt = (
        "You are a retrieval query diversifier for a financial document RAG system. "
        "Generate alternative phrasings of a user's question. Each variant should use "
        "different vocabulary — synonyms, different term order, different level of specificity. "
        f"Generate exactly {n} variants, one per line. "
        "Output ONLY the variants, numbered 1. 2. 3. No other text."
    )
    user_prompt = f"Original question: {query}\n\n{n} alternative phrasings:"
    raw = _call_llm(client, system_prompt, user_prompt, model)

    # Parse the numbered list into a clean Python list
    variants = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading numbers and punctuation: "1. Revenue..." → "Revenue..."
        if line[0].isdigit():
            # Remove "1." or "1:" or "1) " at the start
            line = line.lstrip("0123456789").lstrip(".):- ").strip()
        if line:
            variants.append(line)

    return variants[:n]   # Safety cap — return at most n variants


def generate_hyde(query: str, model: str = DEFAULT_MODEL) -> str:
    """
    Generate a Hypothetical Document Embedding (HyDE) passage.

    HyDE (Gao et al., 2022): ask the LLM to write a short passage that WOULD answer
    the question if it were from the source document. The passage uses document-native
    vocabulary. Embedding the passage instead of the query closes the query-document
    embedding gap.

    Returns a single string: the hypothetical document passage (~100-150 words).
    The passage may be factually wrong — that is expected and acceptable.
    It only needs to use the right vocabulary to improve retrieval.
    """
    client = _get_groq_client()
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
    return _call_llm(client, system_prompt, user_prompt, model)