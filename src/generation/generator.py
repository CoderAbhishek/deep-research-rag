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

import os
from typing import List, Dict, Any
from groq import Groq

# ─── Constants ────────────────────────────────────────────────────────────────

# Module-level constant so every function in this module uses the same model.
# Change this one line to switch models globally.
# We use compound-mini because it is available on this Groq account.
# For production: groq/llama-3.1-70b-versatile gives better instruction
# following at higher cost.
DEFAULT_MODEL = "compound-beta-mini"


# ─── System prompt ────────────────────────────────────────────────────────────

# The system prompt sets the model's role, constraints, and output rules.
# It is sent as the "system" role message — the model treats this as an
# authoritative instruction that applies to everything in the conversation.
#
# Key choices in this prompt:
#
# "ONLY the context provided" — grounding instruction. Without this,
# the model will answer from parametric knowledge (its training data),
# which may be stale or wrong.
#
# "Do not use any knowledge not present in the context" — this restatement
# is intentional. Prompt engineering research shows that repeating a
# constraint in different words improves adherence. One phrasing may slip
# past the model's attention; two phrasings make the constraint harder to
# ignore.
#
# "Every factual claim must be attributed to a [SOURCE X] label" — forces
# the model to think about which chunk supports each claim. This has a
# secondary benefit: the model is less likely to hallucinate a claim it
# cannot attribute.
#
# "say exactly: 'The provided documents...'" — giving the model a specific
# fallback phrase to use prevents it from generating a vague hedge like
# "I'm not sure about this, but..." while still answering. We want a
# clean binary: answer or refuse.
#
# "Do not infer or calculate" — prevents the model from doing arithmetic
# on retrieved figures (e.g., calculating percentages from raw numbers).
# We only want facts that appear verbatim in the text.

SYSTEM_PROMPT = """You are a research analyst assistant. You answer questions about business documents.

STRICT RULES — follow these exactly:
1. Answer using ONLY the context documents provided below. Do not use any knowledge not present in the context.
2. Cite your sources. Every factual claim must be attributed to a [SOURCE X] label. Example: "North America revenue was ₹1,00,167 crore [SOURCE 1]."
3. If the context does not contain sufficient information to answer, respond with exactly: "The provided documents do not contain sufficient information to answer this question."
4. Copy numbers, names, and figures exactly as they appear in the sources. Do not round, convert, or paraphrase figures.
5. Do not infer, calculate, or extrapolate beyond what the sources explicitly state."""


# ─── Context formatting ───────────────────────────────────────────────────────

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
        # Extract metadata with safe defaults.
        meta = chunk["metadata"]
        file_name   = meta.get("file_name",   "unknown")
        page_number = meta.get("page_number", "?")

        # Build the label line and combine with chunk text.
        label = f"[SOURCE {i}] File: {file_name} | Page: {page_number}"
        parts.append(f"{label}\n{chunk['text']}")

    # Blank line between chunks — makes them visually distinct in the prompt.
    return "\n\n".join(parts)


# ─── Answer generation ────────────────────────────────────────────────────────

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
        model:  Groq model identifier (without the "groq/" prefix that
                the LiteLLM-style wrapper uses). Default is compound-beta-mini.

    Returns:
        A dict with:
            'answer'     — the LLM's answer string
            'sources'    — list of (file_name, page_number) tuples,
                           one per chunk, in the same order as the
                           [SOURCE X] labels in the context
            'query'      — the original query (for logging/tracing)
            'num_chunks' — number of chunks fed to the model
            'model'      — the model used (for logging/tracing)

    Why we return metadata alongside the answer:
    In production, you need to log what model answered a query, with how
    many chunks, from which sources. Without this, debugging a wrong answer
    requires reconstructing the entire query session from logs.
    """
    # Lazy client creation: Groq client is created inside the function,
    # not at module level. This means importing generator.py does not
    # fail if GROQ_API_KEY is absent (e.g., during unit testing with
    # mocked environment). The error is raised only when generate_answer()
    # is actually called.
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Build the context string from retrieved chunks.
    context = format_context(chunks)

    # User message: context first, then question.
    # The "context first" ordering matters: LLMs tend to give more weight
    # to information near the beginning of the context window. Placing
    # the retrieved evidence before the question means the model reads
    # the evidence before seeing what it is being asked — priming it to
    # anchor its answer in the provided text rather than its priors.
    user_message = f"""Context documents:

{context}

---

Question: {query}

Answer the question using only the context documents above. Cite your sources using the [SOURCE X] labels."""

    # API call.
    # temperature=0.1: near-deterministic. We want the model to reproduce
    # figures exactly, not rephrase them. Higher temperature = more creative
    # = more risk of paraphrasing "₹1,00,167 crore" as "approximately
    # ₹1 lakh crore" or hallucinating adjacent figures.
    # max_tokens=512: enough for a detailed cited answer (typically 150-300
    # tokens for our query type). Capping prevents runaway generation.
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.1,
        max_tokens=512,
    )

    # Extract the answer text from the response object.
    # response.choices[0].message.content is the standard OpenAI-compatible
    # response format that Groq follows.
    answer = response.choices[0].message.content

    # Build source list in [SOURCE X] order so the caller can display
    # or log which documents contributed to the answer.
    sources = [
        {
            "source_num":   i + 1,
            "file_name":    chunk["metadata"].get("file_name",   "unknown"),
            "page_number":  chunk["metadata"].get("page_number", "?"),
        }
        for i, chunk in enumerate(chunks)
    ]

    return {
        "answer":     answer,
        "sources":    sources,
        "query":      query,
        "num_chunks": len(chunks),
        "model":      model,
    }