"""
output_checks.py - flag figures in an answer that do not appear in the context (DR-24, partial).

The pipeline's main promise is that numbers come from the documents. This
check makes that verifiable per answer: every number in the answer is looked
up in the text the model was given. A number that is not there was computed,
converted, rounded or invented by the model (for example the earlier q31
answer that added up a "total" the report never states).

It FLAGS, it does not block: a converted unit can be a correct answer, and
blocking would add false refusals (already the main failure, DR-35). The flag
is returned to the caller, logged as a WARNING, and recorded in evaluation.

Matching rules:
  - [SOURCE n] labels are removed first, so citation numbers are not checked.
  - Only numbers with at least 2 digits are checked ("5 segments" is not).
  - Grouping commas are ignored on both sides, so Indian (1,00,167) and
    international (100,167) grouping match each other. Spaces never join
    digits ("March 31, 2026" is 31 and 2026).
  - A match must be a whole number in the context, not part of a longer one
    ("167" does not match inside "1,00,167").
Not covered: numbers written as words, a figure that is in the context but
belongs to a different row or year (q18's distractor), wrong citations.
"""

import re
from typing import Iterable, List

_CITATION = re.compile(r"SOURCE\s*\d+", re.IGNORECASE)
# A number: digits, optionally grouped with commas directly between digits, optional decimals.
# Spaces never join digits: "March 31, 2026" is 31 and 2026, not 312026.
_NUMBER = re.compile(r"\d(?:[\d,]*\d)?(?:\.\d+)?")


def _canonical(raw: str) -> str:
    """Drop grouping commas: '1,00,167' -> '100167'."""
    return raw.replace(",", "")


def extract_figures(answer: str) -> List[str]:
    """Numbers in the answer (canonical form, in order, unique), ignoring citation labels."""
    text = _CITATION.sub(" ", answer or "")
    seen = {}
    for m in _NUMBER.finditer(text):
        fig = _canonical(m.group())
        if sum(ch.isdigit() for ch in fig) >= 2:
            seen.setdefault(fig, None)
    return list(seen)


def _context_numbers(texts: Iterable[str]) -> set:
    found = set()
    for t in texts:
        for m in _NUMBER.finditer(t or ""):
            found.add(_canonical(m.group()))
    return found


def unsupported_figures(answer: str, context_texts: Iterable[str]) -> List[str]:
    """Figures in the answer that appear nowhere in the context, as whole numbers."""
    available = _context_numbers(context_texts)
    return [f for f in extract_figures(answer) if f not in available]
