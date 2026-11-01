"""
input_checks.py - reject questions the pipeline should not spend LLM calls on (DR-23, partial).

Covered: empty or whitespace-only input, input over MAX_QUESTION_CHARS, and
input with no letters or digits at all. Each would cost two LLM calls and
return a refusal or noise.

Not covered (recorded as a known limitation): prompt injection, off-topic
detection beyond what the grounding prompt already refuses, abuse filtering.

MAX_QUESTION_CHARS: the longest evaluation question is 86 characters; 1,000
leaves room for detailed questions while bounding prompt size and cost.
"""

MAX_QUESTION_CHARS = 1000


class InvalidQuestion(ValueError):
    """Raised with a message that is safe to show to the user."""


def validate_question(question: str) -> str:
    """Return the question stripped of surrounding whitespace, or raise InvalidQuestion."""
    text = (question or "").strip()
    if not text:
        raise InvalidQuestion("Please enter a question.")
    if len(text) > MAX_QUESTION_CHARS:
        raise InvalidQuestion(f"Please keep the question under {MAX_QUESTION_CHARS} characters "
                              f"(this one has {len(text)}).")
    if not any(ch.isalnum() for ch in text):
        raise InvalidQuestion("Please enter a question in words.")
    return text
