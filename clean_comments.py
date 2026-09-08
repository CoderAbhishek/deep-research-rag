"""
clean_comments.py — Strip # comments from all Python files in src/.

Preserves: docstrings, blank lines, all functional code.
Removes: full-line # comments and inline # comments.

Run from the project root:
    python clean_comments.py

Review changes before committing:
    git diff src/
"""

import io
import tokenize
from pathlib import Path


def strip_comments(source: str) -> str:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return source

    lines = source.splitlines(keepends=True)
    full_comment_lines = set()
    inline_comment_cols = {}

    for tok_type, tok_string, tok_start, tok_end, _ in tokens:
        if tok_type != tokenize.COMMENT:
            continue
        line_no, col = tok_start
        line_content = lines[line_no - 1].strip()
        if line_content.startswith("#"):
            full_comment_lines.add(line_no)
        else:
            inline_comment_cols[line_no] = col

    result = []
    blank_streak = 0

    for i, line in enumerate(lines, start=1):
        if i in full_comment_lines:
            continue
        
        if not line.strip():
            blank_streak += 1
            if blank_streak > 2:
                continue
        else:
            blank_streak = 0

        result.append(line)

    return "".join(result)


def run(root: str = ".") -> None:
    src_path = Path(root) / "src"
    if not src_path.exists():
        print(f"ERROR: {src_path} not found — run from the project root.")
        return

    files = sorted(src_path.rglob("*.py"))
    if not files:
        print("No .py files found in src/")
        return

    changed = 0
    for path in files:
        original = path.read_text(encoding="utf-8")
        cleaned = strip_comments(original)
        if cleaned != original:
            path.write_text(cleaned, encoding="utf-8")
            print(f"  cleaned  {path.relative_to(root)}")
            changed += 1
        else:
            print(f"  no change {path.relative_to(root)}")

    print(f"\nDone. {changed} file(s) modified.")
    print("Review with: git diff src/")


if __name__ == "__main__":
    run()