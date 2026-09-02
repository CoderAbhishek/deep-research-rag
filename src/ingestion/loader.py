import fitz                          # PyMuPDF — the library is called "fitz" historically
from pathlib import Path             # Modern Python path handling (safer than raw strings)
from typing import List, Dict, Any   # Type hints — make the code self-documenting


def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a single PDF and return its pages as a list of dictionaries.

    Each dictionary has two keys:
        "text"     : str  — the raw extracted text of that page
        "metadata" : dict — source, file_name, page_number, total_pages

    Args:
        file_path: Path to the PDF file (string)

    Returns:
        List of page dictionaries (empty pages are skipped)
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    pages = []

    with fitz.open(str(path)) as doc:
        total_pages = len(doc)

        for page_index in range(total_pages):
            page = doc[page_index]
            text = page.get_text("text")
            text = text.strip()

            if not text:
                continue

            page_doc = {
                "text": text,
                "metadata": {
                    "source":       str(path.resolve()),
                    "file_name":    path.name,
                    "page_number":  page_index + 1,
                    "total_pages":  total_pages,
                }
            }

            pages.append(page_doc)

    return pages


def load_documents_from_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    Load all PDF files in a directory and return all pages as one flat list.

    Args:
        directory_path: Path to a folder containing PDF files

    Returns:
        Combined list of page dictionaries from all PDFs, in file order
    """
    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    if not directory.is_dir():
        raise ValueError(f"Expected a directory, got: {directory_path}")

    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {directory_path}")

    print(f"Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    all_pages = []

    for pdf_path in pdf_files:
        print(f"\nLoading: {pdf_path.name}")
        pages = load_pdf(str(pdf_path))
        print(f"  Extracted {len(pages)} non-empty pages")
        all_pages.extend(pages)

    print(f"\nTotal pages across all documents: {len(all_pages)}")
    return all_pages