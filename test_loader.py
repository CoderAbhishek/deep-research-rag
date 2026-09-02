"""
Quick test for the document loader.
Run from the project root: python test_loader.py
"""
from src.ingestion.loader import load_documents_from_directory
from collections import Counter

DOCUMENTS_DIR = "data/documents"

# Load all PDFs
pages = load_documents_from_directory(DOCUMENTS_DIR)

# --- Inspect the first page ---
first = pages[0]
print("\n" + "="*60)
print("FIRST PAGE SAMPLE")
print("="*60)
print(f"File       : {first['metadata']['file_name']}")
print(f"Page       : {first['metadata']['page_number']} of {first['metadata']['total_pages']}")
print(f"Text length: {len(first['text'])} characters")
print(f"\nFirst 600 characters of text:\n")
print(first['text'][:600])

# --- Pages per document ---
print("\n" + "="*60)
print("PAGES PER DOCUMENT")
print("="*60)
file_names = [p['metadata']['file_name'] for p in pages]
counts = Counter(file_names)
for file_name, count in counts.items():
    print(f"  {file_name}: {count} pages")

# --- Text length statistics ---
print("\n" + "="*60)
print("TEXT LENGTH STATISTICS (characters per page)")
print("="*60)
lengths = [len(p['text']) for p in pages]
print(f"  Minimum : {min(lengths)}")
print(f"  Maximum : {max(lengths)}")
print(f"  Average : {sum(lengths) // len(lengths)}")

# --- Sample a middle page from the annual report ---
print("\n" + "="*60)
print("SAMPLE: PAGE 50 FROM ANNUAL REPORT (if it exists)")
print("="*60)
ar_pages = [p for p in pages if 'infosys' in p['metadata']['file_name'].lower()]
if len(ar_pages) >= 50:
    sample = ar_pages[49]  # 0-indexed, so index 49 = page 50
    print(f"Page {sample['metadata']['page_number']}:")
    print(sample['text'][:800])