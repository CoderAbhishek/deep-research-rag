from src.ingestion.loader import load_documents_from_directory
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_and_store
import chromadb

# ── Step 1: Load ──────────────────────────────────────────────
print("=" * 50)
print("STEP 1: LOADING DOCUMENTS")
print("=" * 50)
pages = load_documents_from_directory("data/documents")

# ── Step 2: Chunk ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: CHUNKING")
print("=" * 50)
chunks = chunk_documents(pages, chunk_size=512, chunk_overlap=64)
print(f"\nResult: {len(pages)} pages → {len(chunks)} chunks")
print(f"Average chunks per page: {len(chunks) / len(pages):.1f}")

sample = chunks[100]
print(f"\nSample chunk (index 100):")
print(f"  Characters: {len(sample['text'])}")
print(f"  Page: {sample['metadata']['page_number']} of {sample['metadata']['total_pages']}")
print(f"  Chunk: {sample['metadata']['chunk_index']} of {sample['metadata']['total_chunks_in_page']}")
print(f"  Text preview: {sample['text'][:300]}")

# ── Step 3: Embed and store ───────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: EMBEDDING AND STORING")
print("=" * 50)
collection = embed_and_store(chunks)

# ── Step 4: Test retrieval ────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4: TEST RETRIEVAL")
print("=" * 50)

query = "What is Infosys revenue from North America?"
print(f"\nQuery: {query}\n")

results = collection.query(
    query_texts=[query],
    n_results=3,
)

for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"Result {i + 1}:")
    print(f"  File: {meta['file_name']}")
    print(f"  Page: {meta['page_number']}")
    print(f"  Chunk: {meta['chunk_index']}")
    print(f"  Text: {doc[:300]}")
    print()