from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any


def chunk_documents(
    pages: List[Dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Dict[str, Any]]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]

        split_texts = splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(split_texts):
            chunk_doc = {
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_index,
                    "total_chunks_in_page": len(split_texts),
                }
            }
            chunks.append(chunk_doc)

    return chunks