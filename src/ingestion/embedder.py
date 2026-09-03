from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def get_chroma_client(persist_dir: str = "./chroma_db") -> chromadb.PersistentClient:
    client = chromadb.PersistentClient(path=persist_dir)
    return client


def embed_and_store(
    chunks: List[Dict[str, Any]],
    collection_name: str = "documents",
    model_name: str = "all-MiniLM-L6-v2",
    persist_dir: str = "./chroma_db",
) -> chromadb.Collection:

    model = get_embedding_model(model_name)
    client = get_chroma_client(persist_dir)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [
        f"{m['file_name']}_p{m['page_number']}_c{m['chunk_index']}"
        for m in metadatas
    ]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids,
    )

    print(f"Stored {len(texts)} chunks in collection '{collection_name}'")
    return collection