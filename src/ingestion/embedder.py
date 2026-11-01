import logging
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Model loaded. Embedding dimension: %d", model.get_sentence_embedding_dimension())
    return model


def get_chroma_client(persist_dir: str = "./chroma_db") -> chromadb.PersistentClient:
    # Telemetry off (DR-40), same as the query pipeline.
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
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

    logger.info("Embedding %d chunks...", len(texts))
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids,
    )

    logger.info("Stored %d chunks in collection %r", len(texts), collection_name)
    return collection