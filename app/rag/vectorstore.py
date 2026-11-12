"""ChromaDB persistent client and collection wrapper."""
from __future__ import annotations

import contextlib

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name="refund_policies",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def reset_collection() -> None:
    """Delete and recreate collection (for testing/re-ingestion)."""
    global _collection
    client = get_client()
    with contextlib.suppress(Exception):
        client.delete_collection("refund_policies")
    _collection = client.get_or_create_collection(
        name="refund_policies",
        metadata={"hnsw:space": "cosine"},
    )
