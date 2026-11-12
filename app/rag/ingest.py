"""Policy ingestion: load files, chunk, embed, upsert to Chroma."""
from __future__ import annotations

from typing import Any

from app.config import settings
from app.rag.embeddings import get_encoder
from app.rag.vectorstore import get_collection


def ingest_policies() -> int:
    """Load all *.txt from policies_dir, chunk by paragraph, embed, upsert.
    Returns number of chunks ingested.
    """
    collection = get_collection()
    encoder = get_encoder()

    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    for path in sorted(settings.policies_dir.glob("*.txt")):
        category = path.stem
        text = path.read_text(encoding="utf-8")
        # Paragraph-based chunking: split on blank line
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": path.name, "category": category, "chunk": i})
            ids.append(f"{category}-{i}")

    if not documents:
        return 0

    embeddings = encoder.encode(documents, normalize_embeddings=True).tolist()
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)  # type: ignore[arg-type]
    return len(documents)


def reingest_policies() -> int:
    """Force full re-ingestion by resetting collection first."""
    from app.rag.vectorstore import reset_collection
    reset_collection()
    return ingest_policies()
