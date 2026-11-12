"""Retrieval helper: query Chroma for relevant policy chunks."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.rag.embeddings import get_encoder
from app.rag.vectorstore import get_collection


def retrieve(query: str, k: int = 5, category: str | None = None) -> list[dict]:
    """Return top-k chunks for query.
    Each result: {text, score (cosine similarity 0-1), metadata}.
    """
    collection = get_collection()
    query_emb = get_encoder().encode([query], normalize_embeddings=True).tolist()

    where = {"category": category} if category else None
    results = collection.query(
        query_embeddings=query_emb,
        n_results=k,
        where=where,  # type: ignore[arg-type]
        include=["documents", "metadatas", "distances"],
    )

    out: list[dict] = []
    docs: list[str] = (results.get("documents") or [[]])[0]
    metas: list[Mapping[str, Any]] = (results.get("metadatas") or [[]])[0]
    dists: list[float] = (results.get("distances") or [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists, strict=False):
        # Chroma returns cosine distance (0=identical, 2=opposite)
        # Convert to similarity in [-1, 1] where higher = better
        score = 1.0 - dist
        out.append({"text": doc, "score": score, "metadata": meta})
    return out
