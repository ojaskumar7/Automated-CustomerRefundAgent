"""Sentence Transformer encoder wrapper."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def get_encoder() -> SentenceTransformer:
    """Return cached encoder instance (lazy init)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def encode(texts: list[str]) -> np.ndarray:
    """Encode texts to normalized embeddings (cosine-ready)."""
    return get_encoder().encode(texts, normalize_embeddings=True)
