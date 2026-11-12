"""Deterministic LLM stub for CI/testing without Groq API key."""
from __future__ import annotations

from app.llm.groq_client import LLMUnderstanding

# Keyword-based categorization for deterministic CI mode
_KEYWORD_MAP: dict[str, list[str]] = {
    "defective_product": [
        "broken",
        "defect",
        "cracked",
        "damaged",
        "not working",
        "malfunction",
        "faulty",
    ],
    "not_as_described": [
        "wrong",
        "different",
        "not as described",
        "misleading",
        "color",
        "size",
        "material",
    ],
    "late_delivery": [
        "late",
        "delay",
        "delayed",
        "shipping",
        "delivery",
        "arrived late",
        "slow",
    ],
    "billing_error": [
        "charged",
        "charge",
        "billing",
        "duplicate",
        "wrong amount",
        "overcharge",
        "refund my money",
    ],
    "change_of_mind": [
        "changed my mind",
        "don't want",
        "no longer need",
        "return",
        "change of mind",
    ],
}


def _categorize(text: str) -> tuple[str, float]:
    text_lower = text.lower()
    for category, keywords in _KEYWORD_MAP.items():
        for kw in keywords:
            if kw in text_lower:
                return category, 0.85
    return "other", 0.3


def _extract_amount(text: str) -> float | None:
    import re
    # Match $XX.XX or $XX or XX dollars
    match = re.search(r"\$?(\d+(?:\.\d{2})?)\s*(?:dollars?|USD)?", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


class StubGroqClient:
    """Deterministic replacement for GroqClient in CI/testing."""

    def understand(
        self,
        grievance_text: str,
        policy_chunks: list[dict],
        max_retries: int = 3,
    ) -> LLMUnderstanding:
        category, confidence = _categorize(grievance_text)
        amount = _extract_amount(grievance_text)
        return LLMUnderstanding(
            extracted_reason=category,
            requested_amount=amount,
            category=category,
            confidence=confidence,
            reasoning=f"Keyword match: '{category}' detected in grievance text",
        )


def get_stub_client() -> StubGroqClient:
    return StubGroqClient()


def understand_grievance_stub(
    grievance_text: str,
    policy_chunks: list[dict],
) -> LLMUnderstanding:
    """Convenience function using stub client."""
    return get_stub_client().understand(grievance_text, policy_chunks)
