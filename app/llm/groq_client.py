"""Groq LLM client for structured grievance understanding (advisor only)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from groq import Groq

from app.config import settings


@dataclass
class LLMUnderstanding:
    extracted_reason: str
    requested_amount: float | None
    category: str
    confidence: float
    reasoning: str


SYSTEM_PROMPT = """You are an expert e-commerce refund policy analyst. Your job is to understand a customer's grievance and extract structured information. You NEVER decide eligibility or approve refunds — you only provide structured analysis to a deterministic Python engine.

Output ONLY valid JSON with this exact schema:
{
  "extracted_reason": "brief reason category (defective_product, not_as_described, late_delivery, billing_error, change_of_mind, other)",
  "requested_amount": float or null,
  "category": "policy category that best matches (defective_product, not_as_described, late_delivery, billing_error, change_of_mind, other)",
  "confidence": float between 0.0 and 1.0,
  "reasoning": "one sentence explaining your categorization"
}

Rules:
- extracted_reason must be one of the listed categories
- requested_amount: parse any dollar amount mentioned; null if not mentioned
- category: match the closest policy category
- confidence: how certain you are (0.0-1.0)
- reasoning: concise, one sentence"""  # noqa: E501


USER_PROMPT_TEMPLATE = """Customer grievance: {grievance_text}

Relevant refund policy context:
{policy_context}

Extract structured understanding as JSON."""


class GroqClient:
    def __init__(self) -> None:
        api_key = settings.groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self._client = Groq(api_key=api_key)
        self._model = settings.groq_model

    def understand(
        self,
        grievance_text: str,
        policy_chunks: list[dict],
        max_retries: int = 3,
    ) -> LLMUnderstanding:
        """Call Groq with retry/backoff. Returns structured understanding."""
        policy_context = "\n\n".join(
            f"[Policy: {c['metadata']['category']}] {c['text']}"
            for c in policy_chunks[:3]
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                grievance_text=grievance_text,
                policy_context=policy_context or "(no policy context available)",
            )},
        ]

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(  # type: ignore[call-overload]
                    model=self._model,
                    messages=messages,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    max_tokens=300,
                )
                content = resp.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from Groq")
                data = json.loads(content)
                return LLMUnderstanding(
                    extracted_reason=data["extracted_reason"],
                    requested_amount=data.get("requested_amount"),
                    category=data["category"],
                    confidence=float(data["confidence"]),
                    reasoning=data["reasoning"],
                )
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s
                    continue
                raise RuntimeError(
                    f"Groq structured output failed after {max_retries} attempts: {last_exc}"
                ) from last_exc
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(
                    f"Groq API error after {max_retries} attempts: {last_exc}"
                ) from last_exc

        # Should not reach here
        raise RuntimeError(f"Groq failed: {last_exc}")


# Module-level singleton
_client: GroqClient | None = None


def get_groq_client() -> GroqClient:
    global _client
    if _client is None:
        _client = GroqClient()
    return _client


def get_llm_client(use_stub: bool | None = None):
    """Return GroqClient or StubGroqClient based on env or explicit flag.
    use_stub=None (default): auto-detect from LLM_STUB env var or missing API key.
    use_stub=True: force stub.
    use_stub=False: force real Groq (requires API key).
    """
    if use_stub is None:
        use_stub = os.environ.get("LLM_STUB", "").lower() in ("1", "true", "yes") or not (
            settings.groq_api_key or os.environ.get("GROQ_API_KEY")
        )
    if use_stub:
        from app.llm.stub import get_stub_client
        return get_stub_client()
    return get_groq_client()


def understand_grievance(
    grievance_text: str,
    policy_chunks: list[dict],
    use_stub: bool | None = None,
) -> LLMUnderstanding:
    """Convenience function using appropriate client."""
    return get_llm_client(use_stub).understand(grievance_text, policy_chunks)
