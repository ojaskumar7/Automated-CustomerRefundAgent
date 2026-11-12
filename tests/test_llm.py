"""Phase 3 LLM advisor tests (stub mode for CI)."""
from __future__ import annotations

from app.llm.groq_client import LLMUnderstanding, get_llm_client, understand_grievance
from app.llm.stub import StubGroqClient, understand_grievance_stub


def test_stub_client_categorizes_defective() -> None:
    client = StubGroqClient()
    result = client.understand("my screen is cracked and broken", [])
    assert isinstance(result, LLMUnderstanding)
    assert result.extracted_reason == "defective_product"
    assert result.category == "defective_product"
    assert result.confidence >= 0.8


def test_stub_client_categorizes_late_delivery() -> None:
    client = StubGroqClient()
    result = client.understand("delivery was very late and delayed", [])
    assert result.extracted_reason == "late_delivery"
    assert result.category == "late_delivery"


def test_stub_client_categorizes_billing_error() -> None:
    client = StubGroqClient()
    result = client.understand("I was charged twice for the same order", [])
    assert result.extracted_reason == "billing_error"
    assert result.category == "billing_error"


def test_stub_client_categorizes_change_of_mind() -> None:
    client = StubGroqClient()
    result = client.understand("I changed my mind and want to return it", [])
    assert result.extracted_reason == "change_of_mind"
    assert result.category == "change_of_mind"


def test_stub_client_categorizes_not_as_described() -> None:
    client = StubGroqClient()
    result = client.understand("the color is wrong, not as described", [])
    assert result.extracted_reason == "not_as_described"
    assert result.category == "not_as_described"


def test_stub_client_extracts_amount() -> None:
    client = StubGroqClient()
    result = client.understand("I want a refund of $49.99 for the broken item", [])
    assert result.requested_amount == 49.99


def test_stub_client_no_amount_returns_none() -> None:
    client = StubGroqClient()
    result = client.understand("item is broken", [])
    assert result.requested_amount is None


def test_understand_grievance_stub_convenience() -> None:
    result = understand_grievance_stub("broken screen", [])
    assert result.extracted_reason == "defective_product"


def test_get_llm_client_returns_stub_when_no_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_STUB", raising=False)
    client = get_llm_client()
    assert isinstance(client, StubGroqClient)


def test_get_llm_client_returns_stub_when_env_set(monkeypatch) -> None:
    monkeypatch.setenv("LLM_STUB", "1")
    client = get_llm_client()
    assert isinstance(client, StubGroqClient)


def test_understand_grievance_uses_stub_by_default(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_STUB", raising=False)
    result = understand_grievance("late delivery", [])
    assert result.extracted_reason == "late_delivery"
