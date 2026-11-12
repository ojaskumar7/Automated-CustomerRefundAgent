"""Phase 4 Eligibility engine tests."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.config import settings
from app.db import init_db
from app.eligibility import (
    auto_approve_decision,
    check_already_refunded,
    check_repeat_limit,
    check_time_window,
    evaluate_eligibility,
    match_policy_category,
)
from app.llm.groq_client import LLMUnderstanding


def _now_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "orders.db"
    monkeypatch.setattr(settings, "db_path", db_path)
    monkeypatch.setattr(settings, "refund_auto_approve_threshold", 50.0)
    monkeypatch.setattr(settings, "max_refunds_30d", 3)
    monkeypatch.setattr(settings, "return_window_days", 30)
    from app.db import close_db
    close_db()
    init_db()
    return db_path


def _seed_order(db_path: Path, order_id: str, customer_id: str, amount: float,
                refunded: float = 0.0, days_ago: int = 5) -> None:
    created = (_now_utc() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO orders (id, customer_id, amount, currency, "
            "status, created_at, refunded_amount) VALUES (?,?,?,?,?,?,?)",
            (order_id, customer_id, amount, "USD", "delivered", created, refunded),
        )
        conn.commit()


def _seed_refund(db_path: Path, grievance_id: str, order_id: str, customer_id: str,
                 amount: float, days_ago: int = 2) -> None:
    issued = (_now_utc() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO grievances (id, order_id, customer_id, "
            "grievance_text, status) VALUES (?,?,?,?,?)",
            (grievance_id, order_id, customer_id, "test", "refunded"),
        )
        conn.execute(
            "INSERT INTO refunds (id, grievance_id, order_id, amount, "
            "status, simulated_ref, issued_at) VALUES (?,?,?,?,?,?,?)",
            (f"ref-{grievance_id}", grievance_id, order_id, amount,
             "simulated_issued", f"sim-{grievance_id}", issued),
        )
        conn.commit()


def _make_llm(category: str = "defective_product",
              amount: float | None = 49.99, confidence: float = 0.9) -> LLMUnderstanding:
    return LLMUnderstanding(
        extracted_reason=category,
        requested_amount=amount,
        category=category,
        confidence=confidence,
        reasoning="test",
    )


def test_check_time_window_within(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0, days_ago=5)
    recent = (_now_utc() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
    ok, msg = check_time_window(recent)
    assert ok


def test_check_time_window_outside(tmp_db: Path) -> None:
    old_date = (_now_utc() - timedelta(days=45)).strftime("%Y-%m-%d %H:%M:%S")
    ok, msg = check_time_window(old_date, return_window_days=30)
    assert not ok
    assert "outside_return_window" in msg


def test_check_already_refunded_partial(tmp_db: Path) -> None:
    order = {"amount": 100.0, "refunded_amount": 30.0}
    ok, msg, max_refund = check_already_refunded(order, None)
    assert ok
    assert max_refund == 70.0


def test_check_already_refunded_capped_by_requested(tmp_db: Path) -> None:
    order = {"amount": 100.0, "refunded_amount": 30.0}
    ok, msg, max_refund = check_already_refunded(order, 50.0)
    assert ok
    assert max_refund == 50.0  # min(70, 50)


def test_check_already_refunded_fully_refunded(tmp_db: Path) -> None:
    order = {"amount": 100.0, "refunded_amount": 100.0}
    ok, msg, max_refund = check_already_refunded(order, None)
    assert not ok
    assert msg == "fully_refunded"
    assert max_refund == 0.0


def test_check_repeat_limit_under(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0)
    _seed_refund(tmp_db, "g1", "o1", "c1", 30.0, days_ago=5)
    _seed_refund(tmp_db, "g2", "o1", "c1", 20.0, days_ago=10)
    ok, msg = check_repeat_limit("c1")
    assert ok


def test_check_repeat_limit_exceeded(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0)
    for i in range(3):
        _seed_refund(tmp_db, f"g{i}", "o1", "c1", 10.0, days_ago=i + 1)
    ok, msg = check_repeat_limit("c1")
    assert not ok
    assert "repeat_limit_exceeded" in msg


def test_match_policy_category_matches(tmp_db: Path) -> None:
    cat, boost = match_policy_category("defective_product", "my screen is cracked")
    assert cat == "defective_product"
    assert boost == 0.1


def test_match_policy_category_fallback(tmp_db: Path) -> None:
    cat, boost = match_policy_category("other", "something completely unrelated xyz")
    # Should fall back to top retrieved category
    valid_cats = (
        "defective_product", "not_as_described", "late_delivery",
        "billing_error", "change_of_mind", "other",
    )
    assert cat in valid_cats


def test_evaluate_eligibility_happy_path(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0, days_ago=5)
    llm = _make_llm("defective_product", 49.99, 0.9)
    result = evaluate_eligibility("o1", "c1", "screen cracked", llm)
    assert result.eligible
    assert result.reason == "eligible"
    assert result.max_refundable_amount == 49.99
    assert result.category == "defective_product"


def test_evaluate_eligibility_order_not_found(tmp_db: Path) -> None:
    llm = _make_llm()
    result = evaluate_eligibility("nonexistent", "c1", "broken", llm)
    assert not result.eligible
    assert result.reason == "order_not_found"


def test_evaluate_eligibility_outside_window(tmp_db: Path) -> None:
    old_date = (_now_utc() - timedelta(days=45)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(str(tmp_db)) as conn:
        conn.execute(
            "INSERT INTO orders (id, customer_id, amount, currency, "
            "status, created_at) VALUES (?,?,?,?,?,?)",
            ("o1", "c1", 100.0, "USD", "delivered", old_date),
        )
        conn.commit()
    llm = _make_llm()
    result = evaluate_eligibility("o1", "c1", "broken", llm)
    assert not result.eligible
    assert "outside_return_window" in result.reason


def test_evaluate_eligibility_fully_refunded(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0, refunded=100.0, days_ago=5)
    llm = _make_llm()
    result = evaluate_eligibility("o1", "c1", "broken", llm)
    assert not result.eligible
    assert result.reason == "fully_refunded"


def test_evaluate_eligibility_repeat_limit(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0, days_ago=5)
    for i in range(3):
        _seed_refund(tmp_db, f"g{i}", "o1", "c1", 10.0, days_ago=i + 1)
    llm = _make_llm()
    result = evaluate_eligibility("o1", "c1", "broken", llm)
    assert not result.eligible
    assert "repeat_limit_exceeded" in result.reason


def test_evaluate_eligibility_amount_capped_at_remaining(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0, refunded=80.0, days_ago=5)
    llm = _make_llm(amount=100.0)  # requests more than remaining
    result = evaluate_eligibility("o1", "c1", "broken", llm)
    assert result.eligible
    assert result.max_refundable_amount == 20.0  # capped at remaining


def test_auto_approve_under_threshold() -> None:
    auto, reason = auto_approve_decision(49.99)
    assert auto
    assert reason == "auto_approved"


def test_auto_approve_over_threshold() -> None:
    auto, reason = auto_approve_decision(75.0)
    assert not auto
    assert reason == "escalated_above_threshold"


def test_auto_approve_at_threshold() -> None:
    auto, reason = auto_approve_decision(50.0)
    assert auto
    assert reason == "auto_approved"
