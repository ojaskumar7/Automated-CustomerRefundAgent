"""Phase 5 Refund + Escalation tests."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.config import settings
from app.db import close_db, init_db
from app.escalation import get_escalation_queue, human_decide
from app.refund import deny_refund, execute_refund, process_grievance_full


def _now_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "orders.db"
    monkeypatch.setattr(settings, "db_path", db_path)
    monkeypatch.setattr(settings, "refund_auto_approve_threshold", 50.0)
    monkeypatch.setattr(settings, "max_refunds_30d", 3)
    monkeypatch.setattr(settings, "return_window_days", 30)
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


def _seed_grievance(db_path: Path, grievance_id: str, order_id: str, customer_id: str,
                    status: str = "escalated") -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO grievances (id, order_id, customer_id, grievance_text, status) "
            "VALUES (?,?,?,?,?)",
            (grievance_id, order_id, customer_id, "test grievance", status),
        )
        conn.commit()


def test_execute_refund_updates_all_tables(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0)
    _seed_grievance(tmp_db, "g1", "o1", "c1", "eligible")

    result = execute_refund("g1", "o1", "c1", 49.99)

    assert result.status == "simulated_issued"
    assert result.amount == 49.99
    assert result.refund_id.startswith("sim_")

    with sqlite3.connect(str(tmp_db)) as conn:
        conn.row_factory = sqlite3.Row
        refund = conn.execute("SELECT * FROM refunds WHERE id = ?", (result.refund_id,)).fetchone()
        assert refund["status"] == "simulated_issued"
        assert refund["amount"] == 49.99

        grievance = conn.execute("SELECT * FROM grievances WHERE id = ?", ("g1",)).fetchone()
        assert grievance["status"] == "refunded"

        order = conn.execute("SELECT * FROM orders WHERE id = ?", ("o1",)).fetchone()
        assert order["refunded_amount"] == 49.99


def test_execute_refund_human_override(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0)
    _seed_grievance(tmp_db, "g1", "o1", "c1", "escalated")

    result = execute_refund("g1", "o1", "c1", 75.0, human_override=True)

    with sqlite3.connect(str(tmp_db)) as conn:
        conn.row_factory = sqlite3.Row
        refund = conn.execute("SELECT * FROM refunds WHERE id = ?", (result.refund_id,)).fetchone()
        assert refund["human_override"] == 1


def test_deny_refund(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0)
    _seed_grievance(tmp_db, "g1", "o1", "c1", "eligible")

    result = deny_refund("g1", "outside_return_window")

    assert result.status == "denied"
    assert result.amount == 0.0

    with sqlite3.connect(str(tmp_db)) as conn:
        conn.row_factory = sqlite3.Row
        grievance = conn.execute("SELECT * FROM grievances WHERE id = ?", ("g1",)).fetchone()
        assert grievance["status"] == "rejected"


def test_process_grievance_full_auto_approve(tmp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_STUB", "1")
    _seed_order(tmp_db, "o1", "c1", 30.0)  # order amount under threshold

    result = process_grievance_full("o1", "c1", "broken item for $25", idempotency_key="g1")

    # order amount 30 < threshold 50 -> auto_approved, refund status is 'simulated_issued'
    assert result.status == "simulated_issued"
    assert result.amount == 25.0  # stub extracts $25


def test_process_grievance_full_escalates_over_threshold(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_STUB", "1")
    _seed_order(tmp_db, "o1", "c1", 100.0)

    result = process_grievance_full("o1", "c1", "screen is cracked", idempotency_key="g1")

    # LLM stub returns no amount, so max_refundable = 100 > 50 -> escalated
    assert result.status == "escalated"
    assert result.amount == 100.0


def test_process_grievance_full_idempotent(tmp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_STUB", "1")
    _seed_order(tmp_db, "o1", "c1", 30.0)  # under threshold

    r1 = process_grievance_full("o1", "c1", "broken item", idempotency_key="idem-1")
    r2 = process_grievance_full("o1", "c1", "broken item", idempotency_key="idem-1")

    assert r1.grievance_id == r2.grievance_id == "idem-1"
    assert r2.message == "already processed"


def test_get_escalation_queue(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0)
    _seed_grievance(tmp_db, "g1", "o1", "c1", "escalated")
    _seed_grievance(tmp_db, "g2", "o1", "c1", "eligible")  # not escalated

    queue = get_escalation_queue()
    assert len(queue) == 1
    assert queue[0]["id"] == "g1"
    assert queue[0]["status"] == "escalated"


def test_human_decide_approve(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0, refunded=0.0)
    _seed_grievance(tmp_db, "g1", "o1", "c1", "escalated")

    result = human_decide("g1", "approve", reason="approved by manager")

    assert result.status == "refunded"
    assert result.amount == 100.0  # full order amount
    assert result.refund_id is not None


def test_human_decide_reject(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0)
    _seed_grievance(tmp_db, "g1", "o1", "c1", "escalated")

    result = human_decide("g1", "reject", reason="policy violation")

    assert result.status == "rejected"
    assert result.amount == 0.0
    assert result.refund_id is None


def test_human_decide_adjust_within_max(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0, refunded=0.0)
    _seed_grievance(tmp_db, "g1", "o1", "c1", "escalated")

    result = human_decide("g1", "adjust", amount=30.0, reason="partial refund")

    assert result.status == "refunded"
    assert result.amount == 30.0


def test_human_decide_adjust_exceeds_max_requires_override(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0, refunded=80.0)  # only 20 remaining
    _seed_grievance(tmp_db, "g1", "o1", "c1", "escalated")

    with pytest.raises(ValueError, match="exceeds max refundable"):
        human_decide("g1", "adjust", amount=50.0, reason="trying to override")


def test_human_decide_adjust_with_override(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0, refunded=80.0)  # only 20 remaining
    _seed_grievance(tmp_db, "g1", "o1", "c1", "escalated")

    result = human_decide("g1", "adjust", amount=50.0, reason="override", override_max=True)

    assert result.status == "refunded"
    assert result.amount == 50.0
    # Check human_override flag
    with sqlite3.connect(str(tmp_db)) as conn:
        conn.row_factory = sqlite3.Row
        refund = conn.execute(
            "SELECT human_override FROM refunds WHERE grievance_id = ?", ("g1",)
        ).fetchone()
        assert refund["human_override"] == 1


def test_human_decide_requires_reason(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0)
    _seed_grievance(tmp_db, "g1", "o1", "c1", "escalated")

    with pytest.raises(ValueError, match="reason is required"):
        human_decide("g1", "approve", reason="")


def test_human_decide_requires_escalated_status(tmp_db: Path) -> None:
    _seed_order(tmp_db, "o1", "c1", 100.0)
    _seed_grievance(tmp_db, "g1", "o1", "c1", "eligible")

    with pytest.raises(ValueError, match="not escalated"):
        human_decide("g1", "approve", reason="test")

