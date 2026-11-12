"""Eligibility engine: pure Python business logic for refund decisions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from app.config import settings
from app.db import get_db
from app.llm.groq_client import LLMUnderstanding
from app.rag.retrieval import retrieve


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    reason: str
    max_refundable_amount: float
    category: str
    details: dict[str, Any]


def _parse_iso_datetime(dt_str: str) -> datetime:
    """Parse ISO format datetime from SQLite."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    # Fallback: try parsing with fromisoformat (handles microseconds)
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def _now_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def verify_order(order_id: str) -> tuple[bool, str, dict | None]:
    """Verify order exists and return (ok, message, order_dict)."""
    db = get_db()
    row = db.execute(
        "SELECT id, customer_id, amount, currency, status, created_at, refunded_amount "
        "FROM orders WHERE id = ?",
        (order_id,),
    ).fetchone()

    if row is None:
        return False, "order_not_found", None

    order = dict(row)
    return True, "ok", order


def check_time_window(
    order_created_at: str, return_window_days: int | None = None
) -> tuple[bool, str]:
    """Check if grievance is within return window."""
    window = return_window_days or settings.return_window_days
    order_dt = _parse_iso_datetime(order_created_at)
    elapsed = _now_utc() - order_dt
    if elapsed > timedelta(days=window):
        return False, f"outside_return_window_{window}d"
    return True, "ok"


def check_already_refunded(order: dict, requested_amount: float | None) -> tuple[bool, str, float]:
    """Check if order already fully refunded. Returns (ok, reason, max_refundable)."""
    refunded = order.get("refunded_amount", 0.0)
    order_amount = order["amount"]
    remaining = order_amount - refunded

    if remaining <= 0:
        return False, "fully_refunded", 0.0

    # If LLM extracted an amount, cap at remaining
    max_refundable = remaining
    if requested_amount is not None:
        max_refundable = min(max_refundable, requested_amount)

    return True, "ok", max_refundable


def check_repeat_limit(customer_id: str) -> tuple[bool, str]:
    """Check if customer exceeds repeat refund limit in rolling 30 days."""
    db = get_db()
    cutoff = (_now_utc() - timedelta(days=30)).isoformat()

    count = db.execute(
        """
        SELECT COUNT(*) as cnt
        FROM refunds r
        JOIN grievances g ON r.grievance_id = g.id
        WHERE g.customer_id = ? AND r.status = 'simulated_issued' AND r.issued_at >= ?
        """,
        (customer_id, cutoff),
    ).fetchone()["cnt"]

    if count >= settings.max_refunds_30d:
        return False, f"repeat_limit_exceeded_{count}_in_30d"
    return True, "ok"


def match_policy_category(llm_category: str, grievance_text: str) -> tuple[str, float]:
    """Validate LLM category against retrieved policy chunks.
    Returns (matched_category, confidence_boost).
    """
    # Retrieve top chunks for the grievance
    chunks = retrieve(grievance_text, k=5)
    if not chunks:
        return "other", 0.0

    # Check if LLM category appears in top results
    top_categories = [c["metadata"]["category"] for c in chunks[:3]]
    if llm_category in top_categories:
        return llm_category, 0.1  # boost confidence

    # Fall back to top retrieved category
    return top_categories[0], 0.0


def evaluate_eligibility(
    order_id: str,
    customer_id: str,
    grievance_text: str,
    llm_understanding: LLMUnderstanding,
) -> EligibilityResult:
    """Main eligibility evaluation. Pure Python, no side effects."""
    # 1. Verify order
    ok, msg, order = verify_order(order_id)
    if not ok:
        return EligibilityResult(
            eligible=False,
            reason=msg,
            max_refundable_amount=0.0,
            category=llm_understanding.category,
            details={"order_id": order_id},
        )
    assert order is not None  # for mypy

    # 2. Time window
    ok, msg = check_time_window(order["created_at"])
    if not ok:
        return EligibilityResult(
            eligible=False,
            reason=msg,
            max_refundable_amount=0.0,
            category=llm_understanding.category,
            details={"order_id": order_id, "order_created_at": order["created_at"]},
        )

    # 3. Already refunded check + compute max refundable
    ok, msg, max_refundable = check_already_refunded(order, llm_understanding.requested_amount)
    if not ok:
        return EligibilityResult(
            eligible=False,
            reason=msg,
            max_refundable_amount=0.0,
            category=llm_understanding.category,
            details={"order_id": order_id, "refunded_amount": order.get("refunded_amount", 0)},
        )

    # 4. Repeat limit
    ok, msg = check_repeat_limit(customer_id)
    if not ok:
        return EligibilityResult(
            eligible=False,
            reason=msg,
            max_refundable_amount=0.0,
            category=llm_understanding.category,
            details={"customer_id": customer_id},
        )

    # 5. Policy category validation
    matched_category, confidence_boost = match_policy_category(
        llm_understanding.category, grievance_text
    )

    # Eligible!
    final_confidence = min(1.0, llm_understanding.confidence + confidence_boost)
    return EligibilityResult(
        eligible=True,
        reason="eligible",
        max_refundable_amount=round(max_refundable, 2),
        category=matched_category,
        details={
            "order_id": order_id,
            "customer_id": customer_id,
            "order_amount": order["amount"],
            "already_refunded": order.get("refunded_amount", 0),
            "llm_requested_amount": llm_understanding.requested_amount,
            "llm_category": llm_understanding.category,
            "matched_category": matched_category,
            "llm_confidence": llm_understanding.confidence,
            "final_confidence": final_confidence,
            "llm_reasoning": llm_understanding.reasoning,
        },
    )


def auto_approve_decision(max_refundable: float) -> tuple[bool, str]:
    """Determine if eligible refund should be auto-approved or escalated."""
    if max_refundable <= settings.refund_auto_approve_threshold:
        return True, "auto_approved"
    return False, "escalated_above_threshold"
