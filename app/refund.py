"""Simulated refund execution gated on eligibility."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from app.db import get_db, transaction
from app.eligibility import auto_approve_decision, evaluate_eligibility
from app.llm.groq_client import understand_grievance
from app.rag.retrieval import retrieve


@dataclass(frozen=True)
class RefundResult:
    refund_id: str
    status: str  # simulated_issued | denied | escalated | failed
    amount: float
    message: str
    grievance_id: str


def _now_iso() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


def _generate_ref_id() -> str:
    return f"sim_{uuid.uuid4().hex[:12]}"


def _generate_grievance_id() -> str:
    return uuid.uuid4().hex


def execute_refund(
    grievance_id: str,
    order_id: str,
    customer_id: str,
    amount: float,
    human_override: bool = False,
) -> RefundResult:
    """Execute simulated refund within a single transaction.
    Returns RefundResult with status and simulated reference.
    """
    ref = _generate_ref_id()
    issued_at = _now_iso()

    with transaction() as conn:
        # Insert refund record
        conn.execute(
            "INSERT INTO refunds "
            "(id, grievance_id, order_id, amount, status, "
            "simulated_ref, issued_at, human_override) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ref,
                grievance_id,
                order_id,
                amount,
                "simulated_issued",
                ref,
                issued_at,
                int(human_override),
            ),
        )
        # Update grievance status
        conn.execute(
            "UPDATE grievances SET status = 'refunded', decided_by = 'system', "
            "decision_reason = ? WHERE id = ?",
            (f"refund_executed_{ref}", grievance_id),
        )
        # Update order refunded_amount
        conn.execute(
            "UPDATE orders SET refunded_amount = refunded_amount + ? WHERE id = ?",
            (amount, order_id),
        )

    return RefundResult(
        refund_id=ref,
        status="simulated_issued",
        amount=amount,
        message=f"Refund {ref} issued for ${amount:.2f}",
        grievance_id=grievance_id,
    )


def deny_refund(grievance_id: str, reason: str) -> RefundResult:
    """Record a denied refund."""
    ref = _generate_ref_id()

    with transaction() as conn:
        conn.execute(
            "INSERT INTO refunds "
            "(id, grievance_id, order_id, amount, status, simulated_ref) "
            "VALUES (?, ?, (SELECT order_id FROM grievances WHERE id = ?), 0, 'denied', ?)",
            (ref, grievance_id, grievance_id, ref),
        )
        conn.execute(
            "UPDATE grievances SET status = 'rejected', decided_by = 'system', "
            "decision_reason = ? WHERE id = ?",
            (reason, grievance_id),
        )

    return RefundResult(
        refund_id=ref,
        status="denied",
        amount=0.0,
        message=reason,
        grievance_id=grievance_id,
    )


def process_grievance_full(
    order_id: str,
    customer_id: str,
    grievance_text: str,
    idempotency_key: str | None = None,
) -> RefundResult:
    """Full pipeline: retrieve -> LLM -> eligibility -> execute/deny/escalate.
    Idempotent on idempotency_key (used as grievance_id if provided).
    """
    grievance_id = idempotency_key or _generate_grievance_id()

    db = get_db()
    # Check idempotency
    existing = db.execute(
        "SELECT id, status FROM grievances WHERE id = ?", (grievance_id,)
    ).fetchone()
    if existing:
        return RefundResult(
            refund_id="",
            status=existing["status"],
            amount=0.0,
            message="already processed",
            grievance_id=grievance_id,
        )

    # 1. Retrieve policy chunks
    chunks = retrieve(grievance_text, k=5)

    # 2. LLM understanding (stub in CI)
    llm = understand_grievance(grievance_text, chunks)

    # 3. Eligibility
    eligibility = evaluate_eligibility(order_id, customer_id, grievance_text, llm)

    # 4. Store grievance
    with transaction() as conn:
        conn.execute(
            "INSERT INTO grievances (id, order_id, customer_id, grievance_text, extracted_reason, "
            "requested_amount, category, status, decided_by, decision_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                grievance_id,
                order_id,
                customer_id,
                grievance_text,
                llm.extracted_reason,
                llm.requested_amount,
                eligibility.category,
                "eligible" if eligibility.eligible else "ineligible",
                "agent",
                eligibility.reason,
            ),
        )

    if not eligibility.eligible:
        return deny_refund(grievance_id, eligibility.reason)

    # 5. Auto-approve or escalate
    auto_approve, decision_reason = auto_approve_decision(eligibility.max_refundable_amount)

    if auto_approve:
        return execute_refund(
            grievance_id=grievance_id,
            order_id=order_id,
            customer_id=customer_id,
            amount=eligibility.max_refundable_amount,
        )
    else:
        # Escalate to human queue
        with transaction() as conn:
            conn.execute(
                "UPDATE grievances SET status = 'escalated', decided_by = 'agent', "
                "decision_reason = ? WHERE id = ?",
                (decision_reason, grievance_id),
            )
        return RefundResult(
            refund_id="",
            status="escalated",
            amount=eligibility.max_refundable_amount,
            message=f"Escalated for human review: {decision_reason}",
            grievance_id=grievance_id,
        )

