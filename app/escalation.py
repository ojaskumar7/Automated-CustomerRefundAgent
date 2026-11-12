"""Human review queue: approve/reject/adjust with mandatory reason."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.db import get_db
from app.refund import deny_refund, execute_refund


@dataclass(frozen=True)
class HumanDecisionResult:
    grievance_id: str
    status: str  # refunded | rejected
    refund_id: str | None
    amount: float
    message: str


def get_escalation_queue() -> list[dict[str, Any]]:
    """Return all escalated grievances awaiting human review."""
    db = get_db()
    rows = db.execute(
        """
        SELECT g.id, g.status, g.order_id, g.customer_id, g.grievance_text, g.extracted_reason,
               g.requested_amount, g.category, g.created_at, g.decision_reason,
               o.amount as order_amount, o.refunded_amount
        FROM grievances g
        JOIN orders o ON g.order_id = o.id
        WHERE g.status = 'escalated'
        ORDER BY g.created_at ASC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def get_grievance_details(grievance_id: str) -> dict[str, Any] | None:
    """Get full grievance details for human review."""
    db = get_db()
    row = db.execute(
        """
        SELECT g.*, o.amount as order_amount, o.refunded_amount, o.currency
        FROM grievances g
        JOIN orders o ON g.order_id = o.id
        WHERE g.id = ?
        """,
        (grievance_id,),
    ).fetchone()
    return dict(row) if row else None


def human_decide(
    grievance_id: str,
    action: str,  # approve | reject | adjust
    amount: float | None = None,
    reason: str = "",
    override_max: bool = False,
) -> HumanDecisionResult:
    """Process human decision on escalated grievance.
    - approve: issue refund at max_refundable (from grievance details)
    - reject: deny with reason
    - adjust: issue refund at specified amount (must not exceed max unless override_max=True)
    """
    if not reason:
        raise ValueError("reason is required")

    grievance = get_grievance_details(grievance_id)
    if grievance is None:
        raise ValueError("grievance not found")
    if grievance["status"] != "escalated":
        raise ValueError(f"grievance not escalated (status={grievance['status']})")

    order_amount = grievance["order_amount"]
    already_refunded = grievance["refunded_amount"]
    max_refundable = order_amount - already_refunded

    if action == "approve":
        refund_amount = max_refundable
    elif action == "reject":
        deny_refund(grievance_id, f"Human rejected: {reason}")
        return HumanDecisionResult(
            grievance_id=grievance_id,
            status="rejected",
            refund_id=None,
            amount=0.0,
            message=f"Rejected: {reason}",
        )
    elif action == "adjust":
        if amount is None:
            raise ValueError("amount required for adjust action")
        if amount > max_refundable and not override_max:
            raise ValueError(
                f"adjusted amount ${amount:.2f} exceeds max refundable ${max_refundable:.2f} "
                "(use override_max=True to force)"
            )
        refund_amount = amount
    else:
        raise ValueError(f"invalid action: {action}")

    # Execute refund
    result = execute_refund(
        grievance_id=grievance_id,
        order_id=grievance["order_id"],
        customer_id=grievance["customer_id"],
        amount=refund_amount,
        human_override=override_max,
    )

    return HumanDecisionResult(
        grievance_id=grievance_id,
        status="refunded",
        refund_id=result.refund_id,
        amount=refund_amount,
        message=f"Human {action}d: {reason} (${refund_amount:.2f})",
    )

