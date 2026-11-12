"""Pydantic schemas for API request/response."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GrievanceSubmitRequest(BaseModel):
    order_id: str = Field(..., description="Order the grievance refers to")
    grievance_text: str = Field(..., min_length=1, description="Customer's complaint text")
    idempotency_key: str | None = Field(
        None, description="Optional client key to prevent duplicate submissions"
    )


class GrievanceSubmitResponse(BaseModel):
    grievance_id: str
    status: str
    message: str


class HumanDecisionRequest(BaseModel):
    action: str = Field(..., description="approve | reject | adjust")
    amount: float | None = Field(None, description="Required when action=adjust")
    reason: str = Field(..., min_length=1, description="Mandatory reason for the decision")
    override_max: bool = Field(
        False, description="Explicit override to exceed computed max refundable amount"
    )


class HumanDecisionResponse(BaseModel):
    grievance_id: str
    status: str
    refund_id: str | None = None
    message: str


class GrievanceStatusResponse(BaseModel):
    grievance_id: str
    order_id: str
    status: str
    decided_by: str | None = None
    decision_reason: str | None = None
    refund_id: str | None = None
    refund_amount: float | None = None
    refund_status: str | None = None
