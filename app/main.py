"""FastAPI application: routes, lifespan, DB init."""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.db import close_db, get_db, init_db
from app.escalation import HumanDecisionResult, get_escalation_queue, human_decide
from app.models import (
    GrievanceStatusResponse,
    GrievanceSubmitRequest,
    GrievanceSubmitResponse,
    HumanDecisionRequest,
    HumanDecisionResponse,
)
from app.refund import RefundResult, process_grievance_full

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("customer_respond_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: initializing database")
    init_db()
    yield
    logger.info("Shutting down: closing database")
    close_db()


app = FastAPI(
    title="Customer Respond Agent",
    description="AI-powered e-commerce refund resolution agent",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.post("/grievances", response_model=GrievanceSubmitResponse)
async def submit_grievance(req: GrievanceSubmitRequest) -> GrievanceSubmitResponse:
    """Submit grievance and run full pipeline (RAG -> LLM -> Eligibility -> Refund/Escalate)."""
    try:
        result: RefundResult = process_grievance_full(
            order_id=req.order_id,
            customer_id=req.order_id,  # using order_id as customer_id for simplicity
            grievance_text=req.grievance_text,
            idempotency_key=req.idempotency_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # LLM/eligibility failure -> escalate
        logger.exception("Pipeline error, escalating")
        grievance_id = req.idempotency_key or str(uuid.uuid4())
        with get_db() as conn:
            conn.execute(
                "INSERT INTO grievances "
                "(id, order_id, customer_id, grievance_text, status, decided_by, decision_reason) "
                "VALUES (?, ?, ?, ?, 'escalated', 'agent', ?)",
                (grievance_id, req.order_id, req.order_id, f"pipeline_error: {exc}"),
            )
        return GrievanceSubmitResponse(
            grievance_id=grievance_id,
            status="escalated",
            message=f"escalated due to error: {exc}",
        )

    return GrievanceSubmitResponse(
        grievance_id=result.grievance_id,
        status=result.status,
        message=result.message,
    )


@app.get("/grievances/{grievance_id}", response_model=GrievanceStatusResponse)
async def get_grievance(grievance_id: str) -> GrievanceStatusResponse:
    db = get_db()
    row = db.execute(
        "SELECT * FROM grievances WHERE id = ?", (grievance_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="grievance not found")

    refund = db.execute(
        "SELECT id, amount, status FROM refunds WHERE grievance_id = ?", (grievance_id,)
    ).fetchone()

    return GrievanceStatusResponse(
        grievance_id=row["id"],
        order_id=row["order_id"],
        status=row["status"],
        decided_by=row["decided_by"],
        decision_reason=row["decision_reason"],
        refund_id=refund["id"] if refund else None,
        refund_amount=refund["amount"] if refund else None,
        refund_status=refund["status"] if refund else None,
    )


@app.get("/escalations")
async def list_escalations() -> list[dict]:
    """Return all grievances awaiting human review."""
    return get_escalation_queue()


@app.post("/escalations/{grievance_id}/decide", response_model=HumanDecisionResponse)
async def decide_escalation(
    grievance_id: str, req: HumanDecisionRequest
) -> HumanDecisionResponse:
    """Human approve/reject/adjust on escalated grievance."""
    try:
        result: HumanDecisionResult = human_decide(
            grievance_id=grievance_id,
            action=req.action,
            amount=req.amount,
            reason=req.reason,
            override_max=req.override_max,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return HumanDecisionResponse(
        grievance_id=result.grievance_id,
        status=result.status,
        refund_id=result.refund_id,
        message=result.message,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error"},
    )
