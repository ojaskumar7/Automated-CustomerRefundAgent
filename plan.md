# Plan: AI-Powered E-Commerce Refund Resolution Agent

## Goal
FastAPI backend + Streamlit frontend. Customer types a grievance; RAG (ChromaDB + Sentence Transformers) retrieves refund policy context; order verified in SQLite; eligibility decided by pure Python business logic; simulated refunds gated on eligibility; human escalation queue with approve/reject/adjust-amount + mandatory reason. The Groq LLM is an advisor only — it never approves or executes refunds.

## Stack
Python, FastAPI, Groq API, ChromaDB, Sentence Transformers, SQLite, Streamlit.

## Decisions (finalized)
- **Input:** text grievance via FastAPI (`order_id` + `grievance_text`).
- **Refund system:** custom/internal SQLite DB; refunds simulated, recorded in SQLite (no real payment processor).
- **Auto-approve:** eligible refunds ≤ `REFUND_AUTO_APPROVE_THRESHOLD` (env, default 50) auto-approved; above threshold escalated. Eligibility checks always run first.
- **Amount authority:** the eligibility engine computes the maximum refundable amount independently (order amount minus already-refunded amount, capped by policy). The LLM's `requested_amount` is advisory only and is never authoritative — the engine's computed max is the ceiling, enforced in `refund.py` regardless of human override.
- **Repeat customer limit:** a customer is ineligible if they have ≥ `MAX_REFUNDS_30D` (env, default 3) approved refunds within the rolling 30 days preceding the grievance. Counting is based on `refunds` rows with status `simulated_issued` joined to grievances/orders by `customer_id`; the 30-day window is computed from grievance `created_at` looking back.
- **Policy:** sample policies for defective, not-as-described, late delivery, billing error, change-of-mind under `data/policies/`.
- **Env knobs:** `REFUND_AUTO_APPROVE_THRESHOLD` (default 50), `MAX_REFUNDS_30D` (default 3), `RETURN_WINDOW_DAYS` (policy-dependent, default 30), `GROQ_API_KEY`, `GROQ_MODEL`.
- **Human review:** approve / reject / adjust amount + mandatory reason; verified refundable amount is max unless explicit override flag.

## Data model (SQLite)
- `orders` — id, customer_id, amount, currency, status, payment_method, created_at, refunded_amount
- `grievances` — id, order_id, customer_id, grievance_text, extracted_reason, requested_amount (advisory, from LLM; never authoritative), category, status (pending/eligible/ineligible/escalated/refunded/rejected), decided_by (agent|human|system), decision_reason, created_at
- `refunds` — id, grievance_id, order_id, amount, status (simulated_issued/denied/failed), simulated_ref, issued_at, human_override
- `audit_log` — id, grievance_id, action, actor (llm|system|human), details, created_at

## Project structure
```
/home/barely_ok/CustomerRespondAgent/
├── app/
│   ├── main.py          FastAPI app, routes, lifespan DB init
│   ├── config.py        Pydantic Settings (GROQ_API_KEY, REFUND_AUTO_APPROVE_THRESHOLD, paths)
│   ├── db.py            SQLite connection, schema migrations, transaction wrapper
│   ├── models.py        Pydantic schemas
│   ├── rag/             embeddings.py, vectorstore.py, ingest.py
│   ├── llm/             groq_client.py (advisor only; structured JSON output, retry/backoff)
│   ├── eligibility.py   Pure Python: order verify, policy match, time window, amount cap, repeat limit
│   ├── refund.py        Simulated refund tool + execution gated on eligibility
│   ├── escalation.py    Human review queue helpers
│   └── audit.py         Append-only audit log
├── frontend/
│   └── streamlit_app.py Customer form + status view; admin queue approve/reject/adjust + reason
├── data/
│   ├── policies/        defective_product.txt, not_as_described.txt, late_delivery.txt,
│   │                    billing_error.txt, change_of_mind.txt
│   └── orders.db
├── tests/
│   ├── test_eligibility.py
│   ├── test_rag.py
│   ├── test_api.py
│   └── test_refund.py
├── requirements.txt
├── .env.example
└── README.md
```

## Phases (ordered task list)
1. **Foundation:** skeleton, config/.env, SQLite schema + migrations, FastAPI health check, Pydantic models, logging.
2. **RAG pipeline:** SentenceTransformers encoder, ChromaDB collection, ingest `data/policies/`, retrieval helper.
3. **Groq LLM advisor:** client wrapper, structured JSON complaint understanding, prompts, retry/backoff. Output feeds eligibility engine only.
4. **Eligibility engine (Python):** order verification, policy matching, time window, amount cap, repeat limit. Unit-tested without LLM.
5. **Simulated refund + escalation:** `refund.py` gated on eligibility; human review queue endpoints; idempotency; audit logging.
6. **Streamlit frontend:** customer submission/status; admin queue approve/reject/adjust + mandatory reason; max-amount enforcement.
7. **Hardening:** error handling + logging; unit tests (eligibility, refund, RAG); integration tests (FastAPI TestClient + temp SQLite + mocked Groq); LLM-free CI mode; `ruff check` + `mypy`; README runbook.

## Validation plan
- Unit tests: `eligibility.py` edge cases — below/above $50 threshold, outside return window, already refunded, repeat customer (3 issued refunds in 30d → ineligible; 2 → eligible), disallowed reason, LLM requested_amount above/below/equal to engine max (max always wins), human override above max (rejected unless override flag).
- Integration tests with temp SQLite + mocked Groq: small valid → auto-refunded; large → escalated/no refund; human approve/reject/adjust → refund issued/denied/audited; duplicate idempotency key → no second refund; missing order → ineligible.
- LLM-free CI mode (deterministic stub) so tests run without Groq key.
- `ruff check` and `mypy` on `app/`.

## Risks & mitigations
| Risk | Mitigation |
|---|---|
| LLM hallucinates eligibility | LLM never decides; Python engine is final authority. |
| LLM requested_amount inflated | Engine computes max independently; refund.py enforces the ceiling even on human override. |
| Double refund on retry | Idempotency key + unique constraint; single transaction. |
| Groq unavailable | Fail closed → escalate to human queue. |
| Human adjusts above max | Enforced in `refund.py` unless explicit override flag. |
| Repeat-limit bypass | Count is on issued refunds in rolling 30d; re-eligible only after window elapses. |

## Out of scope (v1)
Voice/speech-to-text, real payment processor (simulated refunds only), auth/authorization, email/webhook notifications.

## Open items
None. Plan is implementation-ready.
