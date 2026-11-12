# Customer Respond Agent

AI-powered e-commerce refund resolution agent. FastAPI backend + Streamlit frontend. Customer submits grievance; RAG retrieves policy context; LLM extracts structured intent; pure Python eligibility engine decides; simulated refunds gated on eligibility; human escalation queue with approve/reject/adjust + mandatory reason.

## Architecture

```
Customer Grievance
       │
       ▼
┌──────────────────┐
│  RAG Pipeline    │  ← ChromaDB + Sentence Transformers
│  (policy chunks) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Groq LLM        │  ← Structured JSON extraction only
│  (advisor only)  │     NEVER approves/executes refunds
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Eligibility     │  ← Pure Python business logic
│  Engine          │     Order verify, time window, amount cap, repeat limit
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
Auto-      Escalate
Approve    to Human
(≤$50)     Queue
```

## Quick Start

### Prerequisites
- Python 3.12+
- Groq API key (optional - runs in stub mode without it)

### Local Development

```bash
# Clone and setup
cd CustomerRespondAgent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (or leave empty for stub mode)

# Initialize database
python -c "from app.db import init_db; init_db()"

# Ingest policies into ChromaDB
python -c "from app.rag.ingest import ingest_policies; print(f'Ingested {ingest_policies()} chunks')"

# Run tests
pytest tests/ -q

# Lint & typecheck
ruff check app/ tests/
mypy app/

# Start API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start Streamlit frontend
streamlit run frontend/streamlit_app.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (empty) | Groq API key for LLM advisor |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `REFUND_AUTO_APPROVE_THRESHOLD` | `50` | Auto-approve refunds ≤ this amount |
| `MAX_REFUNDS_30D` | `3` | Max issued refunds per customer in 30 days |
| `RETURN_WINDOW_DAYS` | `30` | Return window in days |
| `LLM_STUB` | auto | Force stub mode (`1`/`true`/`yes`) |
| `API_BASE` | `http://localhost:8000` | Frontend API base URL |

**LLM-Free CI Mode**: Set `LLM_STUB=1` or omit `GROQ_API_KEY` — uses deterministic keyword-based extraction.

## API Endpoints

### Customer
- `POST /grievances` — Submit grievance (runs full pipeline)
  ```json
  { "order_id": "ORD-123", "grievance_text": "screen cracked", "idempotency_key": "optional" }
  ```
- `GET /grievances/{id}` — Check status

### Admin
- `GET /escalations` — List escalated grievances
- `POST /escalations/{id}/decide` — Human decision
  ```json
  { "action": "approve|reject|adjust", "amount": 49.99, "reason": "required", "override_max": false }
  ```

### Health
- `GET /health` — Health check

## Policy Categories

Policies in `data/policies/`:
- `defective_product.txt` — Manufacturing defects (30 days, full refund)
- `not_as_described.txt` — Wrong item/specs (30 days, full refund)
- `late_delivery.txt` — Delivery delays (7-14 days: 15%, >14 days: full)
- `billing_error.txt` — Incorrect charges (60 days, overcharged amount)
- `change_of_mind.txt` — Returns (14 days, product price minus $9.99 shipping)

## Project Structure

```
CustomerRespondAgent/
├── app/
│   ├── main.py           # FastAPI app, routes
│   ├── config.py         # Pydantic Settings
│   ├── db.py             # SQLite + transaction wrapper
│   ├── models.py         # Pydantic schemas
│   ├── audit.py          # Append-only audit log
│   ├── eligibility.py    # Pure Python eligibility engine
│   ├── refund.py         # Simulated refund execution
│   ├── escalation.py     # Human review queue
│   ├── llm/
│   │   ├── groq_client.py  # Groq LLM client (advisor only)
│   │   └── stub.py         # Deterministic stub for CI
│   └── rag/
│       ├── embeddings.py   # SentenceTransformer wrapper
│       ├── vectorstore.py  # ChromaDB client
│       ├── ingest.py       # Policy ingestion
│       └── retrieval.py    # Query helper
├── frontend/
│   └── streamlit_app.py  # Customer + Admin UI
├── data/
│   ├── policies/         # Policy text files
│   └── chroma/           # ChromaDB persistence
├── tests/
│   ├── test_api.py
│   ├── test_rag.py
│   ├── test_llm.py
│   ├── test_eligibility.py
│   └── test_refund.py
├── requirements.txt
├── pyproject.toml
├── .env.example
└── plan.md
```

## Running Tests

```bash
# All tests
pytest tests/ -q

# Specific module
pytest tests/test_eligibility.py -q

# With coverage
pytest tests/ --cov=app --cov-report=term-missing
```

## CI Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt
      - run: python -c "from app.db import init_db; init_db()"
      - run: python -c "from app.rag.ingest import ingest_policies; ingest_policies()"
      - run: ruff check app/ tests/
      - run: mypy app/
      - run: pytest tests/ -q
      - env:
          LLM_STUB: "1"  # Force deterministic stub mode
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| LLM = advisor only | Never decides eligibility; Python engine is final authority |
| Amount authority = engine | LLM `requested_amount` advisory; engine computes `max_refundable = order.amount - refunded` |
| Repeat limit = 3 in 30d | Rolling window on `simulated_issued` refunds |
| Stub for CI | Deterministic, no API keys, fast |
| SQLite + WAL | Zero ops, concurrent readers |
| ChromaDB local | No vector DB server needed |
| Idempotency keys | Client-controlled deduplication |

## Extending

- **New policy**: Add `data/policies/new_category.txt` and re-run ingestion
- **New LLM provider**: Implement same interface in `app/llm/`
- **Real payments**: Replace `execute_refund()` with payment processor integration
- **Auth**: Add middleware to FastAPI routes

## License

MIT