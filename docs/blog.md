# Building an AI-Powered Refund Resolution Agent: Technical Blog

## Series Overview
This blog documents the end-to-end construction of a FastAPI + Streamlit refund resolution agent. The architecture separates concerns rigorously:
- **RAG** retrieves policy context (advisory only)
- **LLM** extracts structured intent from grievances (advisory only)
- **Pure Python eligibility engine** makes the final call — no LLM ever approves or executes refunds
- **Human-in-the-loop** for amounts above threshold or edge cases

---

## Phase 1: Foundation — Why These Choices?

### Project Structure & Monorepo Layout
```
/home/barely_ok/CustomerRespondAgent/
├── app/           # FastAPI backend
├── frontend/      # Streamlit admin/customer UI
├── data/          # policies/, orders.db, chroma/
├── tests/         # pytest suite
├── requirements.txt
├── pyproject.toml
└── .env.example
```
**Why:** Keeps backend, frontend, data, and tests isolated. `app/` is importable as a package. `data/` is gitignored (except policies) — DB and vector store are local artifacts.

### Configuration: `pydantic-settings` over raw `os.environ`
```python
# app/config.py
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    refund_auto_approve_threshold: float = 50.0
    max_refunds_30d: int = 3
    return_window_days: int = 30
```
**Why:**
- Type-safe, validated config with defaults — no `os.getenv` scattered everywhere
- `.env` support for local dev; production injects via environment
- `extra="ignore"` prevents typos from silently creating attributes

### Database: SQLite with WAL + Foreign Keys
```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
```
**Why:**
- **WAL mode** allows concurrent readers + single writer — perfect for FastAPI's thread pool
- **Foreign keys** enforce referential integrity at DB level (orders → grievances → refunds)
- No separate DB server — zero ops overhead for v1
- Schema is idempotent: `init_db()` runs `CREATE TABLE IF NOT EXISTS` on startup

### Transaction Wrapper: Single Context Manager
```python
@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```
**Why:**
- Eliminates boilerplate `try/except/commit/rollback` in every route
- Guarantees atomicity — critical for refund flow (grievance + refund + audit must succeed together or not at all)
- `sqlite3.Row` factory gives dict-like access

### Singleton Connection + Lifespan
```python
# app/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    close_db()
```
**Why:**
- One connection per process (not per request) — avoids connection churn
- `lifespan` replaces deprecated `@app.on_event("startup")`
- `close_db()` on shutdown flushes WAL cleanly

### FastAPI over Flask/Django
**Why:**
- Native async support (critical for LLM/embedding calls later)
- Automatic OpenAPI docs at `/docs`
- Pydantic integration for request/response validation
- Lightweight — no ORM bloat for our raw-SQL approach

### Pydantic Models for Every API Boundary
```python
class GrievanceSubmitRequest(BaseModel):
    order_id: str
    grievance_text: str = Field(..., min_length=1)
    idempotency_key: Optional[str] = None
```
**Why:**
- Validates input before business logic runs — fails fast with 422
- Self-documenting: OpenAPI schema generated automatically
- `idempotency_key` lets clients safely retry without duplicate grievances

### Idempotency Keys at the API Layer
```python
if req.idempotency_key:
    row = db.execute("SELECT id FROM grievances WHERE id = ?", (req.idempotency_key,)).fetchone()
    if row:
        return GrievanceSubmitResponse(grievance_id=row["id"], status=row["status"], message="already exists")
```
**Why:**
- Network failures cause retries; without idempotency, you get duplicate grievances
- Client controls the key (UUID, hash of order+text, etc.) — server just honors it
- Simple unique constraint on `grievances.id` backs it at DB level

### Audit Log: Best-Effort, Never Breaks Requests
```python
def log(action: str, actor: str, grievance_id: str | None = None, details: Any = None) -> int:
    try:
        cur = db.execute(...)
        db.commit()
        return cur.lastrowid or 0
    except Exception:  # noqa: BLE001
        return 0
```
**Why:**
- Every state change is traceable: LLM extraction, eligibility decision, human review, refund execution
- Blind `except` is intentional — audit must *never* cause a 500
- Structured JSON in `details` enables post-hoc analysis

### Testing: FastAPI TestClient + Temp SQLite per Test
```python
@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "orders.db"
    monkeypatch.setattr(settings, "db_path", db_path)
    close_db()  # drop cached singleton
    init_db()
    return db_path
```
**Why:**
- Real SQLite (not in-memory) exercises WAL, FK, constraints
- `monkeypatch` on `settings.db_path` isolates each test
- `close_db()` resets the singleton between tests — critical for parallel runs
- 7 tests cover health, schema, CRUD, idempotency, 404, transaction rollback/commit

### Lint/Typecheck: `ruff` + `mypy` in CI
```toml
# pyproject.toml
[tool.ruff]
line-length = 100
select = ["E", "F", "I", "W", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.12"
```
**Why:**
- `ruff` is 100x faster than flake8/black/isort combined
- `mypy` catches bugs like `Connection.lastrowid` (cursor has it, not connection)
- Runs in <2s locally — no excuse to skip

---

## Phase 2: RAG Pipeline — Embedding & Retrieval Design

### Embedding Model: `sentence-transformers` + `all-MiniLM-L6-v2`
```python
# app/rag/embeddings.py
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None

def get_encoder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def encode(texts: list[str]) -> np.ndarray:
    return get_encoder().encode(texts, normalize_embeddings=True)
```
**Why `sentence-transformers` (not OpenAI/Cohere API):**
- **Zero external dependency** — runs fully offline, critical for LLM-free CI mode
- **Deterministic** — same input → same embedding every time (no API version drift)
- **Free** — no per-call cost at scale

**Why `all-MiniLM-L6-v2`:**
| Model | Dim | Size | Speed | MTEB Retrieval (avg) |
|-------|-----|------|-------|---------------------|
| all-MiniLM-L6-v2 | 384 | 80 MB | Very fast | ~49 |
| BAAI/bge-small-en-v1.5 | 384 | 130 MB | Fast | ~52 |
| all-mpnet-base-v2 | 768 | 420 MB | Slow | ~55 |

- 384-dim is small enough for Chroma's HNSW index to be snappy
- 80 MB downloads once, caches locally
- Retrieval quality is *good enough* for short policy paragraphs — we can swap models later by changing one constant

**Why `normalize_embeddings=True`:**
- Chroma uses cosine similarity; normalized vectors make dot product = cosine
- Avoids manual normalization at query time

### Vector Store: ChromaDB Persistent Client
```python
# app/rag/vectorstore.py
import chromadb
from chromadb.config import Settings as ChromaSettings

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client

def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name="refund_policies",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection
```
**Why ChromaDB:**
- **Persistent local file store** — no server, no Docker, no network
- **HNSW index** — fast ANN search out of the box
- **Metadata filtering** — we tag chunks with `category` (defective, late_delivery, etc.) for hybrid retrieval
- **Python-native** — simple API, no ORM needed

**Why `get_or_create_collection`:**
- Idempotent — safe to call on every startup
- Collection persists across restarts; embeddings don't need re-ingestion

### Ingestion: Policy Files → Chunks → Embeddings → Chroma
```
data/policies/
├── defective_product.txt
├── not_as_described.txt
├── late_delivery.txt
├── billing_error.txt
└── change_of_mind.txt
```

```python
# app/rag/ingest.py
def ingest_policies() -> int:
    """Load policy files, chunk by paragraph, embed, upsert. Returns chunk count."""
    collection = get_collection()
    encoder = get_encoder()

    documents, metadatas, ids = [], [], []
    for path in settings.policies_dir.glob("*.txt"):
        category = path.stem
        text = path.read_text(encoding="utf-8")
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": path.name, "category": category, "chunk": i})
            ids.append(f"{category}-{i}")

    embeddings = encoder.encode(documents, normalize_embeddings=True).tolist()
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
    return len(documents)
```
**Why paragraph-based chunking (`\n\n`):**
- Policies are short (100-500 words each) — paragraphs are semantic units
- No overlap needed; no sliding window complexity
- Metadata preserves `category` for filtered retrieval (e.g., only search `defective_product` chunks)

**Why `upsert` not `add`:**
- Re-running ingestion updates existing chunks (same IDs) — safe for development iteration
- No duplicate vectors on re-ingest

### Retrieval: Top-K with Metadata Filter
```python
# app/rag/retrieval.py
def retrieve(query: str, k: int = 5, category: str | None = None) -> list[dict]:
    """Return list of {text, score, metadata}."""
    collection = get_collection()
    query_emb = get_encoder().encode([query], normalize_embeddings=True).tolist()

    where = {"category": category} if category else None
    results = collection.query(
        query_embeddings=query_emb,
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    out = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        out.append({"text": doc, "score": 1 - dist, "metadata": meta})  # cosine → similarity
    return out
```
**Why return similarity (1 - distance) not raw distance:**
- Higher = better — intuitive for callers
- Chroma returns cosine distance (0=identical, 2=opposite); `1 - dist` maps to [-1, 1] similarity

**Why optional `category` filter:**
- Eligibility engine can narrow search to relevant policy (e.g., only `late_delivery` chunks for a "late" complaint)
- Reduces noise, improves precision

### Why This RAG Design Works for Refund Policies
| Requirement | Solution |
|-------------|----------|
| Offline CI | Local `sentence-transformers` + Chroma files |
| Deterministic | Fixed model, no API randomness |
| Policy-specific retrieval | Metadata filter by `category` |
| Fast startup | Persistent index, no re-embedding |
| Swappable model | One constant change in `embeddings.py` |
| Auditable | Every retrieved chunk has source file + chunk index |

---

## Next: Phase 3 — Groq LLM Advisor (Structured Extraction Only)

The LLM will receive: grievance text + top-K policy chunks. It returns structured JSON:
```json
{
  "extracted_reason": "defective_product",
  "requested_amount": 49.99,
  "category": "defective_product",
  "confidence": 0.92,
  "reasoning": "Customer mentions 'broken screen' which maps to defective product policy."
}
```
**Critical:** The LLM *never* decides eligibility. Its output feeds the Python engine which applies hard rules (time window, amount cap, repeat limit). This is the architectural guardrail that makes the system auditable and safe.