"""Phase 1 foundation tests."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.db import close_db, init_db, transaction
from app.main import app


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "orders.db"
    monkeypatch.setattr(settings, "db_path", db_path)
    close_db()  # drop any cached singleton connection from a prior test
    init_db()
    return db_path


@pytest.fixture()
def client(tmp_db: Path) -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_schema_tables(tmp_db: Path) -> None:
    conn = sqlite3.connect(str(tmp_db))
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"orders", "grievances", "refunds", "audit_log"} <= tables


def test_submit_and_status(client: TestClient) -> None:
    r = client.post("/grievances", json={"order_id": "o1", "grievance_text": "broken"})
    assert r.status_code == 200
    gid = r.json()["grievance_id"]
    # Status could be pending, refunded, escalated, etc. depending on pipeline
    assert "status" in r.json()

    r = client.get(f"/grievances/{gid}")
    assert r.status_code == 200
    assert "status" in r.json()


def test_idempotency_key(client: TestClient) -> None:
    payload = {"order_id": "o1", "grievance_text": "broken", "idempotency_key": "key-1"}
    r1 = client.post("/grievances", json=payload)
    r2 = client.post("/grievances", json=payload)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["grievance_id"] == r2.json()["grievance_id"]
    assert r2.json()["message"] == "already processed"


def test_missing_grievance_404(client: TestClient) -> None:
    r = client.get("/grievances/does-not-exist")
    assert r.status_code == 404


def test_transaction_rollback(tmp_db: Path) -> None:
    with pytest.raises(RuntimeError), transaction() as conn:
        conn.execute(
            "INSERT INTO orders (id, customer_id, amount) VALUES (?,?,?)",
            ("o1", "c1", 10.0),
        )
        raise RuntimeError("boom")
    conn = sqlite3.connect(str(tmp_db))
    count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    conn.close()
    assert count == 0


def test_transaction_commit(tmp_db: Path) -> None:
    with transaction() as conn:
        conn.execute(
            "INSERT INTO orders (id, customer_id, amount) VALUES (?,?,?)",
            ("o1", "c1", 10.0),
        )
    conn = sqlite3.connect(str(tmp_db))
    count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    conn.close()
    assert count == 1
