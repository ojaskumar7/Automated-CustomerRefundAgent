"""SQLite connection, schema, and transaction helpers."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config import settings

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    amount REAL NOT NULL CHECK (amount >= 0),
    currency TEXT NOT NULL DEFAULT 'USD',
    status TEXT NOT NULL DEFAULT 'placed',
    payment_method TEXT NOT NULL DEFAULT 'card',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    refunded_amount REAL NOT NULL DEFAULT 0 CHECK (refunded_amount >= 0)
);

CREATE TABLE IF NOT EXISTS grievances (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    grievance_text TEXT NOT NULL,
    extracted_reason TEXT,
    requested_amount REAL,
    category TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','eligible','ineligible','escalated','refunded','rejected')),
    decided_by TEXT NOT NULL DEFAULT 'agent'
        CHECK (decided_by IN ('agent','human','system')),
    decision_reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS refunds (
    id TEXT PRIMARY KEY,
    grievance_id TEXT NOT NULL UNIQUE,
    order_id TEXT NOT NULL,
    amount REAL NOT NULL CHECK (amount >= 0),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','simulated_issued','denied','failed')),
    simulated_ref TEXT,
    issued_at TEXT,
    human_override INTEGER NOT NULL DEFAULT 0 CHECK (human_override IN (0,1)),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (grievance_id) REFERENCES grievances(id) ON DELETE CASCADE,
    FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grievance_id TEXT,
    action TEXT NOT NULL,
    actor TEXT NOT NULL CHECK (actor IN ('llm','system','human')),
    details TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_grievances_order ON grievances(order_id);
CREATE INDEX IF NOT EXISTS idx_grievances_customer ON grievances(customer_id);
CREATE INDEX IF NOT EXISTS idx_grievances_status ON grievances(status);
CREATE INDEX IF NOT EXISTS idx_refunds_order ON refunds(order_id);
CREATE INDEX IF NOT EXISTS idx_refunds_grievance ON refunds(grievance_id);
CREATE INDEX IF NOT EXISTS idx_refunds_issued_at ON refunds(issued_at);
CREATE INDEX IF NOT EXISTS idx_audit_grievance ON audit_log(grievance_id);
"""


def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row factory set."""
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create schema if missing (idempotent)."""
    conn = get_connection()
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """Single-transaction wrapper; commits on success, rolls back on error."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Module-level singleton connection for FastAPI lifespan / route use.
_conn: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = get_connection()
    return _conn


def close_db() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
