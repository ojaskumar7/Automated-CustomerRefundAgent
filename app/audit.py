"""Append-only audit logging."""
from __future__ import annotations

import json
from typing import Any

from app.db import get_db


def log(
    action: str,
    actor: str,
    grievance_id: str | None = None,
    details: Any | None = None,
) -> int:
    """Insert an audit row. Never raises to callers (best-effort)."""
    try:
        db = get_db()
        cur = db.execute(
            "INSERT INTO audit_log (grievance_id, action, actor, details) VALUES (?, ?, ?, ?)",
            (
                grievance_id,
                action,
                actor,
                json.dumps(details, default=str) if details is not None else None,
            ),
        )
        db.commit()
        return cur.lastrowid or 0
    except Exception:  # noqa: BLE001 - audit must never break the request path
        return 0
