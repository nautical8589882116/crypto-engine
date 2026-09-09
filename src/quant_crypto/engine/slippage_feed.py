"""Realized paper-fill friction, visible to the control API (Stage C/D3).

The dashboard must never invent friction. PaperBroker records what each fill
actually paid against the book price it saw at that moment; a session publishes
those records here and GET /api/v1/slippage serves them. Nothing recorded =>
snapshot is None and the UI shows an empty state, never a plausible number.

Records accumulate across sessions (bounded, newest kept) until reset.
"""

from __future__ import annotations

import threading
import time

_MAX_RECORDS = 200

_records: list[dict] = []
_updated_at: str | None = None
_lock = threading.Lock()


def publish_fills(fills: list[dict]) -> None:
    """Append a session's fill-friction records. A no-op for an empty list."""
    global _updated_at
    if not fills:
        return
    with _lock:
        _records.extend(fills)
        del _records[:-_MAX_RECORDS]  # keep the newest window only
        _updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def slippage_snapshot() -> dict | None:
    """Realized friction, or None when nothing has ever been recorded."""
    with _lock:
        if not _records:
            return None
        records = list(_records)
        updated_at = _updated_at
    total = sum(r.get("slippage_rupees", 0.0) for r in records)
    return {
        "updated_at": updated_at,
        "count": len(records),
        "total_slippage_rupees": round(total, 4),
        "fills": records[-50:],  # most recent, for the dashboard table
    }


def reset_slippage() -> None:
    global _updated_at
    with _lock:
        _records.clear()
        _updated_at = None
