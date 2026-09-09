"""Research Librarian — persistent memory for regimes, trades, audits.

SQLite-backed structured store (no sqlite-vec dependency) for the Tier 4
feedback loop's long-term memory. Mirrors the parent engine's Librarian but
drops the broken vector-embedding path (JSON-string embeddings there were
non-functional) and stores strategy configs in their own table.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_crypto.schemas.order import TradeAudit
from quant_crypto.schemas.regime import Regime
from quant_crypto.schemas.strategy import StrategyConfig

_SCHEMA = """
CREATE TABLE IF NOT EXISTS regimes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    regime TEXT NOT NULL,
    rationale TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Librarian:
    """SQLite-backed research memory."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # --- Regimes ---
    def log_regime(self, date: str, regime: Regime, rationale: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO regimes (date, regime, rationale, created_at) VALUES (?, ?, ?, ?)",
            (date, regime.value, rationale, self._now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def latest_regime(self) -> tuple[str, Regime] | None:
        row = self.conn.execute(
            "SELECT date, regime FROM regimes ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return (row[0], Regime(row[1])) if row else None

    # --- Strategies ---
    def save_strategy(self, cfg: StrategyConfig) -> int:
        cur = self.conn.execute(
            "INSERT INTO strategies (date, payload, created_at) VALUES (?, ?, ?)",
            (self._now()[:10], cfg.model_dump_json(), self._now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def latest_strategy(self) -> StrategyConfig | None:
        row = self.conn.execute(
            "SELECT payload FROM strategies ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return StrategyConfig.model_validate(json.loads(row[0])) if row else None

    # --- Audits ---
    def log_audit(self, audit: TradeAudit) -> int:
        cur = self.conn.execute(
            "INSERT OR REPLACE INTO audits (trade_id, payload, created_at) VALUES (?, ?, ?)",
            (audit.trade_id, audit.model_dump_json(), self._now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def list_audits(self, limit: int = 50) -> list[TradeAudit]:
        rows = self.conn.execute(
            "SELECT payload FROM audits ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [TradeAudit.model_validate(json.loads(r[0])) for r in rows]

    def close(self) -> None:
        self.conn.close()
