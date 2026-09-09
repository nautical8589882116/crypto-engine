"""Feedback loop — closes the self-improvement circle (Tier 4 → Tier 1).

Closed trades become TradeAudits in the librarian; the strategy manager then
re-adapts execution parameters from those audits. Mirrors the parent's
feedback/loop.py.
"""

from __future__ import annotations

from quant_crypto.cloud.librarian import Librarian
from quant_crypto.cloud.premarket import PremarketSnapshot
from quant_crypto.cloud.strategy_manager import StrategyManager
from quant_crypto.schemas.order import TradeAudit
from quant_crypto.schemas.strategy import StrategyConfig


class FeedbackLoop:
    def __init__(self, strategy_manager: StrategyManager, librarian: Librarian):
        self.strategy_manager = strategy_manager
        self.librarian = librarian

    def record_trade(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        expected_price: float,
        closed_at: str,
        slippage_bps: float = 0.0,
        execution_friction_bps: float = 0.0,
        pnl: float = 0.0,
    ) -> TradeAudit:
        audit = TradeAudit(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            expected_price=expected_price,
            slippage_bps=slippage_bps,
            execution_friction_bps=execution_friction_bps,
            pnl=pnl,
            closed_at=closed_at,
        )
        self.librarian.log_audit(audit)
        return audit

    def adapt(self, snapshot: PremarketSnapshot, n_recent: int = 20) -> StrategyConfig:
        audits = self.librarian.list_audits(limit=n_recent)
        if not audits:
            return self.strategy_manager.run_daily(snapshot)
        return self.strategy_manager.adapt_after_trade(audits, snapshot)
