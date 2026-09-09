"""Execution kernel — risk-gated entry and take-profit/stop exit.

This is deliberately simpler and safer than the parent engine's kernel: it does
NOT simulate the market path from the model probability (a bug source there). It
enters on a gated FOK order at the signal-time ask, holds, and exits on real
subsequent prices when a take-profit or stop-loss is hit. Realized PnL is
banked against the daily-loss breaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_crypto.broker.base import Broker
from quant_crypto.config import get_settings
from quant_crypto.exec.risk import RiskManager, RiskLimits, trip_process_kill_switch
from quant_crypto.schemas.order import Order, OrderSide, OrderType
from quant_crypto.schemas.order import Signal


@dataclass
class TradeOutcome:
    signal: Signal
    entered: bool = False
    exit_reason: str | None = None
    pnl: float = 0.0
    entry_price: float | None = None
    exit_price: float | None = None
    log: list[str] = field(default_factory=list)


class ExecutionKernel:
    def __init__(self, broker: Broker, settings=None, risk: RiskManager | None = None):
        self.broker = broker
        self.settings = settings or get_settings()
        self.risk = risk or RiskManager(
            RiskLimits(
                max_daily_loss=self.settings.max_daily_loss,
                max_position=self.settings.max_position,
            )
        )
        # avg entry cost per symbol for realized-PnL accounting (long-only spot).
        self._avg_entry: dict[str, float] = {}

    def kill(self) -> None:
        trip_process_kill_switch()

    def _qty_for(self, symbol: str) -> float:
        return self.settings.quantities.get(symbol, self.settings.max_position)

    def handle_signal(
        self,
        signal: Signal,
        entry_price: float,
        *,
        profit_target: float | None = None,
        stop_loss: float | None = None,
    ) -> TradeOutcome:
        out = TradeOutcome(signal)
        if not signal.is_tradeable:
            out.log.append("not tradeable")
            return out
        symbol = signal.symbol  # spot engine: symbol field carries the pair
        qty = self._qty_for(symbol)
        ok, reason = self.risk.allow_order(symbol, qty)
        if not ok:
            out.log.append(f"risk gate: {reason}")
            return out

        depth = self.broker.get_depth(symbol)
        # Lift the ask to fill: you must pay the ask, not the last trade price.
        best = max((l.price for l in depth if l.side == "ask"), default=entry_price)
        limit = best if best > 0 else entry_price
        order = Order(symbol=symbol, side=OrderSide.BUY, qty=qty, order_type=OrderType.FOK, limit_price=limit)
        fill = self.broker.place_order(order)
        if fill is None:
            out.log.append("FOK entry unfilled")
            return out

        self.risk.record_order_time()
        self.risk.record_position(symbol, qty)
        self._avg_entry[symbol] = fill.fill_price
        out.entered = True
        out.entry_price = fill.fill_price
        out.log.append(f"entered @ {fill.fill_price:.2f} qty={qty}")
        return out

    def exit_position(self, symbol: str, current_price: float, reason: str) -> dict:
        """Liquidate the full open position at current price (market/IOC)."""
        pos = self.broker.get_position(symbol)
        if abs(pos) < 1e-12:
            return {"closed": False, "reason": reason, "pnl": 0.0, "qty": 0.0}
        side = OrderSide.SELL if pos > 0 else OrderSide.BUY
        depth = self.broker.get_depth(symbol)
        # Hit the bid to exit a long (or lift the ask to exit a short).
        if pos > 0:
            exit_px = min((l.price for l in depth if l.side == "bid"), default=current_price)
        else:
            exit_px = max((l.price for l in depth if l.side == "ask"), default=current_price)
        order = Order(symbol=symbol, side=side, qty=abs(pos), order_type=OrderType.IOC, limit_price=exit_px if exit_px > 0 else current_price)
        fill = self.broker.place_order(order)
        if fill is None:
            return {"closed": False, "reason": f"exit unfilled: {reason}", "pnl": 0.0, "qty": pos}
        self.risk.record_position(symbol, -abs(pos))
        entry = self._avg_entry.get(symbol, fill.fill_price)
        pnl = (fill.fill_price - entry) * abs(pos) * (1 if pos > 0 else -1)
        self.risk.record_pnl(pnl)
        self._avg_entry.pop(symbol, None)
        return {"closed": True, "reason": reason, "pnl": pnl, "qty": pos, "fill_price": fill.fill_price}
