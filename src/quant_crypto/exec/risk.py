"""Risk circuit-breakers for the execution engine.

Enforces hard safety limits before any order is placed:
- daily loss limit (halts trading for the day once breached)
- max position cap (no accumulation beyond the configured maximum)
- kill-switch (global flag that disables all new trades)
- order-rate limiter (min interval between orders)

These are deterministic guards — no AI involved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RiskLimits:
    max_daily_loss: float = 5000.0
    max_position: float = 0.001  # base units (crypto fractional sizing)
    kill_switch: bool = False
    min_order_interval: float = 0.0  # seconds


# --- Process-global master kill-switch ------------------------------------
# Independent of any RiskManager instance: tripping it halts every manager in
# this process (dashboard button, MCP tool, or operator console all land here).
_PROCESS_KILL_SWITCH = False


def trip_process_kill_switch() -> None:
    """Trip the master kill-switch: all RiskManagers block new orders."""
    global _PROCESS_KILL_SWITCH
    _PROCESS_KILL_SWITCH = True


def reset_process_kill_switch() -> None:
    """Clear the master kill-switch (e.g. next trading day, after review)."""
    global _PROCESS_KILL_SWITCH
    _PROCESS_KILL_SWITCH = False


def is_process_kill_switch_tripped() -> bool:
    """True when the master kill-switch is active."""
    return _PROCESS_KILL_SWITCH


class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self._realized_pnl: float = 0.0
        self._positions: dict[str, float] = {}
        self._last_order_time: float = 0.0
        self.halted = False

    def check_kill_switch(self) -> bool:
        """Return True if trading is blocked by the kill-switch."""
        return self.limits.kill_switch or _PROCESS_KILL_SWITCH

    def check_daily_loss(self) -> bool:
        """Return True if the daily loss limit has been breached."""
        return self._realized_pnl <= -self.limits.max_daily_loss

    def check_position(self, symbol: str, add_qty: float) -> bool:
        """Return True if adding `add_qty` to `symbol` exceeds the cap."""
        current = self._positions.get(symbol, 0.0)
        return abs(current + add_qty) > self.limits.max_position

    def check_rate_limit(self) -> bool:
        """Return True if an order was placed too recently."""
        if self.limits.min_order_interval <= 0:
            return False
        return (time.time() - self._last_order_time) < self.limits.min_order_interval

    def allow_order(self, symbol: str, qty: float) -> tuple[bool, str | None]:
        """Full pre-trade gate; returns (allowed, reason_if_blocked)."""
        if self.halted:
            return False, "halted"
        if self.check_kill_switch():
            self.halted = True
            return False, "kill_switch"
        if self.check_daily_loss():
            self.halted = True
            return False, "daily_loss_limit"
        if self.check_position(symbol, qty):
            return False, "position_cap"
        if self.check_rate_limit():
            return False, "rate_limit"
        return True, None

    def record_pnl(self, pnl: float) -> None:
        self._realized_pnl += pnl
        if self.check_daily_loss():
            self.halted = True

    def record_position(self, symbol: str, delta: float) -> None:
        self._positions[symbol] = self._positions.get(symbol, 0.0) + delta

    def record_order_time(self) -> None:
        self._last_order_time = time.time()

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl
