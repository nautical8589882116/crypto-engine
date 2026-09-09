"""IOC liquidation sweep.

When a trade fails its hurdle or time-stop, aggressively chase bids (IOC) to
exit cleanly. The sweep respects a max-slippage cap — it will not sell below
a floor price.
"""

from __future__ import annotations

from dataclasses import dataclass

from quant_crypto.broker.base import Broker, DepthLevel
from quant_crypto.schemas.order import Order, OrderSide, OrderType

# The Rust hot path (quant_crypto._kernel, built via maturin) is the preferred
# implementation of the sweep planner below. Falls back to the pure-Python
# reference (_plan_sweep_py) when the extension module is not built.
try:
    from quant_crypto import _kernel as _rust_kernel  # type: ignore[attr-defined]

    _KERNEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on unbuilt installs
    _rust_kernel = None  # type: ignore[assignment]
    _KERNEL_AVAILABLE = False


@dataclass
class SweepResult:
    qty_remaining: int
    total_filled: int
    stopped_by_cap: bool
    avg_fill_price: float | None = None


def plan_sweep(
    book: list[DepthLevel],
    qty: int,
    is_buy: bool,
    floor_price: float,
) -> SweepResult:
    """Simulate an IOC sweep against the book, respecting a price floor/cap.

    For a sell (is_buy=False), walk bids downward until floor_price. For a
    buy, walk asks upward until floor_price acts as a ceiling.

    Dispatches to the compiled Rust `quant_crypto._kernel.plan_sweep` when available;
    otherwise uses the pure-Python reference implementation.
    """
    if _KERNEL_AVAILABLE and _rust_kernel is not None:
        levels = [(lvl.price, lvl.qty) for lvl in book]
        remaining, filled, stopped = _rust_kernel.plan_sweep(levels, qty, is_buy, floor_price)
        return SweepResult(
            qty_remaining=remaining,
            total_filled=filled,
            stopped_by_cap=stopped,
        )
    return _plan_sweep_py(book, qty, is_buy, floor_price)


def _plan_sweep_py(
    book: list[DepthLevel],
    qty: int,
    is_buy: bool,
    floor_price: float,
) -> SweepResult:
    """Pure-Python reference implementation (used for Rust parity testing)."""
    levels = sorted(book, key=lambda l: l.price, reverse=not is_buy)
    remaining = qty
    total_filled = 0
    stopped = False

    for lvl in levels:
        if remaining <= 0:
            break
        if is_buy:
            if lvl.price > floor_price:
                stopped = True
                break
        else:
            if lvl.price < floor_price:
                stopped = True
                break
        take = min(lvl.qty, remaining)
        total_filled += take
        remaining -= take

    return SweepResult(
        qty_remaining=remaining,
        total_filled=total_filled,
        stopped_by_cap=stopped,
    )


class Liquidator:
    def __init__(self, broker: Broker):
        self.broker = broker

    def sweep(self, symbol: str, qty: int, side: OrderSide, floor_price: float) -> SweepResult:
        """Place IOC orders to unwind `qty`; returns the remaining quantity.

        Reports the broker's actual average fill price so callers can compute
        realized PnL (None when nothing filled).
        """
        book = self.broker.get_depth(symbol)
        is_buy = side is OrderSide.BUY
        result = plan_sweep(book, qty, is_buy, floor_price)
        avg_fill_price: float | None = None
        if result.total_filled > 0:
            fill = self.broker.place_order(
                Order(
                    symbol=symbol,
                    side=side,
                    qty=result.total_filled,
                    order_type=OrderType.IOC,
                    limit_price=floor_price,
                )
            )
            if fill is not None:
                avg_fill_price = fill.fill_price
        return SweepResult(
            qty_remaining=result.qty_remaining,
            total_filled=result.total_filled,
            stopped_by_cap=result.stopped_by_cap,
            avg_fill_price=avg_fill_price,
        )
