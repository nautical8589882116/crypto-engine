"""Live market-state derivation off the recorder stream (dashboard-facing).

Module-global, process-wide: the recorder publishes ticks here so the dashboard
shows real order-flow even before a model session exists. Honestly returns None
("no data yet") rather than fabricating zeros.
"""

from __future__ import annotations

import threading
from collections import deque

from quant_crypto.data.binance_feed import Tick

WINDOW = 300
_LOCK = threading.Lock()
_QUEUE: deque = deque(maxlen=WINDOW)
_SEEN = 0


def publish_tick(tick: Tick) -> None:
    global _SEEN
    with _LOCK:
        _QUEUE.append((tick.bid_qty, tick.ask_qty, tick.ltp, tick.bid, tick.ask))
        _SEEN += 1


def market_state_snapshot() -> dict | None:
    with _LOCK:
        if _SEEN < 2:
            return None
        total_bid = sum(q[0] for q in _QUEUE)
        total_ask = sum(q[1] for q in _QUEUE)
        ltp = _QUEUE[-1][2]
        bids = [q[3] for q in _QUEUE if q[3] > 0]
        asks = [q[4] for q in _QUEUE if q[4] > 0]
        bid = min(bids) if bids else None
        ask = max(asks) if asks else None
        ofi_num = sum(1 for q in _QUEUE if q[3] > 0 and q[4] > 0)
        return {
            "ltp": ltp,
            "bid": bid,
            "ask": ask,
            "spread": round(ask - bid, 4) if (bid is not None and ask is not None) else None,
            "taker_imbalance": round((total_bid - total_ask) / (total_bid + total_ask), 4)
            if (total_bid + total_ask) > 0 else None,
            "window": len(_QUEUE),
            "ticks_seen": _SEEN,
            "healthy": ofi_num > 0,
        }


def reset_market_state() -> None:
    global _SEEN
    with _LOCK:
        _QUEUE.clear()
        _SEEN = 0
