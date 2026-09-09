"""Order-flow feature engineering for crypto spot (taker imbalance)."""

from __future__ import annotations

import numpy as np


def compute_ofi(bid_qty: float, ask_qty: float, prev_bid: float, prev_ask: float) -> float:
    """Normalized order-flow imbalance from rolling taker buy/sell volumes."""
    db = bid_qty - prev_bid
    da = ask_qty - prev_ask
    denom = abs(db) + abs(da)
    if denom <= 0:
        return 0.0
    return (db - da) / denom


def taker_imbalance(bid_qty: float, ask_qty: float) -> float | None:
    """Buy vs sell volume ratio in [-1, 1]; None when no sell volume."""
    if ask_qty <= 0:
        return None
    return (bid_qty - ask_qty) / (bid_qty + ask_qty)
