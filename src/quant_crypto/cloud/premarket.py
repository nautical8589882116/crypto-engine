"""Pre-market snapshot for the crypto strategy manager.

For crypto there is no single "opening bell" — markets trade 24/7. The snapshot
is whatever context the operator (or a scheduled job) feeds in: recent returns,
volatility, funding rate, order-flow, or a news flag. All fields default to 0.0
so a partial feed never breaks strategy selection.
"""

from __future__ import annotations

from pydantic import BaseModel


class PremarketSnapshot(BaseModel):
    """Market context consumed by the Tier 1 strategy manager."""

    btc_24h_return_pct: float = 0.0
    eth_24h_return_pct: float = 0.0
    btc_volatility_pct: float = 0.0      # realized vol, annualized-ish
    btc_funding_rate: float = 0.0        # perpetual funding (0 if spot-only)
    global_taker_imbalance: float = 0.0  # -1..1
    news_flag: str = ""                  # e.g. "none" | "macro" | "crypto-specific"
    source: str = "manual"
