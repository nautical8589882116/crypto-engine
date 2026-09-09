"""Binance REST market data (klines) for corpus building and backtest."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

# https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
_REST = "https://api.binance.com/api/v3"


@dataclass
class Kline:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_ms: int
    taker_buy_base: float


def fetch_klines(
    symbol: str,
    interval: str = "1m",
    limit: int = 1000,
    *,
    client: httpx.Client | None = None,
) -> list[Kline]:
    """Fetch historical spot klines. Public endpoint — no auth needed."""
    own = client is None
    c = client or httpx.Client(timeout=15.0)
    try:
        r = c.get(
            f"{_REST}/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )
        r.raise_for_status()
        rows = r.json()
    finally:
        if own:
            c.close()
    out: list[Kline] = []
    for row in rows:
        out.append(
            Kline(
                open_time_ms=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time_ms=int(row[6]),
                taker_buy_base=float(row[9]),
            )
        )
    return out
