"""Coinbase spot market-data feed (public WebSocket, no auth).

Used as an alternative to `BinanceTickFeed` for hosting regions where Binance
returns HTTP 451 (geo-blocked). Coinbase is US-based and reachable from US cloud
IPs, and its `ticker` channel gives REAL top-of-book (bid/ask + sizes) — an
improvement over Binance agg-trades, which carry no depth.

Emits the same `Tick` model, so the rest of the engine is unchanged:
  ltp      = last trade price
  bid/ask  = best bid/ask from the book
  bid_qty  = best-bid size   (-> c0 order-flow imbalance, real book depth)
  ask_qty  = best-ask size
  volume   = 0.0             (ticker has no per-trade size)
  oi       = 24h traded volume (-> c2 volume-delta channel)
"""

from __future__ import annotations

import asyncio
import json
import time

from quant_crypto.data.binance_feed import Tick, _get_logger

COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"


def _f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class CoinbaseTickFeed:
    """Streams Coinbase `ticker` updates for one or more products (e.g. BTC-USD)."""

    def __init__(
        self,
        products: list[str],
        *,
        url: str = COINBASE_WS,
        reconnect_delay: float = 2.0,
        max_reconnects: int = 10,
    ):
        import websockets

        self._ws = websockets
        self.products = [p.upper() for p in products]
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        self._callbacks: list = []
        self.running = False

    def on_tick(self, cb) -> None:
        self._callbacks.append(cb)

    def _emit(self, tick: Tick) -> None:
        for cb in self._callbacks:
            try:
                cb(tick)
            except Exception:  # noqa: BLE001 - a consumer must not kill the feed
                pass

    def _on_message(self, msg: dict) -> None:
        if msg.get("type") != "ticker":
            return
        product = msg.get("product_id")
        if not product:
            return
        price = _f(msg.get("price"))
        if price <= 0:
            return
        bid = _f(msg.get("best_bid")) or price
        ask = _f(msg.get("best_ask")) or price
        self._emit(
            Tick(
                timestamp=time.time(),
                symbol=product,
                ltp=price,
                bid=bid,
                ask=ask,
                bid_qty=_f(msg.get("best_bid_size")),
                ask_qty=_f(msg.get("best_ask_size")),
                volume=0.0,
                oi=_f(msg.get("volume_24h")),
            )
        )

    async def start(self) -> None:
        self.running = True
        log = _get_logger()
        sub = json.dumps(
            {"type": "subscribe", "product_ids": self.products, "channels": ["ticker"]}
        )
        attempt = 0
        while self.running:
            try:
                async with self._ws.connect(self.url, ping_interval=20) as sock:
                    await sock.send(sub)
                    attempt = 0
                    log.info("coinbase ws connected: %s %s", self.url, self.products)
                    async for raw in sock:
                        if not self.running:
                            break
                        try:
                            if isinstance(raw, bytes):
                                raw = raw.decode("utf-8", "replace")
                            self._on_message(json.loads(raw))
                        except Exception:  # noqa: BLE001
                            continue
            except Exception as exc:  # noqa: BLE001 - reconnect loop
                if not self.running:
                    break
                log.warning("coinbase ws error (attempt %d): %r", attempt + 1, exc)
                attempt += 1
                if attempt > self.max_reconnects:
                    break
                await asyncio.sleep(self.reconnect_delay)
        self.running = False

    async def stop(self) -> None:
        self.running = False
