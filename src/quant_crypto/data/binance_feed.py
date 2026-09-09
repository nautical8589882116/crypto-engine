"""Binance spot feed abstraction — WebSocket agg-trades + klines.

Produces the engine's `Tick` model (same 8-field layout the ring buffer and
`window.py` channels expect) from the Binance public WebSocket market stream.

Spot crypto has no open interest and (without the depth stream) no real
bid/ask, so the Tick's derived fields are:
  ltp       = agg-trade price
  volume    = agg-trade quantity
  bid/ask   = synthetic micro-book from the rolling min/max of recent trade
              prices (a coarse but honest spread for paper fills)
  bid_qty   = rolling taker-BUY volume  (order-flow imbalance)
  ask_qty   = rolling taker-SELL volume (order-flow imbalance)
  oi        = cumulative traded volume  (so window channel c2 = volume delta)

This makes the three Mamba channels, from window.py:
  c0 = taker buy/sell imbalance (OFI)   c1 = price delta   c2 = volume delta
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Any

from pydantic import BaseModel

SYMBOL_STREAM_TEMPLATE = "{symbol}@aggTrade/{symbol}@kline_1s"


class Tick(BaseModel):
    """A normalized market tick (epoch-second timestamp)."""

    timestamp: float
    symbol: str
    ltp: float
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    volume: float
    oi: float


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class BinanceTickFeed:
    """Streams agg-trades + 1s klines for one or more spot symbols.

    WebSocket per Binance docs: wss://stream.binance.com:9443/stream?streams=...
    """

    def __init__(
        self,
        symbols: list[str],
        *,
        url: str = "wss://stream.binance.com:9443",
        window: int = 200,
        reconnect_delay: float = 2.0,
        max_reconnects: int = 10,
    ):
        import websockets

        self._ws = websockets  # lazy import keeps package import light
        self.symbols = [s.lower() for s in symbols]
        self.window = window
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        # per-symbol rolling state
        self._prices: dict[str, deque[float]] = {}
        self._buy_vol: dict[str, deque[float]] = {}
        self._sell_vol: dict[str, deque[float]] = {}
        self._cum_vol: dict[str, float] = {}
        self._callbacks: list[Any] = []
        self.running = False
        for s in self.symbols:
            self._init_symbol(s)

    def _init_symbol(self, sym: str) -> None:
        if sym not in self._prices:
            self._prices[sym] = deque(maxlen=self.window)
            self._buy_vol[sym] = deque(maxlen=self.window)
            self._sell_vol[sym] = deque(maxlen=self.window)
            self._cum_vol[sym] = 0.0

    def on_tick(self, cb: Any) -> None:
        self._callbacks.append(cb)

    def _streams(self) -> list[str]:
        out = []
        for s in self.symbols:
            out.append(f"{s}@aggTrade")
            out.append(f"{s}@kline_1s")
        return out

    def _emit(self, tick: Tick) -> None:
        for cb in self._callbacks:
            try:
                cb(tick)
            except Exception:  # noqa: BLE001 - a consumer must not kill the feed
                pass

    def _on_message(self, raw: dict) -> None:
        stream = raw.get("stream", "")
        data = raw.get("data", {})
        etype = data.get("e")
        sym_raw = data.get("s", "")
        sym = sym_raw.lower()
        if not sym or sym not in self._prices:
            return
        if etype == "aggTrade":
            self._handle_trade(sym, data)
        elif etype == "kline":
            self._handle_kline(sym, data.get("k", {}))

    def _handle_trade(self, sym: str, d: dict) -> None:
        price = _f(d.get("p"))
        qty = _f(d.get("q"))
        ts = _f(d.get("E")) / 1000.0 or time.time()
        is_buyer_maker = bool(d.get("m"))
        self._prices[sym].append(price)
        if is_buyer_maker:  # buyer was maker => seller aggressed
            self._sell_vol[sym].append(qty)
        else:
            self._buy_vol[sym].append(qty)
        self._cum_vol[sym] += qty
        prices = list(self._prices[sym])
        bid = min(prices) if prices else price
        ask = max(prices) if prices else price
        tick = Tick(
            timestamp=ts,
            symbol=sym.upper(),
            ltp=price,
            bid=bid,
            ask=ask,
            bid_qty=sum(self._buy_vol[sym]),
            ask_qty=sum(self._sell_vol[sym]),
            volume=qty,
            oi=self._cum_vol[sym],
        )
        self._emit(tick)

    def _handle_kline(self, sym: str, k: dict) -> None:
        # 1s klines confirm the session is alive and provide bar context;
        # the engine trades off agg-trades. Nothing to emit here by default.
        pass

    async def start(self) -> None:
        self.running = True
        # Because Binance requires stream names lowercased, build the URL.
        streams = "/".join(self._streams())  # Binance combined streams use '/'
        ws_url = f"{self.url}/stream?streams={streams}"
        attempt = 0
        while self.running:
            try:
                async with self._ws.connect(ws_url, ping_interval=20) as sock:
                    attempt = 0
                    async for msg in sock:
                        if not self.running:
                            break
                        try:
                            if isinstance(msg, bytes):
                                msg = msg.decode("utf-8", "replace")
                            self._on_message(json.loads(msg))
                        except Exception:  # noqa: BLE001
                            continue
            except Exception:  # noqa: BLE001 - reconnect loop
                if not self.running:
                    break
                attempt += 1
                if attempt > self.max_reconnects:
                    break
                await asyncio.sleep(self.reconnect_delay)
        self.running = False

    async def stop(self) -> None:
        self.running = False


class ReplayFeed:
    """Replays recorded JSONL ticks into a callable feed (for paper/backtest)."""

    def __init__(self, path: str, delay: float = 0.0, max_ticks: int | None = None):
        self.path = path
        self.delay = delay
        self.max_ticks = max_ticks
        self._callbacks: list[Any] = []
        self.running = False

    def on_tick(self, cb: Any) -> None:
        self._callbacks.append(cb)

    async def start(self) -> None:
        self.running = True
        n = 0
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                tick = Tick.model_validate_json(line)
                for cb in self._callbacks:
                    try:
                        cb(tick)
                    except Exception:  # noqa: BLE001
                        continue
                n += 1
                if self.max_ticks is not None and n >= self.max_ticks:
                    break
                if self.delay:
                    await asyncio.sleep(self.delay)
        self.running = False

    async def stop(self) -> None:
        self.running = False
