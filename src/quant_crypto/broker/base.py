"""Broker abstraction + PaperBroker + gated BinanceBroker.

Crypto spot trades in fractional base units, so qty is a float (e.g. 0.001 BTC).
PaperBroker fills against a synthetic book with adverse-only slippage so paper
never flatters itself. BinanceBroker is gated behind LIVE_TRADING=1 exactly like
the parent engine's DhanBroker: it refuses to construct unless the operator has
explicitly enabled live trading (and a testnet toggle for dry runs).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

from quant_crypto.config import get_settings
from quant_crypto.schemas.order import Order, OrderSide, OrderType


class Fill(BaseModel):
    order_id: str
    symbol: str
    side: str
    qty: float
    fill_price: float


class DepthLevel(BaseModel):
    price: float
    qty: float
    side: Literal["bid", "ask"] | None = None


class Broker(ABC):
    @abstractmethod
    def get_depth(self, symbol: str) -> list[DepthLevel]:
        ...

    @abstractmethod
    def place_order(self, order: Order) -> Fill | None:
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> float:
        ...


class PaperBroker(Broker):
    """Synthetic-book filler; frictionless when constructed bare (unit tests)."""

    def __init__(self, slippage_bps: float = 0.0, slippage_spread_fraction: float = 0.0):
        self._books: dict[str, list[DepthLevel]] = {}
        self._positions: dict[str, float] = {}
        self._counter = 0
        self._slippage_bps = slippage_bps
        self._slippage_spread_fraction = slippage_spread_fraction
        self._spreads: dict[str, float] = {}
        self.fill_records: list[dict] = []
        self.slippage_points_total = 0.0

    @classmethod
    def from_settings(cls, settings) -> "PaperBroker":
        return cls(
            slippage_bps=settings.paper_slippage_bps,
            slippage_spread_fraction=settings.paper_slippage_spread_fraction,
        )

    def set_depth(self, symbol: str, levels: list[DepthLevel]) -> None:
        self._books[symbol] = levels

    def set_quote(self, symbol: str, *, bid: float, ask: float, bid_qty: float, ask_qty: float) -> None:
        self._books[symbol] = [
            DepthLevel(price=bid, qty=max(bid_qty, 1e-9), side="bid"),
            DepthLevel(price=ask, qty=max(ask_qty, 1e-9), side="ask"),
        ]
        self._spreads[symbol] = max(ask - bid, 0.0)

    def get_depth(self, symbol: str) -> list[DepthLevel]:
        return self._books.get(symbol, [])

    def _apply_slippage(self, price: float, symbol: str, is_buy: bool) -> float:
        adverse = price * self._slippage_bps / 10_000.0
        adverse += self._spreads.get(symbol, 0.0) * self._slippage_spread_fraction
        return price + adverse if is_buy else price - adverse

    def place_order(self, order: Order) -> Fill | None:
        book = self._books.get(order.symbol, [])
        if not book:
            return None
        eligible = [l for l in book if l.side is None or (l.side == "ask" if order.side == OrderSide.BUY else l.side == "bid")]
        eligible = [l for l in eligible if (order.side == OrderSide.BUY and l.price <= order.limit_price) or (order.side == OrderSide.SELL and l.price >= order.limit_price)]
        eligible.sort(key=lambda l: l.price)  # best first
        remaining = order.qty
        filled_qty = 0.0
        total_cost = 0.0
        for l in eligible:
            if remaining <= 0:
                break
            take = min(l.qty, remaining)
            filled_qty += take
            total_cost += take * l.price
            remaining -= take
        if remaining > 0:
            return None  # FOK: cannot fill fully
        avg = total_cost / filled_qty if filled_qty else order.limit_price
        avg = self._apply_slippage(avg, order.symbol, is_buy=(order.side == OrderSide.BUY))
        self._counter += 1
        signed = filled_qty if order.side == OrderSide.BUY else -filled_qty
        self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + signed
        slip = abs(avg - (total_cost / filled_qty if filled_qty else order.limit_price))
        self.slippage_points_total += slip
        self.fill_records.append(
            {"order_id": f"paper-{self._counter}", "symbol": order.symbol, "side": order.side.value, "qty": filled_qty, "fill_price": avg}
        )
        return Fill(order_id=f"paper-{self._counter}", symbol=order.symbol, side=order.side.value, qty=filled_qty, fill_price=avg)

    def get_position(self, symbol: str) -> float:
        return self._positions.get(symbol, 0.0)


class BinanceBroker(Broker):
    """Real Binance spot execution, gated behind LIVE_TRADING=1."""

    def __init__(self, settings=None, client=None):
        self.settings = settings or get_settings()
        self._require_live()
        self.client = client or self._build_client()
        self._books: dict[str, list[DepthLevel]] = {}

    def _require_live(self) -> None:
        if not self.settings.live_trading:
            raise RuntimeError(
                "BinanceBroker refuses to construct: LIVE_TRADING must be 1 "
                "(operator sign-off after API-key approval)."
            )
        if not self.settings.binance_api_key or not self.settings.binance_api_secret:
            raise RuntimeError("BINANCE_API_KEY / BINANCE_API_SECRET are required for live trading.")

    def _build_client(self):
        from binance.client import Client

        return Client(
            self.settings.binance_api_key,
            self.settings.binance_api_secret,
            testnet=self.settings.binance_testnet,
        )

    def get_depth(self, symbol: str) -> list[DepthLevel]:
        try:
            d = self.client.get_order_book(symbol=symbol, limit=10)
            out = [DepthLevel(price=float(b[0]), qty=float(b[1]), side="bid") for b in d.get("bids", [])]
            out += [DepthLevel(price=float(a[0]), qty=float(a[1]), side="ask") for a in d.get("asks", [])]
            self._books[symbol] = out
        except Exception:  # noqa: BLE001
            pass
        return self._books.get(symbol, [])

    def place_order(self, order: Order) -> Fill | None:
        side = "BUY" if order.side == OrderSide.BUY else "SELL"
        tif = "FOK" if order.order_type == OrderType.FOK else "IOC"
        try:
            resp = self.client.create_order(
                symbol=order.symbol,
                side=side,
                type="LIMIT",
                timeInForce=tif,
                quantity=round(order.qty, 8),
                price=round(order.limit_price, 8),
            )
        except Exception:  # noqa: BLE001 - rejected / unfilled
            return None
        status = str(resp.get("status", ""))
        executed = float(resp.get("executedQty", 0.0) or 0.0)
        if status != "FILLED" or executed <= 0:
            return None
        fills = resp.get("fills", [])
        if fills:
            avg = sum(float(f.get("price", 0.0)) * float(f.get("qty", 0.0)) for f in fills) / executed
        else:
            avg = float(resp.get("price", order.limit_price))
        return Fill(
            order_id=str(resp.get("orderId", "binance")),
            symbol=order.symbol,
            side=side.lower(),
            qty=executed,
            fill_price=avg,
        )

    def get_position(self, symbol: str) -> float:
        base = symbol.replace("USDT", "")
        try:
            acct = self.client.get_account()
            for bal in acct.get("balances", []):
                if bal.get("asset") == base:
                    return float(bal.get("free", 0.0) or 0.0) + float(bal.get("locked", 0.0) or 0.0)
        except Exception:  # noqa: BLE001
            pass
        return 0.0
