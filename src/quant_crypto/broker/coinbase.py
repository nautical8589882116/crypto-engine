"""Coinbase Advanced Trade spot execution broker.

Implements the same Broker contract as PaperBroker / BinanceBroker against the
Coinbase Advanced Trade REST API (https://api.coinbase.com/api/v3/brokerage),
which is the execution venue the engine must use from Railway cloud because the
Binance API returns HTTP 451 (geo-blocked) from cloud IPs.

Like BinanceBroker, this is GATED: the constructor refuses to build unless the
operator has explicitly enabled live trading (LIVE_TRADING=1) AND supplied CDP
API credentials. It stays inert for paper runs — the running paper deploy never
constructs it (entrypoint.build_broker keeps PaperBroker unless live + coinbase
execution are enabled).

The HTTP client is injected (`client`) so unit tests pass a stub and never touch
the network. The real `CoinbaseHTTP` client signs private requests with a CDP
HMAC JWT (stdlib hmac/hashlib only — no heavy deps).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid

import requests

from quant_crypto.broker.base import Broker, DepthLevel, Fill
from quant_crypto.config import get_settings
from quant_crypto.schemas.order import Order, OrderSide, OrderType

DEFAULT_BASE_URL = "https://api.coinbase.com/api/v3/brokerage"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


class CoinbaseHTTP:
    """Minimal JSON HTTP client for Coinbase Advanced Trade.

    Private endpoints (accounts, orders) require a CDP HMAC JWT built from the
    API key name + private-key secret (base64). Public book needs no auth.
    """

    def __init__(self, api_name: str, api_private_key: str, *, testnet: bool = False, base_url: str = DEFAULT_BASE_URL):
        self.api_name = api_name or ""
        self.api_private_key = api_private_key or ""
        self.testnet = bool(testnet)
        self.base_url = base_url
        self.session = requests.Session()

    def _jwt_token(self, uri: str) -> str:
        name, secret = self.api_name, self.api_private_key
        header = {"alg": "HS256", "typ": "JWT", "kid": name}
        now = int(time.time())
        payload = {
            "iss": "cdp",
            "sub": name,
            "nbf": now - 30,
            "exp": now + 120,
            "uri": uri,
        }
        signing_input = f"{_b64url(json.dumps(header, separators=(',', ':')).encode('ascii'))}." \
                        f"{_b64url(json.dumps(payload, separators=(',', ':')).encode('ascii'))}"
        key_material = base64.b64decode(secret)
        sig = hmac.new(key_material, signing_input.encode("ascii"), hashlib.sha256).digest()
        return f"{signing_input}.{_b64url(sig)}"

    def _headers(self, uri: str) -> dict:
        return {
            "Authorization": f"Bearer {self._jwt_token(uri)}",
            "Content-Type": "application/json",
        }

    def get(self, url: str) -> dict:
        # Public endpoints need no auth; only sign when invoked via the private base path.
        path = url.split("?", 1)[0]
        headers = {}
        if "/orders" in path or "/accounts" in path or "/product_book" in path:
            headers = self._headers(path)
        resp = self.session.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def post(self, url: str, json: dict) -> dict:  # noqa: A002 - pydantic-style arg, matches test stub
        path = url.split("?", 1)[0]
        resp = self.session.post(url, headers=self._headers(path), json=json, timeout=20)
        resp.raise_for_status()
        return resp.json()


class CoinbaseBroker(Broker):
    """Real Coinbase Advanced Trade spot execution, gated behind LIVE_TRADING=1.

    BUY  LIMIT FOK lifts the ask (limit >= ask must hold).
    SELL IOC      hits the bid (limit <= bid must hold).
    Fractional qty in base units (e.g. 0.001 BTC).
    """

    def __init__(self, settings=None, client=None, *, base_url: str = DEFAULT_BASE_URL):
        self.settings = settings or get_settings()
        self._require_live()
        self.testnet = bool(self.settings.coinbase_testnet)
        self.client = client or CoinbaseHTTP(
            self.settings.coinbase_api_name,
            self.settings.coinbase_api_private_key,
            testnet=self.testnet,
            base_url=base_url,
        )
        self.base_url = getattr(self.client, "base_url", base_url)
        self._books: dict[str, list[DepthLevel]] = {}

    def _require_live(self) -> None:
        if not self.settings.live_trading:
            raise RuntimeError(
                "CoinbaseBroker refuses to construct: LIVE_TRADING must be 1 "
                "(operator sign-off after API-key approval)."
            )
        if not self.settings.coinbase_api_name or not self.settings.coinbase_api_private_key:
            raise RuntimeError(
                "COINBASE_API_NAME / COINBASE_API_PRIVATE_KEY are required for live Coinbase execution."
            )

    def get_depth(self, symbol: str) -> list[DepthLevel]:
        try:
            url = f"{self.base_url}/market/product_book?product_id={symbol}&limit=10"
            data = self.client.get(url)
            book = data.get("pricebook", {}) or {}
            bids = book.get("bids", []) or []
            asks = book.get("asks", []) or []
            out = [DepthLevel(price=float(b["price"]), qty=float(b["size"]), side="bid") for b in bids]
            out += [DepthLevel(price=float(a["price"]), qty=float(a["size"]), side="ask") for a in asks]
            self._books[symbol] = out
        except Exception:  # noqa: BLE001 - a stale/missing book is not fatal
            pass
        return self._books.get(symbol, [])

    def place_order(self, order: Order) -> Fill | None:
        side = "BUY" if order.side == OrderSide.BUY else "SELL"
        if order.order_type == OrderType.FOK:
            config_key = "limit_limit_fok"
        else:
            config_key = "sor_limit_ioc"
        # qty is fractional base units on a spot pair (e.g. 0.001 BTC).
        payload = {
            "client_order_id": str(uuid.uuid4()),
            "product_id": order.symbol,
            "side": side,
            "order_configuration": {
                config_key: {
                    "base_size": str(round(order.qty, 8)),
                    "limit_price": str(round(order.limit_price, 8)),
                }
            },
        }
        try:
            resp = self.client.post(f"{self.base_url}/orders", payload)
        except Exception:  # noqa: BLE001 - rejected / network / unfilled
            return None
        if not resp.get("success", True):
            return None
        o = resp.get("order") or {}
        status = str(o.get("status", ""))
        filled = float(o.get("filled_size", 0.0) or 0.0)
        filled_value = float(o.get("filled_value", 0.0) or 0.0)
        if filled <= 0:
            return None
        if order.order_type == OrderType.FOK and filled < order.qty - 1e-9:
            return None  # FOK must fill fully or it is rejected/unfilled
        avg = (filled_value / filled) if filled else float(order.limit_price)
        return Fill(
            order_id=str(o.get("order_id", "coinbase")),
            symbol=order.symbol,
            side=side.lower(),
            qty=filled,
            fill_price=avg,
        )

    def get_position(self, symbol: str) -> float:
        # Coinbase spot symbols are 'BASE-QUOTE' (e.g. 'BTC-USD'). Read the base
        # asset's spot balance (available + on-hold) from the account list.
        base = symbol.split("-", 1)[0] if "-" in symbol else symbol.replace("USDT", "")
        try:
            data = self.client.get(f"{self.base_url}/accounts")
            for acct in data.get("accounts", []) or []:
                if str(acct.get("currency", "")).upper() == base.upper():
                    avail = acct.get("available_balance", {}) or {}
                    hold = acct.get("hold", {}) or {}
                    return float(avail.get("value", 0.0) or 0.0) + float(hold.get("value", 0.0) or 0.0)
        except Exception:  # noqa: BLE001
            pass
        return 0.0