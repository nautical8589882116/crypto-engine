"""Broker package."""

from quant_crypto.broker.base import BinanceBroker, Broker, DepthLevel, Fill, PaperBroker

try:
    from quant_crypto.broker.coinbase import CoinbaseBroker
except Exception:  # noqa: BLE001 - optional; keep package importable if net deps missing
    CoinbaseBroker = None  # type: ignore[assignment]

__all__ = ["Broker", "BinanceBroker", "PaperBroker", "CoinbaseBroker", "DepthLevel", "Fill"]
