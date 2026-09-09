"""Order domain models for the crypto engine."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    FOK = "fok"
    IOC = "ioc"


class Order(BaseModel):
    """An order to execute. qty is in BASE units (e.g. BTC)."""

    symbol: str
    side: OrderSide
    qty: float = Field(gt=0)
    order_type: OrderType = OrderType.FOK
    limit_price: float = Field(gt=0)


class Signal(BaseModel):
    """Local model signal emitted by the inference runtime."""

    symbol: str = "BTCUSDT"
    regime: str
    direction: str  # 'long' | 'short' | 'flat'
    prob_velocity: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: str

    @property
    def is_tradeable(self) -> bool:
        return self.prob_velocity >= self.threshold and self.direction != "flat"


class TradeAudit(BaseModel):
    """Post-trade execution-friction metrics for the self-improvement loop."""

    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    expected_price: float
    slippage_bps: float
    execution_friction_bps: float
    pnl: float
    closed_at: str

    @property
    def net_friction_bps(self) -> float:
        return self.slippage_bps + self.execution_friction_bps
