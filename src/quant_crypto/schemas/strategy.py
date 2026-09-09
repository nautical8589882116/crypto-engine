"""Strategy configuration produced by the cloud LLM strategy manager."""

from __future__ import annotations

from pydantic import BaseModel, Field

from quant_crypto.schemas.regime import Regime


class StrategyConfig(BaseModel):
    """Configuration pushed from cloud to runtime to hot-swap model weights/rules."""

    regime: Regime
    rationale: str = Field(..., description="LLM's explanation for the regime choice")
    weights: dict[str, float] = Field(
        default_factory=dict, description="Active model/feature weights to hot-swap"
    )
    hurdle_rate: float = Field(
        default=0.0, ge=0.0, description="Minimum break-even velocity (price/ticks)"
    )
    take_profit_pct: float = Field(
        default=0.005, gt=0.0, description="Take-profit as fraction of entry price"
    )
    stop_loss_pct: float = Field(
        default=0.003, gt=0.0, description="Stop-loss as fraction of entry price"
    )
    max_slippage_bps: float = Field(
        default=10.0, gt=0.0, description="Max acceptable slippage in basis points"
    )
    valid_from: str | None = None
