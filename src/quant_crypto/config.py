"""Application configuration for the crypto engine, sourced from env / .env.

Mirrors the parent trading-engine: live trading is gated behind an explicit
flag and requires a daily loss limit and positive per-position sizing.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Default per-symbol position sizing in BASE units (what one "lot" of BTC/ETH
# means here). These are NOT Binance lot sizes — crypto spot trades in fractional
# units. They are the engine's trade unit (notional ~ $200 each at write time).
DEFAULT_QUANTITIES: dict[str, float] = {
    "BTCUSDT": 0.001,
    "ETHUSDT": 0.05,
}


class Settings(BaseSettings):
    """Global engine settings, sourced from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Broker (Binance) ---
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = Field(
        default=False,
        description="Route orders through Binance spot TESTNET instead of live.",
    )

    # --- LLM Providers (Tier 1 strategy) ---
    openai_api_key: str = ""
    openai_model: str = Field(default="gpt-4o")
    anthropic_api_key: str = ""
    anthropic_model: str = Field(default="claude-sonnet-5")
    anthropic_workspace_id: str = ""

    # --- Runtime / Risk ---
    live_trading: bool = False
    max_daily_loss: float = Field(default=500.0, gt=0)
    max_position: float = Field(
        default=0.001, gt=0, description="Per-symbol position cap in BASE units"
    )
    crypto_symbols: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["BTCUSDT", "ETHUSDT"],
        description="Spot symbols the engine records/trades (base quote pairs). "
        "Accepts a JSON list or a comma-separated string (CRYPTO_SYMBOLS=BTCUSDT,ETHUSDT).",
    )
    quantities: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_QUANTITIES),
        description="Trade unit (base units) per symbol for a single entry.",
    )

    # --- Model / signal ---
    prob_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    ring_buffer_size: int = Field(default=300, ge=1)
    model_stride: int = Field(default=25, ge=1)
    hurdle_rate: float = Field(default=1e-6, gt=0.0)

    # --- Economics (fee model, simplified for spot) ---
    # Binance spot: 0.1% taker fee each side (0.075% with BNB; adjust to yours).
    taker_fee_rate: float = Field(default=0.001, ge=0.0)
    max_trade_cost_pct: float = Field(
        default=0.05,
        description="Viability gate: refuse trades whose round-trip cost exceeds "
        "this fraction of the position notional.",
    )

    # --- Paper fill realism ---
    paper_slippage_bps: float = Field(default=5.0, ge=0.0)
    paper_slippage_spread_fraction: float = Field(default=0.5, ge=0.0)
    paper_starting_capital: float = Field(
        default=10_000.0,
        description="Notional paper capital in USDT. Reported by /api/v1/funds "
        "whenever LIVE_TRADING is off, labelled source='paper'.",
    )

    @field_validator("crypto_symbols", mode="before")
    @classmethod
    def _parse_symbols(cls, v):
        """Accept a comma-separated string, a JSON list, or a real list.

        The field is annotated NoDecode so pydantic-settings hands us the raw
        env string; we normalize here (a plain JSON list would otherwise crash
        the app with a SettingsError at startup).
        """
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("["):
                import json

                try:
                    return json.loads(s)
                except ValueError:
                    pass
            return [part.strip() for part in s.split(",") if part.strip()]
        return v

    @model_validator(mode="after")
    def _validate_live_requires_safeguards(self) -> "Settings":
        if self.live_trading and self.max_daily_loss <= 0:
            raise ValueError("LIVE_TRADING requires a positive MAX_DAILY_LOSS limit")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
