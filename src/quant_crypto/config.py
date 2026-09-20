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
# units. They are the engine's trade unit (notional ~ $50-150 each at write time).
DEFAULT_QUANTITIES: dict[str, float] = {
    "BTC-USD": 0.001,
    "ETH-USD": 0.05,
    "SOL-USD": 1.0,
    "XRP-USD": 100.0,
    "ADA-USD": 100.0,
    "DOGE-USD": 500.0,
    "LINK-USD": 5.0,
    "LTC-USD": 1.0,
    "AVAX-USD": 3.0,
    "BCH-USD": 0.5,
}

# Per-symbol position cap in BASE units. A symbol not listed here falls back to
# the global `max_position`. Set to the trade unit so each symbol holds at most
# one position (no accumulation). This fixes the old bug where a single global
# cap (0.001) silently blocked every non-BTC symbol (e.g. ETH 0.05) from entering.
DEFAULT_POSITION_CAPS: dict[str, float] = dict(DEFAULT_QUANTITIES)


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

    # --- Broker (Coinbase Advanced Trade) ---
    # All default-empty / False so the running paper service is unaffected.
    coinbase_api_name: str = ""
    coinbase_api_private_key: str = ""
    coinbase_execution: bool = Field(
        default=False,
        description="Enable real Coinbase spot execution (inert unless LIVE_TRADING=1 "
        "and market_data_source=coinbase).",
    )
    coinbase_testnet: bool = Field(
        default=False,
        description="Route orders through Coinbase sandbox / dry-run instead of live.",
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
        default_factory=lambda: ["BTC-USD", "ETH-USD"],
        description="Spot symbols the engine records/trades (base quote pairs). "
        "Accepts a JSON list or a comma-separated string (CRYPTO_SYMBOLS=BTC-USD,ETH-USD).",
    )
    market_data_source: str = Field(
        default="coinbase",
        description="Market-data feed: 'binance' (aggTrade+klines) or 'coinbase' "
        "(ticker, real book). Use 'coinbase' where Binance returns HTTP 451 "
        "(geo-blocked cloud regions).",
    )
    quantities: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_QUANTITIES),
        description="Trade unit (base units) per symbol for a single entry.",
    )
    position_caps: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_POSITION_CAPS),
        description="Per-symbol position cap in BASE units; a symbol not listed "
        "falls back to `max_position`. Accepts a JSON object (POSITION_CAPS).",
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
