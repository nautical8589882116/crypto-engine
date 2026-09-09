"""Strategy Manager — the Tier 1 brain for crypto.

Periodically (and post-trade) it ingests market context, asks the LLM to select
a market regime and execution parameters, validates the StrategyConfig, and
persists it to the librarian. The resulting take-profit/stop/hurdle values can
then be applied to the paper/live engine.
"""

from __future__ import annotations

from quant_crypto.cloud.llm import LLMClient
from quant_crypto.cloud.librarian import Librarian
from quant_crypto.cloud.premarket import PremarketSnapshot
from quant_crypto.schemas.strategy import StrategyConfig

_SYSTEM_PROMPT = """You are a quantitative strategy manager for spot cryptocurrency
(BTC/ETH). Given market context, select exactly one regime and produce a JSON
strategy configuration for a long-only, momentum-following model with
take-profit/stop-loss exits.

Regimes:
- trend_follow: clear directional momentum, follow it
- low_vol_chop: low volatility, range-bound, tighten take-profit
- high_vol_breakout: elevated volatility, expect expansion
- range_mean_revert: range-bound, fade extremes
- news_spike: abnormal move likely news-driven, reduce risk

Respond with ONLY a JSON object with keys:
regime, rationale (str), weights (dict of float), hurdle_rate (float),
take_profit_pct (float > 0), stop_loss_pct (float > 0),
max_slippage_bps (float > 0).
Keep take_profit_pct and stop_loss_pct as small fractions (e.g. 0.005 = 0.5%).
"""


def _build_messages(snapshot: PremarketSnapshot) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                f"Market context: BTC 24h={snapshot.btc_24h_return_pct:+.2f}% "
                f"vol={snapshot.btc_volatility_pct:.2f}% funding={snapshot.btc_funding_rate:+.5f} | "
                f"ETH 24h={snapshot.eth_24h_return_pct:+.2f}% | "
                f"taker_imbalance={snapshot.global_taker_imbalance:+.2f} news={snapshot.news_flag or 'none'}."
            ),
        }
    ]


class StrategyManager:
    def __init__(self, llm: LLMClient, librarian: Librarian | None = None):
        self.llm = llm
        self.librarian = librarian

    def run_daily(self, snapshot: PremarketSnapshot) -> StrategyConfig:
        raw = self.llm.complete_json(_SYSTEM_PROMPT, _build_messages(snapshot))
        cfg = StrategyConfig.model_validate(raw)
        if self.librarian is not None:
            self.librarian.save_strategy(cfg)
        _publish_llm_decision(
            "REGIME",
            f"LLM regime — BTC {snapshot.btc_24h_return_pct:+.2f}% "
            f"vol {snapshot.btc_volatility_pct:.2f}% → {cfg.regime.value} | "
            f"tp {cfg.take_profit_pct:.3f} sl {cfg.stop_loss_pct:.3f} "
            f"hurdle {cfg.hurdle_rate} slip {cfg.max_slippage_bps}bps",
        )
        return cfg

    def adapt_after_trade(self, audits: list, snapshot: PremarketSnapshot) -> StrategyConfig:
        """Post-trade: feed trade/audit data back to the LLM to adjust params."""
        msgs = _build_messages(snapshot)
        audit_json = [a.model_dump() for a in audits]
        msgs.append(
            {
                "role": "user",
                "content": (
                    "Post-trade audit. Adjust execution parameters "
                    f"(hurdle_rate, take_profit_pct, stop_loss_pct, weights) based on "
                    f"this trade data: {audit_json}"
                ),
            }
        )
        raw = self.llm.complete_json(_SYSTEM_PROMPT, msgs)
        cfg = StrategyConfig.model_validate(raw)
        if self.librarian is not None:
            self.librarian.save_strategy(cfg)
        n = len(audits)
        slips = [getattr(a, "slippage_bps", 0.0) for a in audits]
        avg = sum(slips) / n if n else 0.0
        _publish_llm_decision(
            "ADAPT",
            f"LLM post-trade (n={n}, avg slip {avg:.1f}bps) → {cfg.regime.value} | "
            f"tp {cfg.take_profit_pct:.3f} sl {cfg.stop_loss_pct:.3f}",
        )
        return cfg


def _publish_llm_decision(kind: str, txt: str) -> None:
    """Surface an LLM decision in the dashboard feed ([LLM] rows)."""
    try:
        from quant_crypto.engine.events import publish_feed_event

        publish_feed_event("llm", "LLM", "ok", txt)
    except Exception:  # noqa: BLE001 - a feed hiccup must not break selection
        pass
