"""Integration test: replay a tape through the full PaperEngine loop."""

import asyncio
import tempfile

from quant_crypto.config import get_settings
from quant_crypto.data.binance_feed import ReplayFeed, Tick
from quant_crypto.engine.paper import PaperEngine
from quant_crypto.schemas.regime import Regime
from quant_crypto.schemas.strategy import StrategyConfig


class FakeModel:
    seq_len = 200

    def predict(self, window):
        return 0.99  # strong uptrend tape => always tradeable


def _mk_ticks(n=1200, step=0.5):
    ticks = []
    for i in range(n):
        ltp = 100.0 + i * step
        ticks.append(Tick(timestamp=float(i), symbol="BTCUSDT", ltp=ltp,
                          bid=ltp - 0.05, ask=ltp + 0.05, bid_qty=2.0, ask_qty=1.0,
                          volume=0.5, oi=float(i)))
    return ticks


def test_paper_engine_replay_trades():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for t in _mk_ticks():
            f.write(t.model_dump_json() + "\n")
        path = f.name

    settings = get_settings()
    settings.quantities["BTCUSDT"] = 0.001
    engine = PaperEngine(settings, model=FakeModel(), model_stride=10,
                         take_profit_pct=0.001, stop_loss_pct=0.01,
                         publish_telemetry=False)
    result = asyncio.run(engine.run_realtime(ReplayFeed(path)))
    assert result.ticks_processed > 0
    assert result.entered_trades > 0
    assert any(t.exit_reason == "take_profit" for t in result.trades)


def test_paper_engine_applies_llm_strategy_tp_sl():
    """The LLM StrategyConfig's TP/SL override engine defaults on entry."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for t in _mk_ticks(1200):
            f.write(t.model_dump_json() + "\n")
        path = f.name

    settings = get_settings()
    settings.quantities["BTCUSDT"] = 0.001
    engine = PaperEngine(settings, model=FakeModel(), model_stride=10,
                         take_profit_pct=0.05, stop_loss_pct=0.05,  # wide defaults
                         publish_telemetry=False)
    engine.strategy = StrategyConfig(
        regime=Regime.TREND_FOLLOW, rationale="llm",
        take_profit_pct=0.001, stop_loss_pct=0.02,
    )
    result = asyncio.run(engine.run_realtime(ReplayFeed(path)))
    assert result.entered_trades > 0
    assert any(t.exit_reason == "take_profit" for t in result.trades)
