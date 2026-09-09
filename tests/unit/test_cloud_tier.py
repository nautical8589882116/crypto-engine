"""Tests for the Tier 1 cloud LLM strategy layer + librarian + feedback loop."""

import pytest

from quant_crypto.cloud.librarian import Librarian
from quant_crypto.cloud.llm import _parse_json
from quant_crypto.cloud.premarket import PremarketSnapshot
from quant_crypto.cloud.strategy_manager import StrategyManager
from quant_crypto.feedback.loop import FeedbackLoop
from quant_crypto.schemas.order import TradeAudit
from quant_crypto.schemas.regime import Regime
from quant_crypto.schemas.strategy import StrategyConfig


class FakeLLM:
    def __init__(self, cfg: dict):
        self._cfg = cfg

    def complete_json(self, system, messages):
        return self._cfg


def test_librarian_strategy_roundtrip():
    lib = Librarian()
    cfg = StrategyConfig(regime=Regime.TREND_FOLLOW, rationale="r",
                         take_profit_pct=0.01, stop_loss_pct=0.005)
    lib.save_strategy(cfg)
    got = lib.latest_strategy()
    assert got is not None
    assert got.regime == Regime.TREND_FOLLOW
    assert got.take_profit_pct == 0.01
    assert got.stop_loss_pct == 0.005
    lib.close()


def test_librarian_audit_roundtrip():
    lib = Librarian()
    a = TradeAudit(trade_id="t1", symbol="BTCUSDT", direction="long", entry_price=100,
                   exit_price=102, expected_price=101, slippage_bps=2.0,
                   execution_friction_bps=1.0, pnl=2.0, closed_at="2026-09-09T00:00:00Z")
    lib.log_audit(a)
    out = lib.list_audits()
    assert len(out) == 1
    assert out[0].trade_id == "t1"
    assert out[0].net_friction_bps == 3.0
    lib.close()


def test_librarian_regime():
    lib = Librarian()
    lib.log_regime("2026-09-09", Regime.HIGH_VOL_BREAKOUT, "vol spiked")
    date, regime = lib.latest_regime()
    assert regime == Regime.HIGH_VOL_BREAKOUT
    lib.close()


def test_parse_json_tolerates_fences():
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json('prefix {"regime": "trend_follow"} suffix')["regime"] == "trend_follow"


def test_strategy_manager_run_daily_validates_and_persists():
    lib = Librarian()
    fake = FakeLLM({
        "regime": "trend_follow", "rationale": "momentum",
        "weights": {"ofi": 0.7}, "hurdle_rate": 1e-6,
        "take_profit_pct": 0.005, "stop_loss_pct": 0.003, "max_slippage_bps": 8,
    })
    mgr = StrategyManager(fake, librarian=lib)
    snap = PremarketSnapshot(btc_24h_return_pct=1.2, btc_volatility_pct=2.0)
    cfg = mgr.run_daily(snap)
    assert cfg.regime == Regime.TREND_FOLLOW
    assert cfg.take_profit_pct == 0.005
    assert lib.latest_strategy() is not None
    lib.close()


def test_feedback_loop_records_and_adapts():
    lib = Librarian()
    fake = FakeLLM({"regime": "range_mean_revert", "rationale": "chop",
                    "take_profit_pct": 0.003, "stop_loss_pct": 0.004, "max_slippage_bps": 6})
    mgr = StrategyManager(fake, librarian=lib)
    loop = FeedbackLoop(mgr, lib)
    loop.record_trade("t1", "BTCUSDT", "long", 100, 102, 101, "2026-09-09T00:00:00Z",
                      slippage_bps=2.0, execution_friction_bps=1.0, pnl=2.0)
    assert len(lib.list_audits()) == 1
    cfg = loop.adapt(PremarketSnapshot())
    assert cfg.regime == Regime.RANGE_MEAN_REVERT
    lib.close()


def test_strategy_manager_rejects_bad_llm_output():
    fake = FakeLLM({"regime": "not_a_regime", "rationale": "x",
                    "take_profit_pct": -1.0, "stop_loss_pct": 0.001})
    mgr = StrategyManager(fake, librarian=None)
    with pytest.raises(Exception):
        mgr.run_daily(PremarketSnapshot())
