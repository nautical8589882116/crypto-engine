"""Unit tests: risk circuit-breakers (kill switch, daily loss, position cap)."""

from quant_crypto.exec.risk import (
    RiskLimits,
    RiskManager,
    reset_process_kill_switch,
    trip_process_kill_switch,
)


def _rm(**kw):
    lim = RiskLimits(max_daily_loss=100.0, max_position=0.002, **kw)
    return RiskManager(lim)


def test_allow_order_passes_when_clear():
    rm = _rm()
    ok, reason = rm.allow_order("BTCUSDT", 0.001)
    assert ok and reason is None


def test_kill_switch_blocks():
    reset_process_kill_switch()
    rm = _rm()
    trip_process_kill_switch()
    try:
        ok, reason = rm.allow_order("BTCUSDT", 0.001)
        assert not ok and reason == "kill_switch"
    finally:
        reset_process_kill_switch()


def test_daily_loss_halts():
    rm = _rm()
    rm.record_pnl(-150.0)
    assert rm.halted
    ok, reason = rm.allow_order("BTCUSDT", 0.001)
    assert not ok
    assert reason in ("daily_loss_limit", "halted")


def test_position_cap():
    rm = _rm()
    rm.record_position("BTCUSDT", 0.002)
    ok, reason = rm.allow_order("BTCUSDT", 0.001)
    assert not ok and reason == "position_cap"


def test_rate_limit():
    rm = _rm(min_order_interval=1000.0)
    rm.record_order_time()
    ok, reason = rm.allow_order("BTCUSDT", 0.001)
    assert not ok and reason == "rate_limit"
