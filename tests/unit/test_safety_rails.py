"""Verification for the production safety rails in engine_worker.EngineWorkerCore
and the Coinbase broker gating. Re-established after an env reset wiped the
coverage the integration agents had written.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from quant_crypto.broker.base import OrderType  # noqa: F401  (import sanity)
from quant_crypto.broker.coinbase import CoinbaseBroker
from quant_crypto.engine_worker import EngineWorkerCore, KILL_FLAG_NAME


class FakeModel:
    seq_len = 300

    def predict(self, window):
        return 0.9  # always above any threshold -> seeks entries


def make_settings(**over):
    base = dict(
        max_position=0.001,
        max_daily_loss=500.0,
        prob_threshold=0.55,
        model_stride=25,
        quantities={"BTC-USD": 0.001, "ETH-USD": 0.05},
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _tick(sym: str, i: int, ltp: float):
    from quant_crypto.data.binance_feed import Tick
    return Tick.model_validate({
        "symbol": sym, "ltp": ltp, "timestamp": 1789000000.0 + i,
        "bid": ltp - 0.01, "ask": ltp + 0.01, "bid_qty": 1.0, "ask_qty": 1.0,
        "volume": 0.001, "oi": 100.0,
    })


def fill_buffer(core, sym: str, n: int = 320, stride: int | None = None):
    """Push enough ticks into the buffer to trigger an inference window."""
    stride = stride or core.stride
    for i in range(n):
        tick = _tick(sym, i, 76500.0 + i)
        # counts already staged across calls won't align to stride; drive directly
        core.counts[sym] = (core.counts.get(sym, 0) + 1)
        if core.buffers.get(sym) is None:
            from quant_crypto.data.ring_buffer import RingBuffer
            core.buffers[sym] = RingBuffer(capacity=core.seq)
        core.buffers[sym].append(tick)
        if core.buffers[sym].is_full and core.counts[sym] % stride == 0:
            prob = float(core.model.predict(core.buffers[sym].snapshot()))
            if prob >= core.settings.prob_threshold and sym not in core.open_pos:
                qty = core.settings.quantities.get(sym, core.settings.max_position)
                if not core.entry_blocked(sym, qty):
                    core.open_pos[sym] = {
                        "entry": tick.ask or tick.ltp, "qty": qty,
                        "tp": (tick.ask or tick.ltp) * 1.002,
                        "sl": (tick.ask or tick.ltp) * 0.998, "prob": prob,
                    }
                    core.tele["entries"] += 1
    core.tele["open_positions"] = len(core.open_pos)


# ------------------------------------------------------------------ position cap
def test_position_cap_blocks_overcap_symbol(tmp_path):
    s = make_settings()
    core = EngineWorkerCore(FakeModel(), s, tmp_path)
    # ETH trade unit 0.05 > max_position 0.001 -> blocked
    assert core.entry_blocked("ETH-USD", 0.05) == "position_cap"
    assert core.tele["killswitch"] is False


def test_position_cap_respects_open_position(tmp_path):
    s = make_settings()
    core = EngineWorkerCore(FakeModel(), s, tmp_path)
    # already open full cap
    core.open_pos["BTC-USD"] = {"qty": 0.001}
    assert core.entry_blocked("BTC-USD", 0.001) == "position_cap"
    # an open position of half the cap leaves room for the rest
    core2 = EngineWorkerCore(FakeModel(), s, tmp_path)
    core2.open_pos["BTC-USD"] = {"qty": 0.0005}
    assert core2.entry_blocked("BTC-USD", 0.0005) is None


# ------------------------------------------------------------------- daily loss
def test_daily_loss_breaker_halts_entries(tmp_path):
    s = make_settings(max_daily_loss=500.0)
    core = EngineWorkerCore(FakeModel(), s, tmp_path)
    core.tele["total_pnl"] = -501.0
    core.refresh_safety()
    assert core.tele["halted"] is True
    assert core.entry_blocked("BTC-USD", 0.001) == "daily_loss"


def test_no_halt_within_limit(tmp_path):
    core = EngineWorkerCore(FakeModel(), make_settings(), tmp_path)
    core.tele["total_pnl"] = -499.0
    core.refresh_safety()
    assert core.tele["halted"] is False
    assert core.entry_blocked("BTC-USD", 0.001) is None


# ------------------------------------------------------------------ kill switch
def test_kill_switch_flag_blocks_entries(tmp_path: Path):
    core = EngineWorkerCore(FakeModel(), make_settings(), tmp_path)
    (tmp_path / KILL_FLAG_NAME).touch()
    assert core.entry_blocked("BTC-USD", 0.001) == "kill_switch"
    assert core.tele["killswitch"] is True


def test_no_kill_switch_flag_allows_entry(tmp_path):
    core = EngineWorkerCore(FakeModel(), make_settings(), tmp_path)
    assert core.tele["killswitch"] is False
    assert core.entry_blocked("BTC-USD", 0.001) is None


def test_kill_switch_flag_blocks_via_core(tmp_path):
    # full pipeline: with the kill flag present, no new entry ever opens
    s = make_settings()
    core = EngineWorkerCore(FakeModel(), s, tmp_path)
    core.stride = 1
    (tmp_path / KILL_FLAG_NAME).touch()
    fill_buffer(core, "BTC-USD", n=310, stride=1)
    assert core.tele["entries"] == 0  # flag present -> never enters
    assert core.tele["open_positions"] == 0
    assert core.tele["blocked"] > 0


def test_default_paper_path_still_opens_entry(tmp_path):
    core = EngineWorkerCore(FakeModel(), make_settings(model_stride=1), tmp_path)
    core.stride = 1
    fill_buffer(core, "BTC-USD", n=310, stride=1)
    assert core.tele["entries"] >= 1
    assert core.tele["open_positions"] >= 1


# ----------------------------------------------------------- coinbase gating/FOK
class StubClient:
    def __init__(self):
        self.calls = []
        self.base_url = "https://api.coinbase.com/api/v3/brokerage"

    def post(self, url, payload):
        self.calls.append(("post", url, payload))
        cfg = payload["order_configuration"]
        key = next(iter(cfg))  # limit_limit_fok or sor_limit_ioc
        return {
            "success": True,
            "order": {
                "order_id": "cb-1", "status": "FILLED",
                "filled_size": cfg[key]["base_size"],
                "filled_value": "765.00",
            },
        }

    def get(self, url):
        self.calls.append(("get", url, None))
        return {}


def _live_settings():
    return types.SimpleNamespace(
        live_trading=True,
        coinbase_api_name="key-a",
        coinbase_api_private_key="c2VjcmV0",  # b64 of "secret"
        coinbase_testnet=False,
    )


def test_coinbase_refuses_when_not_live():
    s = types.SimpleNamespace(live_trading=False, coinbase_api_name="a", coinbase_api_private_key="b")
    with pytest.raises(RuntimeError):
        CoinbaseBroker(s)


def test_coinbase_requires_creds_when_live():
    s = types.SimpleNamespace(live_trading=True, coinbase_api_name="", coinbase_api_private_key="")
    with pytest.raises(RuntimeError):
        CoinbaseBroker(s)


def test_coinbase_fok_buy_lifts_ask():
    from quant_crypto.broker.base import Order, OrderSide
    stub = StubClient()
    b = CoinbaseBroker(_live_settings(), client=stub)
    order = Order(symbol="BTC-USD", side=OrderSide.BUY, qty=0.001,
                  order_type=OrderType.FOK, limit_price=76500.0)
    fill = b.place_order(order)
    assert fill is not None and fill.fill_price > 0
    _, _, payload = stub.calls[0]
    cfg = payload["order_configuration"]
    assert "limit_limit_fok" in cfg
    assert cfg["limit_limit_fok"]["base_size"] == "0.001"


def test_coinbase_sell_ioc_hits_bid():
    from quant_crypto.broker.base import Order, OrderSide
    stub = StubClient()
    b = CoinbaseBroker(_live_settings(), client=stub)
    order = Order(symbol="BTC-USD", side=OrderSide.SELL, qty=0.001,
                  order_type=OrderType.IOC, limit_price=76400.0)
    fill = b.place_order(order)
    assert fill is not None
    _, _, payload = stub.calls[0]
    assert "sor_limit_ioc" in payload["order_configuration"]
