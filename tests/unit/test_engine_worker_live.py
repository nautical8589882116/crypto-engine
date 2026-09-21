"""Verification for the LIVE broker seam in engine_worker.EngineWorkerCore.

The worker now accepts a `broker`; when attached, entries/exits route real
orders through the Broker contract instead of the paper sim, while `broker=None`
keeps the paper path byte-for-byte unchanged. These tests exercise the live
routing with a stub broker and confirm the paper path is untouched — so the
deployed paper run can never be changed by enabling the seam.
"""
from __future__ import annotations

import types

from quant_crypto.broker.base import Fill, PaperBroker
from quant_crypto.engine_worker import EngineWorkerCore


class FakeModel:
    seq_len = 300

    def predict(self, window):
        return 0.9  # always above threshold -> seeks entries


class StubBroker(PaperBroker):
    """PaperBroker subclass that always fills and records order calls."""

    def __init__(self):
        super().__init__()
        self.orders = []  # every Order passed to place_order
        self.n_fills = 0
        self._closable = {}

    def place_order(self, order):
        self.orders.append(order)
        self.n_fills += 1
        fill = Fill(
            order_id=f"stub-{self.n_fills}",
            symbol=order.symbol,
            side=order.side.value,
            qty=order.qty,
            fill_price=order.limit_price,
        )
        self.fill_records.append({
            "order_id": fill.order_id, "symbol": order.symbol,
            "side": order.side.value, "qty": order.qty, "fill_price": fill.fill_price,
        })
        return fill

    def get_position(self, symbol):
        return self._closable.get(symbol, 0.0)


def make_settings(**over):
    base = dict(
        max_position=0.001,
        max_daily_loss=500.0,
        prob_threshold=0.55,
        model_stride=25,
        quantities={"BTC-USD": 0.001, "ETH-USD": 0.05},
        position_caps={"BTC-USD": 0.001, "ETH-USD": 0.05},
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _tick(sym, i, ltp):
    from quant_crypto.data.binance_feed import Tick
    return Tick.model_validate({
        "symbol": sym, "ltp": ltp, "timestamp": 1789000000.0 + i,
        "bid": ltp - 0.01, "ask": ltp + 0.01, "bid_qty": 1.0, "ask_qty": 1.0,
        "volume": 0.001, "oi": 100.0,
    })


def _process_to_entry(core, sym, n=320):
    """Drive process() until an inference window fires (counts % stride == 0)."""
    for i in range(n):
        core.process(_tick(sym, i, 76500.0 + i))
        if core.tele["entries"] > 0:
            break


def test_live_core_reports_mode_live(tmp_path):
    core = EngineWorkerCore(FakeModel(), make_settings(), tmp_path, broker=StubBroker())
    assert core.tele["mode"] == "live"
    assert core.broker is not None


def test_paper_core_reports_mode_paper(tmp_path):
    core = EngineWorkerCore(FakeModel(), make_settings(), tmp_path)
    assert core.tele["mode"] == "paper"
    assert core.broker is None


def test_live_entry_routes_real_order_and_keeps_position(tmp_path):
    broker = StubBroker()
    core = EngineWorkerCore(FakeModel(), make_settings(), tmp_path, broker=broker)
    _process_to_entry(core, "BTC-USD")
    assert core.tele["entries"] == 1
    assert len(broker.orders) == 1
    assert broker.orders[0].symbol == "BTC-USD"
    # a real (not phantom) position is held
    assert "BTC-USD" in core.open_pos
    assert core.open_pos["BTC-USD"]["order_id"].startswith("stub-")


def test_live_exit_routes_sell_and_realizes_pnl(tmp_path):
    broker = StubBroker()
    core = EngineWorkerCore(FakeModel(), make_settings(), tmp_path, broker=broker)
    _process_to_entry(core, "BTC-USD")
    # force the price above TP -> triggers a live exit
    entry = core.open_pos["BTC-USD"]
    orders_before = len(broker.orders)
    for i in range(3):
        core.process(_tick("BTC-USD", 1000 + i, entry["tp"] + 1.0))
    assert len(broker.orders) == orders_before + 1  # a SELL was placed
    assert broker.orders[-1].side.value == "sell"
    assert core.tele["trades"] == 1
    assert "BTC-USD" not in core.open_pos


def test_paper_path_unchanged_without_broker(tmp_path):
    """Paper core must not place broker orders — the deployed default."""
    core = EngineWorkerCore(FakeModel(), make_settings(), tmp_path)
    _process_to_entry(core, "BTC-USD")
    assert core.tele["mode"] == "paper"
    assert core.tele["entries"] == 1
    assert core.tele["trades"] == 0  # still open, paper sim
    assert "BTC-USD" in core.open_pos
