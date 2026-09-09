"""Unit tests: PaperBroker fills and FOK semantics."""

from quant_crypto.broker.base import PaperBroker
from quant_crypto.schemas.order import Order, OrderSide, OrderType


def test_fok_fills_when_depth_sufficient():
    pb = PaperBroker()
    pb.set_quote("BTCUSDT", bid=100.0, ask=100.1, bid_qty=1.0, ask_qty=1.0)
    o = Order(symbol="BTCUSDT", side=OrderSide.BUY, qty=0.001, order_type=OrderType.FOK, limit_price=100.1)
    f = pb.place_order(o)
    assert f is not None
    assert f.qty == 0.001
    assert pb.get_position("BTCUSDT") == 0.001


def test_fok_rejects_when_depth_insufficient():
    pb = PaperBroker()
    pb.set_quote("BTCUSDT", bid=100.0, ask=100.1, bid_qty=0.0005, ask_qty=0.0005)
    o = Order(symbol="BTCUSDT", side=OrderSide.BUY, qty=0.001, order_type=OrderType.FOK, limit_price=100.1)
    assert pb.place_order(o) is None
    assert pb.get_position("BTCUSDT") == 0.0


def test_sell_updates_position_negative():
    pb = PaperBroker()
    pb.set_quote("BTCUSDT", bid=100.0, ask=100.1, bid_qty=1.0, ask_qty=1.0)
    o = Order(symbol="BTCUSDT", side=OrderSide.SELL, qty=0.001, order_type=OrderType.FOK, limit_price=100.0)
    f = pb.place_order(o)
    assert f is not None
    assert pb.get_position("BTCUSDT") == -0.001


def test_frictionless_by_default():
    pb = PaperBroker()
    pb.set_quote("BTCUSDT", bid=100.0, ask=100.0, bid_qty=1.0, ask_qty=1.0)
    f = pb.place_order(Order(symbol="BTCUSDT", side=OrderSide.BUY, qty=1.0, limit_price=100.0))
    assert f.fill_price == 100.0
    assert pb.slippage_points_total == 0.0
