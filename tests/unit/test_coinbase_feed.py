"""Unit tests: Coinbase feed + market-data source selection."""

from quant_crypto.data.coinbase_feed import CoinbaseTickFeed


def _ticker(product="BTC-USD", price="76908.39", side="buy",
            bid="76908.39", bid_sz="0.03", ask="76908.40", ask_sz="0.41"):
    return {
        "type": "ticker", "product_id": product, "price": price, "side": side,
        "best_bid": bid, "best_bid_size": bid_sz,
        "best_ask": ask, "best_ask_size": ask_sz, "volume_24h": "7416.9",
    }


def test_coinbase_ticker_to_tick():
    feed = CoinbaseTickFeed(["BTC-USD"])
    seen = []
    feed.on_tick(seen.append)
    feed._on_message(_ticker())
    assert len(seen) == 1
    t = seen[0]
    assert t.symbol == "BTC-USD"
    assert t.ltp == 76908.39
    assert t.bid == 76908.39
    assert t.ask == 76908.40
    assert t.bid_qty == 0.03
    assert t.ask_qty == 0.41
    assert t.oi == 7416.9


def test_coinbase_ignores_non_ticker_and_bad_price():
    feed = CoinbaseTickFeed(["BTC-USD"])
    seen = []
    feed.on_tick(seen.append)
    feed._on_message({"type": "subscriptions", "channels": []})
    feed._on_message(_ticker(price="0"))
    feed._on_message({"type": "ticker"})  # no product
    assert seen == []


def test_symbol_mapping_coinbase_from_usdt():
    from quant_crypto.config import Settings
    from quant_crypto.recorder_worker import _make_feed

    s = Settings()
    s.market_data_source = "coinbase"
    feed, resolved = _make_feed(s, ["BTCUSDT", "ETHUSDT"])
    assert resolved == ["BTC-USD", "ETH-USD"]
    assert feed.products == ["BTC-USD", "ETH-USD"]
