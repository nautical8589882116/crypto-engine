"""Unit tests: Binance feed handler + URL building."""

from quant_crypto.data.binance_feed import BinanceTickFeed, Tick


def _agg(sym="BTCUSDT", price="79400.0", qty="0.001", maker=False, E=1788956000000):
    return {
        "stream": f"{sym.lower()}@aggTrade",
        "data": {"e": "aggTrade", "E": E, "s": sym, "p": price, "q": qty, "m": maker},
    }


def test_feed_builds_combined_url_with_slashes():
    feed = BinanceTickFeed(["BTCUSDT", "ETHUSDT"])
    streams = feed._streams()
    assert "/" in "/".join(streams)
    assert "btcusdt@aggTrade" in streams
    assert "ethusdt@kline_1s" in streams
    assert all(s.split("@")[0].islower() for s in streams)


def test_feed_normalizes_agg_trade_to_tick():
    feed = BinanceTickFeed(["BTCUSDT"])
    seen = []
    feed.on_tick(seen.append)
    feed._on_message(_agg(maker=False))
    assert len(seen) == 1
    t: Tick = seen[0]
    assert t.symbol == "BTCUSDT"
    assert t.ltp == 79400.0
    assert t.bid_qty > 0  # taker buy recorded
    assert t.ask_qty == 0
    assert t.oi > 0  # cumulative volume


def test_feed_splits_taker_side():
    feed = BinanceTickFeed(["BTCUSDT"])
    seen = []
    feed.on_tick(seen.append)
    feed._on_message(_agg(price="100", qty="1", maker=False))  # taker buy
    feed._on_message(_agg(price="101", qty="2", maker=True))  # taker sell
    assert seen[0].bid_qty == 1.0
    assert seen[0].ask_qty == 0.0
    assert seen[1].ask_qty == 2.0
    assert seen[1].bid_qty == 1.0


def test_feed_synthetic_book_bid_ask():
    feed = BinanceTickFeed(["BTCUSDT"], window=5)
    seen = []
    feed.on_tick(seen.append)
    for i, px in enumerate(["100", "102", "101", "103"]):
        feed._on_message(_agg(price=px, qty="1", maker=False, E=1788956000000 + i))
    last = seen[-1]
    assert last.bid == 100.0  # min of window
    assert last.ask == 103.0  # max of window
    assert last.ltp == 103.0


def test_feed_replay():
    import asyncio
    import tempfile

    ticks = [Tick(timestamp=float(i), symbol="BTCUSDT", ltp=100.0, bid=99.9, ask=100.1,
                  bid_qty=1.0, ask_qty=1.0, volume=0.5, oi=1.0) for i in range(5)]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for t in ticks:
            f.write(t.model_dump_json() + "\n")
        path = f.name

    from quant_crypto.data.binance_feed import ReplayFeed

    feed = ReplayFeed(path, max_ticks=3)
    seen = []
    feed.on_tick(seen.append)

    async def run():
        await feed.start()

    asyncio.run(run())
    assert len(seen) == 3


def test_parse_ts_variants():
    from quant_crypto.data.tape import parse_ts

    assert parse_ts(1000) == 1000.0
    assert isinstance(parse_ts("2026-09-09T00:00:00Z"), float)
