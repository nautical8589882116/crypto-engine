"""Unit tests: dataset window building."""

import numpy as np
import pytest

from quant_crypto.data.binance_feed import Tick
from quant_crypto.data.dataset import CorpusParams, build_windows


def _mk(n, uptrend=True):
    ticks = []
    for i in range(n):
        ltp = 100.0 + (i * 0.5 if uptrend else -i * 0.5)
        ticks.append(Tick(timestamp=float(i), symbol="BTCUSDT", ltp=ltp,
                          bid=ltp - 0.1, ask=ltp + 0.1, bid_qty=1.0, ask_qty=1.0,
                          volume=0.5, oi=float(i)))
    return ticks


def test_build_windows_shapes_and_dtype():
    X, y, meta = build_windows(_mk(600), CorpusParams(seq_len=200, horizon=5, stride=100))
    assert X.ndim == 3
    assert X.shape[1:] == (200, 3)
    assert X.dtype == np.float32
    assert y.dtype == np.int8
    assert X.shape[0] == y.shape[0]
    assert meta["seq_len"] == 200
    assert meta["n_features"] == 3


def test_build_windows_labels_uptrend():
    X, y, meta = build_windows(_mk(600, uptrend=True), CorpusParams(seq_len=200, horizon=5, stride=100, threshold=0.0))
    assert int(y.sum()) == y.shape[0]


def test_build_windows_labels_downtrend():
    X, y, meta = build_windows(_mk(600, uptrend=False), CorpusParams(seq_len=200, horizon=5, stride=100, threshold=0.0))
    assert int(y.sum()) == 0


def test_build_windows_too_short_raises():
    with pytest.raises(ValueError):
        build_windows(_mk(50), CorpusParams(seq_len=200, horizon=5))
