"""Corpus builder — turn recorded crypto tick tapes into Mamba training windows.

Windows reuse the same ring-buffer layout and `window.raw_channels` as live
inference, so training and runtime inputs stay consistent. Labels are realized
forward price-velocity over `horizon` ticks; windows crossing feed gaps are
dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quant_crypto.data.binance_feed import Tick
from quant_crypto.data.ring_buffer import RingBuffer
from quant_crypto.data.tape import parse_ts
from quant_crypto.data.window import raw_channels, zscore_rows


@dataclass
class CorpusParams:
    seq_len: int = 300
    horizon: int = 10
    threshold: float = 0.0
    stride: int = 50
    gap_max_s: float = 5.0
    seed: int = 0


def _forward_fill_series(snap_ts, snap_vals, tick_ts):
    """Step forward-fill chain values onto tick timestamps (unused here, kept
    for symmetry — crypto tapes are self-contained, so no parallel series)."""
    return None


def build_windows(
    ticks: list[Tick],
    params: CorpusParams | None = None,
    chain=None,  # accepted for API parity with the parent engine; unused
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (X, y, meta) float32/int8 arrays of (seq_len,3) windows."""
    p = params or CorpusParams()
    n = len(ticks)
    seq = p.seq_len
    if n < seq + p.horizon:
        raise ValueError(f"need >= {seq + p.horizon} ticks, got {n}")

    ts = np.array([parse_ts(t.timestamp) for t in ticks], dtype=np.float64)
    ltp = np.array([t.ltp for t in ticks], dtype=np.float64)

    # Build (n,3) raw channels from the ring-buffer layout.
    to_vec = RingBuffer._tick_to_vector
    buf = np.array([to_vec(t) for t in ticks], dtype=np.float64)
    ch = raw_channels(buf)

    # Label: mean of forward horizon LTP minus last LTP > threshold.
    ys: list[int] = []
    xs: list[np.ndarray] = []
    start = 0
    while start + seq + p.horizon <= n:
        end = start + seq
        # drop windows crossing a feed gap
        span = ts[end - 1] - ts[start]
        if p.gap_max_s > 0 and span > p.gap_max_s * (end - start):
            start += p.stride
            continue
        win = ch[start:end]
        if win.shape[0] < 2:
            start += p.stride
            continue
        z = zscore_rows(win)
        future = ltp[end : end + p.horizon]
        label = 1 if float(future.mean() - ltp[end - 1]) > p.threshold else 0
        xs.append(z.astype(np.float32))
        ys.append(label)
        start += p.stride

    if not xs:
        raise ValueError("no windows built (gaps consumed everything?)")

    X = np.stack(xs)
    y = np.array(ys, dtype=np.int8)
    meta = {
        "n_windows": int(X.shape[0]),
        "seq_len": seq,
        "n_features": int(X.shape[2]),
        "horizon": p.horizon,
        "threshold": p.threshold,
        "n_pos": int(y.sum()),
        "ticks": n,
    }
    return X, y, meta
