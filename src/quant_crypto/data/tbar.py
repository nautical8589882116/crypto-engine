"""Time-bar feature extraction for live inference (60 bars x 7 features).

The released model was trained on fixed 1-minute bars (open/high/low/close,
volume, signed OFI, return) — NOT on variable-length tick windows. This module
aggregates incoming ticks into 1-minute bars per symbol and maintains a rolling
window of the last NBAR bars, so live inference feeds the model the same
semantics the corpus builder produced.

Bar features (7): open, high, low, close, volume, ofi (signed $ flow), return.
Each window is z-scored per column (no future data) before inference.
"""
from __future__ import annotations

import time

import numpy as np

NBAR = 60          # bars of lookback (matches the trained model)
BAR_SECONDS = 60   # 1-minute bars


class TimeBarWindow:
    """Aggregate ticks into 1-min bars and expose a rolling (NBAR, 7) window.

    `push(tick)` buckets by wall-clock minute; when a new minute starts the
    previous bar is finalized and appended. `window()` returns the last NBAR
    bars as a z-scored (NBAR, 7) float32 array, or None until NBAR bars exist.
    """

    def __init__(self, nbar: int = NBAR, bar_seconds: int = BAR_SECONDS):
        self.nbar = nbar
        self.bar_seconds = bar_seconds
        self._bars: list[np.ndarray] = []   # finalized (7,) rows
        self._cur: dict | None = None       # in-progress bar
        self._cur_min: int | None = None

    def _finalize(self) -> np.ndarray:
        c = self._cur
        o, h, l, cl = c["o"], c["h"], c["l"], c["c"]
        vol = c["vol"]; ofi = c["ofi"]
        ret = cl - c["prev_close"] if c["prev_close"] is not None else 0.0
        return np.array([o, h, l, cl, vol, ofi, ret], dtype=np.float64)

    def push(self, tick) -> None:
        ts = getattr(tick, "timestamp", None)
        try:
            t = float(ts) if ts is not None else time.time()
        except (TypeError, ValueError):
            t = time.time()
        minute = int(t // self.bar_seconds)
        ltp = float(getattr(tick, "ltp", 0.0) or 0.0)
        if ltp <= 0:
            return
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        qty = float(getattr(tick, "volume", 0.0) or 0.0)
        # signed flow: Coinbase ticker has no per-trade side; approximate OFI
        # from price direction vs mid as a weak signed-volume proxy.
        if bid and ask:
            signed = qty if ltp >= (bid + ask) / 2.0 else -qty
        else:
            signed = qty

        if self._cur is None or minute != self._cur_min:
            if self._cur is not None:
                self._bars.append(self._finalize())
                if len(self._bars) > self.nbar:
                    self._bars = self._bars[-self.nbar:]
            prev_close = self._bars[-1][3] if self._bars else None
            self._cur = {"o": ltp, "h": ltp, "l": ltp, "c": ltp,
                         "vol": 0.0, "ofi": 0.0, "prev_close": prev_close}
            self._cur_min = minute
        c = self._cur
        c["h"] = max(c["h"], ltp); c["l"] = min(c["l"], ltp); c["c"] = ltp
        c["vol"] += qty
        c["ofi"] += signed

    def window(self) -> np.ndarray | None:
        """Return z-scored (NBAR, 7) float32 window, or None until NBAR bars."""
        if len(self._bars) < self.nbar:
            return None
        arr = np.asarray(self._bars[-self.nbar:], dtype=np.float64)
        mu = arr.mean(axis=0); sd = arr.std(axis=0)
        sd[sd < 1e-12] = 1.0
        return ((arr - mu) / sd).astype(np.float32)

    @property
    def ready(self) -> bool:
        return len(self._bars) >= self.nbar
