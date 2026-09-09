"""Fixed-capacity ring buffer for the rolling tick window.

The predictive model consumes a rolling window of the last N ticks. This is a
numpy-backed circular buffer with an O(1) append and O(1) snapshot, safe for
single-producer (feed) single-consumer (feature pipeline) use.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from quant_crypto.data.binance_feed import Tick


class RingBuffer:
    """Circular buffer holding the most recent `capacity` ticks."""

    def __init__(self, capacity: int = 300, n_features: int = 8):
        self.capacity = capacity
        self.n_features = n_features
        self._buf = np.zeros((capacity, n_features), dtype=np.float64)
        self._timestamps: list[str] = [""] * capacity
        self._head = 0  # index of next write
        self._size = 0
        self._lock = threading.Lock()

    def append(self, tick: Tick) -> None:
        vector = self._tick_to_vector(tick)
        with self._lock:
            self._buf[self._head] = vector
            self._timestamps[self._head] = tick.timestamp
            self._head = (self._head + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    @staticmethod
    def _tick_to_vector(tick: Tick) -> np.ndarray:
        return np.array(
            [
                tick.ltp,
                tick.bid,
                tick.ask,
                tick.bid_qty,
                tick.ask_qty,
                tick.volume,
                tick.oi,
                tick.ltp - (tick.bid + tick.ask) / 2,  # microprice deviation
            ],
            dtype=np.float64,
        )

    def snapshot(self) -> np.ndarray:
        """Return ticks in chronological order as an (n, features) matrix."""
        with self._lock:
            if self._size == 0:
                return np.empty((0, self.n_features), dtype=np.float64)
            if self._size < self.capacity:
                return self._buf[: self._size].copy()
            # Buffer wrapped; reconstruct chronological order.
            out = np.empty_like(self._buf)
            out[: self.capacity - self._head] = self._buf[self._head :]
            out[self.capacity - self._head :] = self._buf[: self._head]
            return out

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity
