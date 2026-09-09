"""Shared tick-window channels (Stage C1).

The model consumes (seq_len, 3) windows of:
  c0 = per-tick order-flow imbalance (Δbid_qty − Δask_qty, normalized)
  c1 = per-tick price change
  c2 = per-tick OI change

built from ring-buffer snapshots (8-column layout, see RingBuffer) at runtime,
and from recorded tapes in the corpus builder. Both paths share this module so
training inputs and live inference inputs stay consistent.
"""

from __future__ import annotations

import numpy as np

# RingBuffer snapshot column indices
COL_LTP = 0
COL_BID_QTY = 3
COL_ASK_QTY = 4
COL_OI = 6


def raw_channels(snap: np.ndarray) -> np.ndarray:
    """(n, 8) ring-buffer snapshot -> (n, 3) raw per-tick channels."""
    if snap.ndim != 2 or snap.shape[1] != 8:
        raise ValueError(f"expected (n, 8) snapshot, got {snap.shape}")
    n = snap.shape[0]
    ch = np.zeros((n, 3), dtype=np.float64)
    if n < 2:
        return ch
    db = np.diff(snap[:, COL_BID_QTY])
    da = np.diff(snap[:, COL_ASK_QTY])
    denom = np.abs(db) + np.abs(da)
    ofi = np.zeros(n)
    np.divide(db - da, denom, out=ofi[1:], where=denom > 0)
    ch[:, 0] = ofi
    ch[1:, 1] = np.diff(snap[:, COL_LTP])
    ch[1:, 2] = np.diff(snap[:, COL_OI])
    return ch


def zscore_rows(ch: np.ndarray) -> np.ndarray:
    """Column-wise z-score with a zero-std guard (never NaN)."""
    mean = ch.mean(axis=0)
    std = ch.std(axis=0)
    std[std < 1e-12] = 1.0
    return (ch - mean) / std


def window_from_snapshot(snap: np.ndarray) -> np.ndarray:
    """Latest ring-buffer snapshot -> (seq, 3) float32 model window.

    Standardizes per window (no future data needed) so the live path can feed
    MambaInference the same channel semantics the corpus builder produced.
    """
    return zscore_rows(raw_channels(snap)).astype(np.float32)
