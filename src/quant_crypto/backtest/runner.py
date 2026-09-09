"""Backtest runner — replay a crypto tape through the model; OOS go/no-go gate.

Mirrors the parent engine's `backtest/runner.py` but trades the crypto 3-channel
window (OFI / Δprice / Δvolume) directly, no options chain. PnL is reported in
base-currency price points, and in USDT notional when lot_units/slippage/fees
are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from quant_crypto.data.binance_feed import Tick
from quant_crypto.data.ring_buffer import RingBuffer
from quant_crypto.data.tape import parse_ts
from quant_crypto.data.window import raw_channels

Predictor = Callable[[np.ndarray], float]


@dataclass
class BacktestParams:
    seq_len: int = 300
    horizon: int = 10
    gate: float = 0.85
    stride: int = 50
    slippage_bps: float = 1.0
    lot_units: float = 1.0  # base units per "lot"
    fee_rate: float = 0.001  # spot taker fee each side


@dataclass
class BacktestReport:
    n_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    pnl_usdt: float = 0.0
    total_costs_usdt: float = 0.0
    max_drawdown: float = 0.0
    trades: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "total_pnl_points": round(self.total_pnl, 4),
            "pnl_usdt": round(self.pnl_usdt, 4),
            "total_costs_usdt": round(self.total_costs_usdt, 4),
            "max_drawdown_points": round(self.max_drawdown, 4),
        }


def _cumulative_z(ch: np.ndarray) -> np.ndarray:
    """Causal (prefix) standardization — no future leakage."""
    out = np.zeros_like(ch)
    for i in range(ch.shape[0]):
        if i == 0:
            out[i] = 0.0
            continue
        mean = ch[: i + 1].mean(axis=0)
        std = ch[: i + 1].std(axis=0)
        std[std < 1e-12] = 1.0
        out[i] = (ch[i] - mean) / std
    return out


def backtest(ticks: list[Tick], predict: Predictor, params: BacktestParams | None = None) -> BacktestReport:
    p = params or BacktestParams()
    report = BacktestReport()
    n = len(ticks)
    if n < p.seq_len + p.horizon:
        return report

    to_vec = RingBuffer._tick_to_vector
    buf = np.array([to_vec(t) for t in ticks], dtype=np.float64)
    ch = raw_channels(buf)
    xz = _cumulative_z(ch)

    entry_prices = np.array([t.ask or t.ltp for t in ticks])
    exit_prices = np.array([t.bid or t.ltp for t in ticks])

    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    start = 0
    while start + p.seq_len + p.horizon <= n:
        end = start + p.seq_len
        window = xz[start:end].astype(np.float32)
        prob = float(predict(window))
        if prob < p.gate:
            start += p.stride
            continue
        anchor = end - 1
        entry = float(entry_prices[anchor])
        exit_p = float(exit_prices[anchor + p.horizon - 1])
        pnl = exit_p - entry  # long-only
        cost_points = entry * p.slippage_bps * 1e-4 * 2.0
        cost_usdt = (entry * p.lot_units) * p.fee_rate * 2.0
        report.total_costs_usdt += cost_usdt
        net_points = pnl - cost_points
        report.n_trades += 1
        if net_points > 0:
            report.wins += 1
        else:
            report.losses += 1
        report.total_pnl += net_points
        report.pnl_usdt += (pnl * p.lot_units) - cost_usdt
        cum += net_points * p.lot_units
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
        report.trades.append(
            {"idx": anchor, "entry": round(entry, 2), "exit": round(exit_p, 2), "pnl": round(net_points, 4)}
        )
        start += p.stride

    report.max_drawdown = abs(max_dd)
    if report.n_trades:
        report.win_rate = report.wins / report.n_trades
    return report
