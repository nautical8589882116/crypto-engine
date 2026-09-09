"""Backtest CLI — the go/no-go gate.

    uv run python scripts/backtest.py --tape /data/tapes/2026-09-09.jsonl \
        --model /data/models/<accepted>.onnx [--seq 300 --gate 0.85]

Prints n_trades / win_rate / total_pnl / max_drawdown. Requires the model ONNX.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_crypto.backtest.runner import BacktestParams, backtest
from quant_crypto.data.binance_feed import Tick
from quant_crypto.model.infer import MambaInference


def _load_tape(path: Path) -> list[Tick]:
    ticks = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                ticks.append(Tick.model_validate_json(line))
    return ticks


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest an ONNX model over a crypto tape")
    p.add_argument("--tape", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seq", type=int, default=300)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--gate", type=float, default=0.85)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--lot-units", type=float, default=0.001)
    args = p.parse_args()

    ticks = _load_tape(Path(args.tape))
    print(f"tape: {len(ticks)} ticks")
    params = BacktestParams(
        seq_len=args.seq, horizon=args.horizon, gate=args.gate, stride=args.stride,
        slippage_bps=args.slippage_bps, lot_units=args.lot_units,
    )
    infer = MambaInference(args.model, seq_len=args.seq, n_features=3)
    report = backtest(ticks, infer.predict, params)
    print("== report ==")
    for k, v in report.as_dict().items():
        print(f"  {k}: {v}")
    if report.n_trades == 0:
        print("no trades at gate", args.gate)


if __name__ == "__main__":
    main()
