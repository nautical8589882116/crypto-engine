"""Local paper runner — replay a crypto tape through the real loop.

    uv run python scripts/run_engine.py <tape.jsonl> <model.onnx> [seq_len]

Replays recorded Binance ticks through feed -> ring buffers -> Mamba inference
-> execution kernel and prints a session summary.
"""

from __future__ import annotations

import asyncio
import sys

from quant_crypto.config import get_settings
from quant_crypto.data.binance_feed import ReplayFeed
from quant_crypto.engine.paper import PaperEngine
from quant_crypto.model.infer import MambaInference


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: run_engine.py <tape.jsonl> <model.onnx> [seq_len]\n"
            "The real loop scores a model over the tape; no fabricated signals."
        )
    tape, model_path = sys.argv[1], sys.argv[2]
    seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    settings = get_settings()
    engine = PaperEngine(settings, model=MambaInference(model_path, seq_len=seq_len))
    result = asyncio.run(engine.run_realtime(ReplayFeed(tape)))
    print(f"Ticks processed:    {result.ticks_processed}")
    print(f"Trades entered:     {result.entered_trades}")
    print(f"Total PnL:          {result.total_pnl:+.4f}")
    for t in result.trades:
        print(f"  {t.symbol} {t.direction} in@{t.entry_price:.2f} out@{t.exit_price:.2f} "
              f"({t.exit_reason}) pnl={t.pnl:+.4f}")


if __name__ == "__main__":
    main()
