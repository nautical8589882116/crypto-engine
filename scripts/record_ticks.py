"""Live tick recorder for Binance spot (BTC/ETH).

    uv run python scripts/record_ticks.py --out /data/tapes/2026-09-09.jsonl
    uv run python scripts/record_ticks.py --out out.jsonl --replay in.jsonl [--max-ticks 1000]

Uses the bounded ingest spine (drop-oldest, stale guard, batched writes).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from quant_crypto.data.binance_feed import BinanceTickFeed, ReplayFeed
from quant_crypto.data.ingest import run_ingest
from quant_crypto.data.tape import TapeWriter


def main() -> None:
    p = argparse.ArgumentParser(description="Record live Binance ticks to a JSONL tape")
    p.add_argument("--out", required=True, help="output tape path (.jsonl)")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT", help="comma list of spot symbols")
    p.add_argument("--replay", default=None, help="replay mode: source tape")
    p.add_argument("--max-ticks", type=int, default=None, help="stop after N ticks")
    p.add_argument("--ingest-buffer", type=int, default=2000, help="ingest queue capacity")
    args = p.parse_args()

    out = Path(args.out)
    if args.replay:
        feed = ReplayFeed(args.replay, max_ticks=args.max_ticks)
    else:
        feed = BinanceTickFeed([s.strip().upper() for s in args.symbols.split(",")])
    writer = TapeWriter(out, buffer_size=args.ingest_buffer)

    async def _run():
        try:
            report = await run_ingest(
                feed, on_tick=writer.write, max_ticks=args.max_ticks, register_active=True
            )
        finally:
            writer.close()
        print(f"received: {report.received} | processed: {report.processed} -> {out}")
        print(f"dropped: overflow={report.dropped_overflow} stale={report.dropped_stale} "
              f"out_of_order={report.out_of_order}")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\nrecording stopped by operator")


if __name__ == "__main__":
    main()
