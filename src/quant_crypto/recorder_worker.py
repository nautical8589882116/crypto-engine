"""Recorder worker — runs the live Binance feed in its own process.

On Python 3.13 the feed's asyncio event loop is starved (0 ticks) whenever the
process also runs another live thread (the HTTP server). Running the recorder
as a separate subprocess gives the feed a whole process/event loop.

Writes ticks to a JSONL tape AND to a `latest.jsonl` shared file that the
control plane (entrypoint) tails for the live market-state panel.

Run with: python -m quant_crypto.recorder_worker
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from quant_crypto.config import get_settings
from quant_crypto.data.binance_feed import BinanceTickFeed
from quant_crypto.data.ingest import run_ingest
from quant_crypto.data.tape import TapeWriter


def _latest_path(out_dir: Path) -> Path:
    return out_dir / "latest.jsonl"


def main() -> None:
    settings = get_settings()
    out_dir = Path(os.environ.get("RECORD_DIR", "/data/tapes"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"{tag}_TICKS.jsonl"
    symbols = [s.upper() for s in settings.crypto_symbols]

    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s recorder %(message)s")
    log = logging.getLogger("recorder_worker")
    log.info("recorder worker start: %s -> %s", symbols, out)

    feed = BinanceTickFeed(symbols)
    writer = TapeWriter(out, buffer_size=2000)
    latest = TapeWriter(_latest_path(out_dir), buffer_size=0)  # flush every tick

    def on_tick(tick):
        latest.write(tick)  # cross-process: server tails this file
        # rotate latest file so it doesn't grow unbounded (keep last ~500 lines)
        try:
            p = _latest_path(out_dir)
            if p.stat().st_size > 2_000_000:
                p.unlink()
                latest2 = TapeWriter(p, buffer_size=0)
                latest2.write(tick)
        except OSError:
            pass

    async def loop():
        try:
            await run_ingest(feed, on_tick=lambda t: (writer.write(t), on_tick(t)), register_active=True)
        finally:
            writer.close()
            latest.close()

    asyncio.run(loop())


if __name__ == "__main__":
    main()
