"""Tick capture helper with feed-silence watchdog (CLI recorder uses this)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from quant_crypto.data.binance_feed import Tick
from quant_crypto.data.ingest import feed_active
from quant_crypto.data.tape import TapeWriter


@dataclass
class CaptureStats:
    ticks_written: int = 0
    gaps: list[tuple[float, float]] = field(default_factory=list)


async def capture(
    feed: Any,
    writer: TapeWriter,
    *,
    gap_threshold_s: float = 10.0,
    watch_interval_s: float = 1.0,
    max_ticks: int | None = None,
) -> CaptureStats:
    """Record feed ticks to a writer; watchdog reports feed-silence gaps."""
    stats = CaptureStats()
    last_ts: float | None = None
    written = 0

    def on_tick(tick: Tick) -> None:
        nonlocal last_ts, written
        ts = tick.timestamp or time.time()
        if last_ts is not None and ts - last_ts > gap_threshold_s:
            stats.gaps.append((last_ts, ts))
        last_ts = ts
        writer.write(tick)
        written += 1

    async def watchdog() -> None:
        while True:
            await asyncio.sleep(watch_interval_s)
            writer.flush()

    feed.on_tick(on_tick)
    wd = asyncio.create_task(watchdog())
    feed_task = asyncio.create_task(feed.start())

    while True:
        if max_ticks is not None and written >= max_ticks:
            break
        if not feed_active(feed):
            break
        if feed_task.done():
            break
        await asyncio.sleep(0.05)

    await feed.stop()
    try:
        await asyncio.wait_for(feed_task, timeout=3)
    except Exception:  # noqa: BLE001
        pass
    wd.cancel()
    writer.flush()
    stats.ticks_written = written
    return stats
