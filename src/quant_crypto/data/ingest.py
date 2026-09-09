"""L2 firehose ingest spine — bounded queue, drop-oldest, zero-silent-loss accounting.

Fully asyncio (the Binance feed is async): the feed drives callbacks that put
into a bounded deque; a consumer drains it. Every tick is accounted as
processed / overflow / stale / out-of-order — never silently lost.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass

from quant_crypto.data.binance_feed import Tick


@dataclass
class IngestReport:
    received: int = 0
    processed: int = 0
    dropped_overflow: int = 0
    dropped_stale: int = 0
    out_of_order: int = 0
    queue_depth: int = 0
    elapsed_s: float = 0.0

    @property
    def accounted(self) -> int:
        return self.processed + self.dropped_overflow + self.dropped_stale + self.out_of_order


class TickIngestQueue:
    """Bounded single-reader tick queue; drops OLDEST on overflow."""

    def __init__(self, maxlen: int = 2000):
        self._q: deque = deque(maxlen=maxlen)
        self._dropped = 0
        self._lock = threading.Lock()

    def put(self, tick: Tick) -> None:
        with self._lock:
            if len(self._q) == self._q.maxlen:
                self._dropped += 1
            self._q.append(tick)

    async def get(self) -> Tick | None:
        while True:
            with self._lock:
                if self._q:
                    return self._q.popleft()
            await asyncio.sleep(0.001)

    @property
    def dropped(self) -> int:
        return self._dropped

    def __len__(self) -> int:
        return len(self._q)


def is_stale_tick(tick: Tick, last_ts: float, max_age_s: float = 60.0) -> str | None:
    if last_ts and tick.timestamp < last_ts:
        return "out_of_order"
    if last_ts and tick.timestamp - last_ts > max_age_s:
        return "stale"
    return None


_ACTIVE: dict = {"ingest": None}


def set_active_ingest(report: IngestReport) -> None:
    _ACTIVE["ingest"] = report


def ingest_snapshot() -> dict:
    r = _ACTIVE["ingest"]
    if r is None:
        return {"received": 0, "processed": 0, "dropped_overflow": 0,
                "dropped_stale": 0, "out_of_order": 0, "queue_depth": 0, "elapsed_s": 0.0}
    return {"received": r.received, "processed": r.processed,
            "dropped_overflow": r.dropped_overflow, "dropped_stale": r.dropped_stale,
            "out_of_order": r.out_of_order, "queue_depth": r.queue_depth,
            "elapsed_s": round(r.elapsed_s, 2)}


def feed_active(feed) -> bool:
    return bool(getattr(feed, "running", False))


async def run_ingest(
    feed,
    *,
    on_tick=None,
    maxlen: int = 2000,
    max_ticks: int | None = None,
    duration_s: float | None = None,
    stale_max_age_s: float = 60.0,
    register_active: bool = False,
) -> IngestReport:
    q = TickIngestQueue(maxlen=maxlen)
    report = IngestReport()
    if register_active:
        set_active_ingest(report)

    t0 = time.time()
    stop = threading.Event()

    def cb(tick: Tick) -> None:
        if stop.is_set():
            return
        report.received += 1
        q.put(tick)

    feed.on_tick(cb)
    feed_task = asyncio.create_task(feed.start())

    async def deliver() -> None:
        last_ts: float = 0.0
        while not stop.is_set():
            if max_ticks is not None and report.processed >= max_ticks:
                break
            try:
                tick = await asyncio.wait_for(q.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            if tick is None:
                continue
            label = is_stale_tick(tick, last_ts, stale_max_age_s)
            if label == "out_of_order":
                report.out_of_order += 1
                continue
            if label == "stale":
                report.dropped_stale += 1
                continue
            last_ts = tick.timestamp
            report.processed += 1
            report.queue_depth = len(q)
            if on_tick is not None:
                if asyncio.iscoroutinefunction(on_tick):
                    await on_tick(tick)
                else:
                    on_tick(tick)
        # Drain anything left in the queue after termination.
        while q._q:
            try:
                tick = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                break
            if tick is not None:
                report.processed += 1
                if on_tick is not None:
                    if asyncio.iscoroutinefunction(on_tick):
                        await on_tick(tick)
                    else:
                        on_tick(tick)
        report.queue_depth = len(q)

    deliver_task = asyncio.create_task(deliver())
    if duration_s is not None:
        await asyncio.sleep(duration_s)
    else:
        # Indefinite mode: wait until the feed stops itself or is externally
        # stopped (e.g. max_ticks reached / process shutdown). Do NOT set stop
        # immediately — that would kill delivery before any tick is drained.
        # Wait on the feed task; the deliver coroutine exits on its own when
        # max_ticks is reached (deliver checks it) or the feed ends.
        try:
            await asyncio.wait_for(feed_task, timeout=None)
        except asyncio.CancelledError:
            pass
    stop.set()
    await deliver_task

    await feed.stop()
    try:
        await asyncio.wait_for(feed_task, timeout=3)
    except (asyncio.TimeoutError, Exception):  # noqa: BLE001
        pass
    stop.set()
    report.elapsed_s = time.time() - t0
    report.dropped_overflow = q.dropped
    if register_active:
        set_active_ingest(report)
    return report
