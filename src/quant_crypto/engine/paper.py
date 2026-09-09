"""Paper realtime loop — feed -> per-symbol ring buffers -> Mamba -> kernel.

Mirrors the parent engine's PaperEngine but handles MULTIPLE spot symbols
(BTC/ETH) with independent windows, and exits on REAL subsequent prices via
take-profit/stop levels instead of a fabricated market path.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from quant_crypto.broker.base import PaperBroker
from quant_crypto.config import get_settings
from quant_crypto.data.binance_feed import Tick
from quant_crypto.data.ring_buffer import RingBuffer
from quant_crypto.data.window import window_from_snapshot
from quant_crypto.engine.telemetry import publish_engine
from quant_crypto.exec.kernel import ExecutionKernel
from quant_crypto.schemas.order import Signal


@dataclass
class LiveTradeRecord:
    symbol: str
    direction: str
    prob: float
    entry_price: float
    exit_price: float
    exit_reason: str
    pnl: float


@dataclass
class LiveRunResult:
    ticks_processed: int = 0
    inferences: int = 0
    signals_generated: int = 0
    entered_trades: int = 0
    trades: list[LiveTradeRecord] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)


class PaperEngine:
    def __init__(
        self,
        settings=None,
        broker=None,
        model=None,
        *,
        model_stride: int | None = None,
        take_profit_pct: float = 0.005,
        stop_loss_pct: float = 0.003,
        publish_telemetry: bool = True,
    ):
        self.settings = settings or get_settings()
        self.broker = broker or PaperBroker.from_settings(self.settings)
        self.model = model
        self.model_stride = model_stride or self.settings.model_stride
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.publish_telemetry = publish_telemetry
        self.strategy = None  # optional LLM StrategyConfig applied per entry
        self.kernel = ExecutionKernel(self.broker, self.settings)
        # per-symbol state
        self._buffers: dict[str, RingBuffer] = {}
        self._seq = getattr(model, "seq_len", 300) if model else 300
        self._tick_counts: dict[str, int] = {}
        self._open: dict[str, dict] = {}  # symbol -> {entry, tp, sl}

    def _buffer_for(self, symbol: str) -> RingBuffer:
        if symbol not in self._buffers:
            self._buffers[symbol] = RingBuffer(capacity=self._seq)
            self._tick_counts[symbol] = 0
        return self._buffers[symbol]

    def _emit_signal(self, symbol: str, prob: float) -> Signal:
        thr = self.settings.prob_threshold
        direction = "long" if prob >= thr else "flat"
        return Signal(
            symbol=symbol,
            regime="trend_follow",
            direction=direction,
            prob_velocity=prob,
            threshold=thr,
            confidence=1.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _check_exits(self, tick: Tick) -> None:
        state = self._open.get(tick.symbol)
        if not state:
            return
        ltp = tick.ltp
        reason = None
        if ltp >= state["tp"]:
            reason = "take_profit"
        elif ltp <= state["sl"]:
            reason = "stop_loss"
        if reason:
            result = self.kernel.exit_position(tick.symbol, ltp, reason)
            if result.get("closed"):
                self._open.pop(tick.symbol, None)
                self._trades.append(
                    LiveTradeRecord(
                        symbol=tick.symbol,
                        direction="long",
                        prob=state.get("prob", 0.0),
                        entry_price=state["entry"],
                        exit_price=result.get("fill_price", ltp),
                        exit_reason=reason,
                        pnl=result.get("pnl", 0.0),
                    )
                )

    def _enter(self, symbol: str, prob: float, ltp: float, strategy=None) -> None:
        if symbol in self._open:
            return  # already holding
        sig = self._emit_signal(symbol, prob)
        if not sig.is_tradeable:
            return
        outcome = self.kernel.handle_signal(sig, entry_price=ltp)
        if not outcome.entered:
            return
        entry = outcome.entry_price
        # Per-trade TP/SL: strategy config (LLM Tier 1) overrides engine defaults.
        tp = self.take_profit_pct
        sl = self.stop_loss_pct
        if strategy is not None:
            if getattr(strategy, "take_profit_pct", None):
                tp = strategy.take_profit_pct
            if getattr(strategy, "stop_loss_pct", None):
                sl = strategy.stop_loss_pct
        self._open[symbol] = {
            "entry": entry,
            "tp": entry * (1 + tp),
            "sl": entry * (1 - sl),
            "prob": prob,
        }

    async def run_realtime(self, feed, *, max_ticks: int | None = None) -> LiveRunResult:
        self._trades: list[LiveTradeRecord] = []

        def on_tick(tick: Tick) -> None:
            buf = self._buffer_for(tick.symbol)
            buf.append(tick)
            self._tick_counts[tick.symbol] += 1
            # Keep the broker's book in sync so FOK entries + IOC exits can fill.
            if hasattr(self.broker, "set_quote"):
                self.broker.set_quote(
                    tick.symbol, bid=tick.bid, ask=tick.ask,
                    bid_qty=tick.bid_qty, ask_qty=tick.ask_qty,
                )
            self._check_exits(tick)
            count = self._tick_counts[tick.symbol]
            if self.model is not None and buf.is_full and (count % self.model_stride == 0):
                window = window_from_snapshot(buf.snapshot())
                prob = float(self.model.predict(window))
                self._enter(tick.symbol, prob, tick.ltp, strategy=self.strategy)

        feed.on_tick(on_tick)
        feed_task = asyncio.create_task(feed.start())
        processed = 0
        while True:
            await asyncio.sleep(0.01)
            processed = sum(self._tick_counts.values())
            if max_ticks is not None and processed >= max_ticks:
                break
            if not getattr(feed, "running", True) and processed > 0:
                break
            if feed_task.done():
                break
        await feed.stop()
        await feed_task

        result = LiveRunResult(
            ticks_processed=processed,
            inferences=processed,  # coarse; exact count not critical for dashboard
            signals_generated=sum(1 for t in self._trades),
            entered_trades=len(self._trades),
            trades=self._trades,
        )
        if self.publish_telemetry:
            publish_engine(
                {
                    "ticks_processed": result.ticks_processed,
                    "trades": len(result.trades),
                    "total_pnl": result.total_pnl,
                    "mode": "paper",
                }
            )
        return result
