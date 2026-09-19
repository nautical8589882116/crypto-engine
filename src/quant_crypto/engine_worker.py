"""Engine worker — live paper inference over the recorded tape, in its own process.

The recorder owns the websocket; this worker tails the tape it writes, runs the
Mamba ONNX model over rolling windows, simulates paper entries/exits, and
publishes telemetry to `telemetry.json` (cross-process, like latest.jsonl) so
the control plane's dashboard can render inference, trades and PnL.

Safety surface (all mirrored across processes so the worker can never be left
unguarded by a control-plane that runs elsewhere):
- Position cap: no new entry may push a symbol's gross position beyond
  `max_position` (an existing open position already consumes part of the cap).
- Daily-loss breaker: once realized `total_pnl <= -max_daily_loss` the worker
  halts new entries and surfaces `halted` in telemetry.
- Cross-process kill switch: the flag file `killswitch.flag` (in RECORD_DIR) is
  checked each loop iteration and before each entry; `/api/v1/kill` creates it
  and `/api/v1/resume` removes it, so the dashboard kill button stops this
  worker even though it runs in a separate process. It is NOT reset by this
  process, so it survives worker restarts until the operator resumes.

Run with: ENGINE_MODEL=/data/models/x.onnx python -m quant_crypto.engine_worker
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from quant_crypto.config import get_settings
from quant_crypto.data.binance_feed import Tick
from quant_crypto.data.ring_buffer import RingBuffer
from quant_crypto.data.window import window_from_snapshot

# Cross-process kill-switch flag file name (shared with entrypoint.py).
KILL_FLAG_NAME = "killswitch.flag"


def _get_logger():
    import logging

    lg = logging.getLogger("quant_crypto.engine")
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s engine %(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg


def _find_model() -> str | None:
    env = os.environ.get("ENGINE_MODEL")
    if env and Path(env).is_file():
        return env
    rec = Path(os.environ.get("RECORD_DIR", "/data/tapes"))
    for d in (rec.parent / "models", Path("/data/models"), Path("models")):
        if d.is_dir():
            for f in sorted(d.glob("*.onnx")):
                return str(f)
    return None


def _kill_flag(rec_dir: Path) -> Path:
    """Path of the cross-process kill-switch flag (shared with entrypoint)."""
    return rec_dir / KILL_FLAG_NAME


class EngineWorkerCore:
    """Inference + paper-sim core, kept separate from the tape loop for testing.

    Holds the open book, telemetry and the three safety guards (position cap,
    daily-loss breaker, cross-process kill-switch flag). `process(tick)` is the
    same tailing entry used by the production loop; tests drive it directly with
    a fake model and injected settings.
    """

    def __init__(self, model, settings, rec: Path):
        self.model = model
        self.settings = settings
        self.rec = Path(rec).absolute()
        self.killflag = _kill_flag(self.rec)

        self.seq = model.seq_len
        self.stride = settings.model_stride
        self.buffers: dict[str, RingBuffer] = {}
        self.counts: dict[str, int] = {}
        self.open_pos: dict[str, dict] = {}

        self.tele = {
            "ticks_processed": 0, "inferences": 0, "trades": 0, "wins": 0,
            "losses": 0, "total_pnl": 0.0, "last_prob": None,
            "entries": 0, "open_positions": 0,
            "threshold": settings.prob_threshold, "session_source": "coinbase-live",
            "mode": "paper", "updated_at": time.time(),
            "unrealized_pnl": 0.0,
            # safety surface
            "blocked": 0, "halted": False, "killswitch": False,
            # execution-slippage (bps = adverse fill cost vs mid; USD = $ cost)
            "slippage_series": [], "slippage_bps_total": 0.0,
            "slippage_usd_total": 0.0, "slippage_count": 0,
            "slippage_unit": "basis points (1 bp = 0.01%)",
        }
        self.blocked_counts: dict[str, int] = {}

    # --- execution slippage ------------------------------------------
    def _record_slippage(self, sym: str, side: str, fill: float, bid, ask, qty: float) -> None:
        """Adverse execution cost of one fill, in basis points and USD.

        Reference is the book mid; a filled buy above / a filled sell below mid
        is slippage the engine pays. Fills that can't be priced (no bid+ask)
        are counted at 0 bps rather than skipped.
        """
        try:
            if bid is not None and ask is not None and ask > bid:
                mid = (bid + ask) / 2.0
            else:
                mid = None
        except TypeError:
            mid = None
        bps = 0.0
        if mid and mid > 0:
            bps = abs((fill - mid) / mid) * 1e4  # bps, always a cost to us
        usd = fill * qty * bps / 1e4
        self.tele["slippage_bps_total"] = float(self.tele.get("slippage_bps_total", 0.0)) + bps
        self.tele["slippage_usd_total"] = float(self.tele.get("slippage_usd_total", 0.0)) + usd
        self.tele["slippage_count"] = int(self.tele.get("slippage_count", 0)) + 1
        series = self.tele.setdefault("slippage_series", [])
        series.append({"t": time.time(), "sym": sym, "side": side,
                       "fill": round(fill, 2), "bps": round(bps, 4), "usd": round(usd, 4)})
        if len(series) > 300:
            del series[:-300]
        self.tele["slippage_last_bps"] = round(bps, 4)
        self.tele["slippage_avg_bps"] = round(
            self.tele["slippage_bps_total"] / max(1, self.tele["slippage_count"]), 4)
        self.last_px: dict[str, float] = {}
        self.tp_pct, self.sl_pct = 0.0015, 0.0015
        self.strategy_file = self.rec / "strategy.json"

    # --- safety guards -------------------------------------------------

    def refresh_safety(self) -> None:
        """Recompute halted/killswitch flags from live state.

        `halted` derives from realized total_pnl (daily-loss breaker), so it only
        clears when PnL recovers above the limit; `killswitch` mirrors the flag
        file. Neither is mutated by accounting — they only gate new entries.
        """
        self.tele["killswitch"] = self.killflag.exists()
        self.tele["halted"] = bool(self.tele["total_pnl"] <= -self.settings.max_daily_loss)

    def entry_blocked(self, sym: str, qty: float) -> str | None:
        """Reason an entry for `sym`/`qty` is blocked, else None.

        Order: cross-process kill switch, daily-loss breaker, position cap. An
        already-open position consumes part of the cap.
        """
        self.refresh_safety()
        if self.tele["killswitch"]:
            self._bump("kill_switch")
            return "kill_switch"
        if self.tele["halted"]:
            self._bump("daily_loss")
            return "daily_loss"
        held = abs(self.open_pos.get(sym, {}).get("qty", 0.0))
        if held + abs(qty) > self.settings.max_position:
            self._bump("position_cap")
            return "position_cap"
        return None

    def _bump(self, reason: str) -> None:
        self.blocked_counts[reason] = self.blocked_counts.get(reason, 0) + 1
        self.tele["blocked"] = sum(self.blocked_counts.values())

    # --- strategy / telemetry ------------------------------------------

    def read_strategy(self) -> None:
        """Pick up the LLM's TP/SL if the strategy tier has published one."""
        try:
            if self.strategy_file.is_file():
                d = json.loads(self.strategy_file.read_text())
                self.tp_pct = float(d.get("take_profit_pct") or self.tp_pct)
                self.sl_pct = float(d.get("stop_loss_pct") or self.sl_pct)
                self.tele["strategy_regime"] = d.get("regime")
        except (OSError, ValueError):
            pass

    def write_telemetry(self) -> None:
        self.tele["updated_at"] = time.time()
        try:
            (self.rec / "telemetry.json").write_text(json.dumps(self.tele))
        except OSError:
            pass

    # --- tick processing ------------------------------------------------

    def process(self, tick: Tick) -> None:
        sym = tick.symbol
        if sym not in self.buffers:
            self.buffers[sym] = RingBuffer(capacity=self.seq)
            self.counts[sym] = 0
        buf = self.buffers[sym]
        buf.append(tick)
        self.counts[sym] += 1
        self.tele["ticks_processed"] += 1

        # exits first (real subsequent prices)
        pos = self.open_pos.get(sym)
        if pos:
            ltp = tick.ltp
            reason = "take_profit" if ltp >= pos["tp"] else ("stop_loss" if ltp <= pos["sl"] else None)
            if reason:
                pnl = (ltp - pos["entry"]) * pos["qty"]
                self.tele["total_pnl"] += pnl
                self.tele["trades"] += 1
                self.tele["wins" if pnl > 0 else "losses"] += 1
                self._record_slippage(sym, "exit", ltp, tick.bid, tick.ask, pos["qty"])
                self.open_pos.pop(sym, None)

        if buf.is_full and self.counts[sym] % self.stride == 0:
            try:
                prob = float(self.model.predict(window_from_snapshot(buf.snapshot())))
            except Exception:  # noqa: BLE001
                return
            self.tele["inferences"] += 1
            self.tele["last_prob"] = prob
            if prob >= self.settings.prob_threshold and sym not in self.open_pos:
                qty = self.settings.quantities.get(sym, self.settings.max_position)
                # Safety gate: kill switch -> daily loss -> position cap.
                if not self.entry_blocked(sym, qty):
                    entry = tick.ask or tick.ltp
                    self.open_pos[sym] = {
                        "entry": entry, "qty": qty,
                        "tp": entry * (1 + self.tp_pct), "sl": entry * (1 - self.sl_pct), "prob": prob,
                    }
                    self._record_slippage(sym, "entry", entry, tick.bid, tick.ask, qty)
                    self.tele["entries"] += 1
            self.tele["open_positions"] = len(self.open_pos)
        # mark-to-market the open book (what the position is worth right now)
        self.last_px[sym] = tick.ltp
        self.tele["unrealized_pnl"] = round(
            sum((self.last_px.get(s, p["entry"]) - p["entry"]) * p["qty"] for s, p in self.open_pos.items()), 6
        )


def main() -> None:
    log = _get_logger()
    settings = get_settings()

    # ------------------------------------------------------------------
    # LIVE SEAM (GATED — DO NOT ENABLE).
    #
    # The deployed pipeline is 100% paper. The broker factory
    # (entrypoint.build_broker) already selects a real coinbase broker when
    # live_trading + coinbase_execution + market_data_source=coinbase are all
    # set. Wire the Broker contract (get_depth/place_order/get_position from
    # src/quant_crypto/broker/base.py) into EngineWorkerCore here to route
    # entries/exits through a real broker. Until that is wired, flipping
    # live_trading hard-fails at startup rather than silently paper-trading
    # real-sized signals.
    # ------------------------------------------------------------------
    if settings.live_trading:
        raise RuntimeError(
            "engine_worker live seam not yet wired via the Broker contract "
            "(src/quant_crypto/broker/base.py) — refusing to trade. "
            "Wire build_broker() into the entry path before setting LIVE_TRADING=1."
        )

    rec = Path(os.environ.get("RECORD_DIR", "/data/tapes"))
    rec.mkdir(parents=True, exist_ok=True)
    tape = rec / "latest.jsonl"
    model_path = _find_model()
    if not model_path:
        log.warning("no ONNX model found (ENGINE_MODEL / models/*.onnx) — engine worker idle")
        return
    from quant_crypto.model.infer import MambaInference

    model = MambaInference(
        model_path, seq_len=300, n_features=3, threshold=settings.prob_threshold
    )
    log.info("engine worker start: model=%s tape=%s", model_path, tape)

    core = EngineWorkerCore(model, settings, rec)

    async def loop() -> None:
        # tail the tape: start at current end, read new lines as they are appended
        pos_bytes = 0
        try:
            pos_bytes = tape.stat().st_size
        except OSError:
            pos_bytes = 0
        last_write = 0.0
        while True:
            try:
                # Cross-process kill switch is checked here (every iteration)
                # so the worker stops opening entries shortly after /api/v1/kill
                # even if no new ticks arrive.
                core.refresh_safety()
                if tape.is_file():
                    size = tape.stat().st_size
                    if size < pos_bytes:  # rotated
                        pos_bytes = 0
                    if size > pos_bytes:
                        with tape.open("r", encoding="utf-8") as fh:
                            fh.seek(pos_bytes)
                            for line in fh:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    core.process(Tick.model_validate_json(line))
                                except Exception:  # noqa: BLE001
                                    continue
                            pos_bytes = fh.tell()
            except OSError:
                pass
            now = time.time()
            if now - last_write > 1.0:
                core.read_strategy()
                core.write_telemetry()
                last_write = now
            await asyncio.sleep(0.5)

    try:
        asyncio.run(loop())
    except KeyboardInterrupt:
        core.write_telemetry()


if __name__ == "__main__":
    main()