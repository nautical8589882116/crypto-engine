"""Engine worker — live paper inference over the recorded tape, in its own process.

The recorder owns the websocket; this worker tails the tape it writes, runs the
Mamba ONNX model over rolling windows, simulates paper entries/exits, and
publishes telemetry to `telemetry.json` (cross-process, like latest.jsonl) so
the control plane's dashboard can render inference, trades and PnL.

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


def main() -> None:
    log = _get_logger()
    settings = get_settings()
    rec = Path(os.environ.get("RECORD_DIR", "/data/tapes"))
    tape = rec / "latest.jsonl"
    out = rec / "telemetry.json"
    model_path = _find_model()
    if not model_path:
        log.warning("no ONNX model found (ENGINE_MODEL / models/*.onnx) — engine worker idle")
        return
    from quant_crypto.model.infer import MambaInference

    model = MambaInference(
        model_path, seq_len=300, n_features=3, threshold=settings.prob_threshold
    )
    log.info("engine worker start: model=%s tape=%s", model_path, tape)

    seq = model.seq_len
    stride = settings.model_stride
    buffers: dict[str, RingBuffer] = {}
    counts: dict[str, int] = {}
    open_pos: dict[str, dict] = {}
    tele = {
        "ticks_processed": 0, "inferences": 0, "trades": 0, "wins": 0,
        "losses": 0, "total_pnl": 0.0, "last_prob": None,
        "entries": 0, "open_positions": 0,
        "threshold": settings.prob_threshold, "session_source": "coinbase-live",
        "mode": "paper", "updated_at": time.time(),
    }
    tp_pct, sl_pct = 0.0015, 0.0015
    strategy_file = rec / "strategy.json"

    def read_strategy() -> None:
        """Pick up the LLM's TP/SL if the strategy tier has published one."""
        nonlocal tp_pct, sl_pct
        try:
            if strategy_file.is_file():
                d = json.loads(strategy_file.read_text())
                tp_pct = float(d.get("take_profit_pct") or tp_pct)
                sl_pct = float(d.get("stop_loss_pct") or sl_pct)
                tele["strategy_regime"] = d.get("regime")
        except (OSError, ValueError):
            pass

    def write_telemetry() -> None:
        tele["updated_at"] = time.time()
        try:
            out.write_text(json.dumps(tele))
        except OSError:
            pass

    def process(tick: Tick) -> None:
        sym = tick.symbol
        if sym not in buffers:
            buffers[sym] = RingBuffer(capacity=seq)
            counts[sym] = 0
        buf = buffers[sym]
        buf.append(tick)
        counts[sym] += 1
        tele["ticks_processed"] += 1

        # exits first (real subsequent prices)
        pos = open_pos.get(sym)
        if pos:
            ltp = tick.ltp
            reason = "take_profit" if ltp >= pos["tp"] else ("stop_loss" if ltp <= pos["sl"] else None)
            if reason:
                pnl = (ltp - pos["entry"]) * pos["qty"]
                tele["total_pnl"] += pnl
                tele["trades"] += 1
                tele["wins" if pnl > 0 else "losses"] += 1
                open_pos.pop(sym, None)

        if buf.is_full and counts[sym] % stride == 0:
            try:
                prob = float(model.predict(window_from_snapshot(buf.snapshot())))
            except Exception:  # noqa: BLE001
                return
            tele["inferences"] += 1
            tele["last_prob"] = prob
            if prob >= settings.prob_threshold and sym not in open_pos:
                qty = settings.quantities.get(sym, settings.max_position)
                entry = tick.ask or tick.ltp
                open_pos[sym] = {
                    "entry": entry, "qty": qty,
                    "tp": entry * (1 + tp_pct), "sl": entry * (1 - sl_pct), "prob": prob,
                }
                tele["entries"] += 1
            tele["open_positions"] = len(open_pos)

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
                                    process(Tick.model_validate_json(line))
                                except Exception:  # noqa: BLE001
                                    continue
                            pos_bytes = fh.tell()
            except OSError:
                pass
            now = time.time()
            if now - last_write > 1.0:
                read_strategy()
                write_telemetry()
                last_write = now
            await asyncio.sleep(0.5)

    try:
        asyncio.run(loop())
    except KeyboardInterrupt:
        write_telemetry()


if __name__ == "__main__":
    main()
