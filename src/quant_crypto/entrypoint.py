"""Process entrypoint (Railway / Docker).

Serves the crypto Mission Control control surface over HTTP:
- GET  /health                liveness probe
- GET  /api/v1/status         kill-switch state + engine/market snapshot
- POST /api/v1/kill           trip the master kill-switch
- POST /api/v1/resume         clear the master kill-switch
- POST /api/v1/flatten        force-flatten registered handler
- GET  /api/v1/marketstate    live order-flow snapshot (if feed publishing)
- GET  /api/v1/feed           recent dashboard feed events
- GET  /api/v1/funds          paper capital (never a real balance unless live)

The Mission Control dashboard (dashboard/index.html) calls these endpoints
directly. The kill-switch is process-global (quant_crypto.exec.risk).
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from quant_crypto.config import get_settings
from quant_crypto.exec.risk import (
    is_process_kill_switch_tripped,
    reset_process_kill_switch,
    trip_process_kill_switch,
)
from quant_crypto.logging_util import get_logger

log = get_logger("quant_crypto.entrypoint")


def _resolve_dashboard() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "dashboard" / "index.html",  # repo checkout
        Path.cwd() / "dashboard" / "index.html",       # container WORKDIR
        Path("/app/dashboard/index.html"),             # container, explicit
    ]
    override = os.environ.get("DASHBOARD_PATH")
    if override:
        candidates.insert(0, Path(override))
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


_DASHBOARD = _resolve_dashboard()
_FLATTEN_HANDLER = None


def set_flatten_handler(handler) -> None:
    global _FLATTEN_HANDLER
    _FLATTEN_HANDLER = handler


def _marketstate_from_tape() -> dict:
    """Read the latest ticks from the shared tape file written by the recorder
    subprocess (cross-process, since the feed can't share the HTTP process)."""
    rec = os.environ.get("RECORD_DIR", "/data/tapes")
    latest = Path(rec) / "latest.jsonl"
    rows = []
    try:
        if latest.is_file() and latest.stat().st_size > 0:
            with latest.open() as fh:
                lines = fh.readlines()[-200:]
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return {}
    if not rows:
        return {}
    last = rows[-1]
    # Aggregate per-symbol so bid/ask/ltp are coherent (don't mix BTC and ETH).
    last_sym = last.get("symbol")
    sym_rows = [r for r in rows if r.get("symbol") == last_sym]
    if not sym_rows:
        sym_rows = rows
    bids = [r.get("bid", 0) for r in sym_rows if r.get("bid")]
    asks = [r.get("ask", 0) for r in sym_rows if r.get("ask")]
    buy_vol = sum(r.get("bid_qty", 0) for r in sym_rows)
    sell_vol = sum(r.get("ask_qty", 0) for r in sym_rows)
    total = buy_vol + sell_vol
    ltp = last.get("ltp")
    first = sym_rows[0]
    first_ltp = first.get("ltp") or ltp
    price_delta = (ltp - first_ltp) if (ltp and first_ltp) else 0.0
    first_oi = first.get("oi") or 0.0
    last_oi = last.get("oi") or 0.0
    volume_delta = last_oi - first_oi
    bid = min(bids) if bids else None
    ask = max(asks) if asks else None
    spread = (ask - bid) if (bid is not None and ask is not None) else None
    spread_bps = (spread / ltp * 10000.0) if (spread is not None and ltp) else None
    return {
        "ltp": ltp,
        "symbol": last_sym,
        "bid": bid,
        "ask": ask,
        "spread": round(spread, 4) if spread is not None else None,
        "spread_bps": round(spread_bps, 2) if spread_bps is not None else None,
        "taker_imbalance": round((buy_vol - sell_vol) / total, 4) if total > 0 else None,
        "price_delta": round(price_delta, 6),
        "price_delta_bps": round(price_delta / first_ltp * 10000.0, 2) if first_ltp else None,
        "volume_delta": round(volume_delta, 4),
        "window": len(sym_rows),
        "ticks_seen": len(sym_rows),
        "healthy": True,
    }


_EQUITY_HISTORY: list = []


def _telemetry_from_file() -> dict | None:
    """Engine telemetry published by the engine worker subprocess, if any."""
    rec = os.environ.get("RECORD_DIR", "/data/tapes")
    p = Path(rec) / "telemetry.json"
    try:
        if p.is_file():
            d = json.loads(p.read_text())
            if isinstance(d, dict):
                return d
    except (OSError, ValueError):
        pass
    return None


def _equity_snapshot() -> dict:
    """Rolling PnL series for the equity curve (from engine telemetry)."""
    import time as _t

    from quant_crypto.engine.telemetry import engine_snapshot

    snap = _telemetry_from_file() or engine_snapshot() or {}
    pnl = float(snap.get("total_pnl", 0.0) or 0.0)
    if not _EQUITY_HISTORY or _EQUITY_HISTORY[-1]["pnl"] != pnl:
        _EQUITY_HISTORY.append({"t": _t.time(), "pnl": pnl})
        del _EQUITY_HISTORY[:-600]
    return {
        "points": _EQUITY_HISTORY,
        "total_pnl": pnl,
        "trades": snap.get("trades", 0),
    }


def _pipeline_steps() -> list:
    """Pipeline display rows from real telemetry counters (or idle)."""
    try:
        from quant_crypto.engine.telemetry import build_pipeline_steps

        snap = _telemetry_from_file()
        if snap:
            return build_pipeline_steps(snap)
    except Exception:  # noqa: BLE001
        pass
    return []


class ControlHandler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_dashboard(self) -> bool:
        try:
            html = _DASHBOARD.read_bytes()
        except OSError:
            log.error("dashboard not found — / will 404", extra={"looked_at": str(_DASHBOARD)})
            return False
        self._send_bytes(200, "text/html; charset=utf-8", html)
        return True

    def _route_get(self) -> bool:
        path = self.path.split("?", 1)[0]
        cfg = get_settings()
        if path == "/health":
            self._send(200, {"status": "ok"})
        elif path == "/api/v1/status":
            from quant_crypto.engine.telemetry import engine_snapshot

            ms = _marketstate_from_tape() or {}
            self._send(
                200,
                {
                    "killswitch": is_process_kill_switch_tripped(),
                    "flatten_registered": _FLATTEN_HANDLER is not None,
                    "telemetry": _telemetry_from_file() or engine_snapshot(),
                    "marketstate": ms,
                    "feed": {
                        "recorder_armed": os.environ.get("AUTO_RECORD") == "1",
                        "source": cfg.market_data_source,
                        "ticks_received": ms.get("ticks_seen", 0),
                        "symbol": ms.get("symbol"),
                        "market_open": True if ms else None,
                    },
                    "mode": "live" if cfg.live_trading else "paper",
                    "trading": bool(cfg.live_trading),
                    "engine": {
                        "symbols": cfg.crypto_symbols,
                        "quantities": cfg.quantities,
                        "max_position": cfg.max_position,
                        "max_daily_loss": cfg.max_daily_loss,
                        "prob_threshold": cfg.prob_threshold,
                    },
                },
            )
        elif path == "/api/v1/funds":
            capital = float(cfg.paper_starting_capital)
            self._send(
                200,
                {"available": capital, "withdrawable": 0.0, "source": "paper",
                 "client": f"{cfg.market_data_source}-paper"},
            )
        elif path == "/api/v1/marketstate":
            self._send(200, _marketstate_from_tape() or {})
        elif path == "/api/v1/equity":
            self._send(200, _equity_snapshot())
        elif path == "/api/v1/pipeline":
            self._send(200, {"steps": _pipeline_steps()})
        elif path == "/api/v1/feed":
            from urllib.parse import parse_qs

            from quant_crypto.engine.events import feed_events_since

            qs = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            try:
                since = int((qs.get("since") or ["0"])[0])
            except ValueError:
                since = 0
            events, latest = feed_events_since(since)
            self._send(200, {"events": events, "latest": latest})
        elif path in ("/", "/index.html", "/dashboard", "/dashboard/index.html"):
            if not self._serve_dashboard():
                return False
        else:
            return False
        return True

    def _route_post(self) -> bool:
        if self.path == "/api/v1/kill":
            trip_process_kill_switch()
            self._send(200, {"killswitch": True})
        elif self.path == "/api/v1/resume":
            reset_process_kill_switch()
            self._send(200, {"killswitch": False})
        elif self.path == "/api/v1/flatten":
            if _FLATTEN_HANDLER is None:
                self._send(501, {"error": "no flatten handler registered"})
            else:
                self._send(200, {"status": "ok", "flatten": _FLATTEN_HANDLER()})
        else:
            return False
        return True

    def do_GET(self) -> None:
        if not self._route_get():
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._route_post():
            self._send(404, {"error": "not found"})

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):  # noqa: A002
        log.info("http", extra={"extra_fields": {"path": self.path}})


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    from quant_crypto.config import get_settings

    # The WebSocket feed + HTTP server cannot share a process: on Python 3.13,
    # a second live thread starves the feed's asyncio event loop (0 ticks). So
    # when AUTO_RECORD is set, the recorder runs as its OWN subprocess.
    if os.environ.get("AUTO_RECORD") == "1":
        _spawn_recorder()
    if os.environ.get("ENGINE_SESSION") == "1":
        _spawn_engine()
    if os.environ.get("STRATEGY_SESSION") == "1":
        threading.Thread(target=_strategy_thread, daemon=True, name="llm-strategy").start()
    log.info("starting crypto brain", extra={"extra_fields": {"port": port, "dashboard": str(_DASHBOARD)}})
    server = HTTPServer(("0.0.0.0", port), ControlHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def _strategy_thread() -> None:
    """Periodic Tier-1 LLM strategy run — publishes dashboard feed events.

    Builds a market snapshot from the live tape, asks the LLM to pick a regime
    and execution params, persists the config (so the engine worker can read
    TP/SL cross-process) and surfaces the decision in the feed.
    """
    import time as _t

    from quant_crypto.cloud.librarian import Librarian
    from quant_crypto.cloud.llm import get_client
    from quant_crypto.cloud.premarket import PremarketSnapshot
    from quant_crypto.cloud.strategy_manager import StrategyManager

    settings = get_settings()
    provider = "anthropic" if settings.anthropic_api_key else ("openai" if settings.openai_api_key else "")
    if not provider:
        log.warning("STRATEGY_SESSION=1 but no LLM API key set — strategy idle")
        return
    try:
        mgr = StrategyManager(get_client(settings, provider),
                              librarian=Librarian(os.environ.get("QUANT_DB_PATH", "/data/librarian.db")))
    except Exception:  # noqa: BLE001
        log.exception("LLM strategy init failed")
        return
    interval = float(os.environ.get("STRATEGY_INTERVAL_S", "900"))
    rec = Path(os.environ.get("RECORD_DIR", "/data/tapes"))
    while True:
        try:
            ms = _marketstate_from_tape() or {}
            snap = PremarketSnapshot(
                btc_24h_return_pct=float(ms.get("price_delta_bps") or 0.0),
                btc_volatility_pct=float(ms.get("spread_bps") or 0.0),
                global_taker_imbalance=float(ms.get("taker_imbalance") or 0.0),
                source="live-tape",
            )
            cfg = mgr.run_daily(snap)
            try:
                rec.mkdir(parents=True, exist_ok=True)
                (rec / "strategy.json").write_text(cfg.model_dump_json())
            except OSError:
                pass
            log.info("LLM strategy applied", extra={"extra_fields": {"regime": cfg.regime.value}})
        except Exception:  # noqa: BLE001
            log.exception("strategy run failed")
        _t.sleep(interval)


def _spawn_engine() -> None:
    """Launch the paper inference engine as a separate subprocess."""
    import subprocess
    import sys

    try:
        subprocess.Popen(
            [sys.executable, "-m", "quant_crypto.engine_worker"],
            env=dict(os.environ), start_new_session=True,
        )
        log.info("engine worker spawned as subprocess")
    except Exception:  # noqa: BLE001
        log.exception("failed to spawn engine worker")


def _spawn_recorder() -> None:
    """Launch the live tick recorder as a separate subprocess.

    Uses the same module via `python -m quant_crypto.recorder_worker` so the
    feed's event loop owns a whole process (no thread starvation).
    """
    import subprocess
    import sys

    env = dict(os.environ)
    cmd = [sys.executable, "-m", "quant_crypto.recorder_worker"]
    try:
        subprocess.Popen(cmd, env=env, start_new_session=True)
        log.info("auto-recorder spawned as subprocess")
    except Exception:  # noqa: BLE001
        log.exception("failed to spawn recorder subprocess")


def _auto_record_thread() -> None:
    """Record live Binance ticks to /data/tapes during operation (hosted).

    Runs in a daemon thread, so it MUST own a dedicated event loop: the parent
    thread (the HTTP server) has no loop, and `asyncio.run()` here creates a
    transient one that can fail to schedule the WebSocket. We create and set a
    loop for this thread explicitly and run it until stop.
    """
    import asyncio

    from quant_crypto.config import get_settings
    from quant_crypto.data.binance_feed import BinanceTickFeed
    from quant_crypto.data.ingest import run_ingest
    from quant_crypto.data.tape import TapeWriter

    settings = get_settings()
    out_dir = Path(os.environ.get("RECORD_DIR", "/data/tapes"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = time_str()
    out = out_dir / f"{tag}_TICKS.jsonl"
    symbols = [s.upper() for s in settings.crypto_symbols]
    log.info("auto-recorder armed", extra={"extra_fields": {"symbols": symbols, "out": str(out)}})
    feed = BinanceTickFeed(symbols)

    def on_tick(tick):
        try:
            from quant_crypto.engine.market_state import publish_tick

            publish_tick(tick)
        except Exception:  # noqa: BLE001
            pass

    writer = TapeWriter(out, buffer_size=2000)

    async def loop():
        try:
            await run_ingest(feed, on_tick=lambda t: (writer.write(t), on_tick(t)), register_active=True)
        finally:
            writer.close()

    # Dedicated event loop owned by this thread.
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    try:
        _loop.run_until_complete(loop())
    except Exception:  # noqa: BLE001
        log.exception("auto-recorder stopped unexpectedly")
    finally:
        _loop.close()


def time_str() -> str:
    import time

    return time.strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    main()
