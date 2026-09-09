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
            from quant_crypto.engine.market_state import market_state_snapshot

            self._send(
                200,
                {
                    "killswitch": is_process_kill_switch_tripped(),
                    "flatten_registered": _FLATTEN_HANDLER is not None,
                    "telemetry": engine_snapshot(),
                    "marketstate": market_state_snapshot(),
                    "mode": "live" if (cfg.live_trading and cfg.binance_api_key) else "paper",
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
                {"available": capital, "withdrawable": 0.0, "source": "paper", "client": "binance-paper"},
            )
        elif path == "/api/v1/marketstate":
            from quant_crypto.engine.market_state import market_state_snapshot

            self._send(200, market_state_snapshot() or {})
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

    if os.environ.get("AUTO_RECORD") == "1":
        threading.Thread(target=_auto_record_thread, daemon=True, name="auto-record").start()
    log.info("starting crypto brain", extra={"extra_fields": {"port": port, "dashboard": str(_DASHBOARD)}})
    server = HTTPServer(("0.0.0.0", port), ControlHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def _auto_record_thread() -> None:
    """Record live Binance ticks to /data/tapes during operation (hosted)."""
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

    try:
        asyncio.run(loop())
    except Exception:  # noqa: BLE001
        log.exception("auto-recorder stopped unexpectedly")


def time_str() -> str:
    import time

    return time.strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    main()
