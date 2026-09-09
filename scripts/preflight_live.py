"""Live preflight gate — refuses to start a live session if unsafe.

    uv run python scripts/preflight_live.py [--model /data/models/<accepted>.onnx]

Exits non-zero on any FAIL. Run before every LIVE_TRADING=1 session.
"""

from __future__ import annotations

import argparse
import sys
import time

from quant_crypto.config import get_settings


def preflight_checks(settings, *, model_path=None, max_daily_loss_cap=None):
    """Return a list of (ok, message) pairs; ALL must pass to go live."""
    checks: list[tuple[bool, str]] = []
    live = bool(settings.live_trading)
    checks.append((bool(settings.binance_api_key), "BINANCE_API_KEY set"))
    checks.append((bool(settings.binance_api_secret), "BINANCE_API_SECRET set"))
    if not live:
        checks.append((True, "LIVE_TRADING=0 — paper mode, safe to run"))
    else:
        checks.append((True, "LIVE_TRADING=1 — operator-supervised live mode"))
        checks.append((settings.max_daily_loss > 0, "MAX_DAILY_LOSS > 0"))
        if max_daily_loss_cap is not None:
            checks.append((settings.max_daily_loss <= max_daily_loss_cap,
                           f"MAX_DAILY_LOSS ({settings.max_daily_loss}) <= cap ({max_daily_loss_cap})"))
        if model_path is not None:
            import os
            checks.append((os.path.isfile(model_path), f"model artifact exists: {model_path}"))
        for sym in settings.crypto_symbols:
            checks.append((settings.quantities.get(sym, 0) > 0, f"quantity sized for {sym}"))
    return checks


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-flight gate before live crypto trading")
    p.add_argument("--model", default=None)
    p.add_argument("--max-daily-loss-cap", type=float, default=None)
    args = p.parse_args()
    settings = get_settings()
    results = preflight_checks(settings, model_path=args.model,
                               max_daily_loss_cap=args.max_daily_loss_cap)
    failures = 0
    for ok, msg in results:
        print(("PASS  " if ok else "FAIL  ") + msg)
        failures += 0 if ok else 1
    if failures:
        print(f"\n{len(results) - failures}/{len(results)} passed — DO NOT GO LIVE")
        sys.exit(1)
    print(f"\n{len(results)}/{len(results)} passed — cleared to start")


if __name__ == "__main__":
    main()
