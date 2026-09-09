# Crypto Quant Engine

Autonomous hybrid quantitative trading engine for **spot cryptocurrency (BTC/ETH)** fed by the Binance API. A sibling of the Nifty/BankNifty `trading-engine`: same four-tier architecture (cloud-LLM strategy → Mamba SSM model → zero-latency execution → feedback loop), retargeted at 24/7 crypto spot markets.

## Architecture

| Tier | Layer | Location | Cadence |
|------|-------|----------|---------|
| 1 | Strategy & Memory (Cloud LLM) | Cloud API | Periodic + post-trade |
| 2 | Regime & Model Runtime (Mamba SSM) | Local / Railway | Continuous |
| 3 | Execution (paper / gated live) | Local / Railway | Tick |
| 4 | Self-Improvement Feedback | Engine → Cloud | Per trade |

## Market data

- **Live:** Binance WebSocket agg-trades + kline streams (`btcusdt@aggTrade`, `btcusdt@kline_1s`/`1m`, same for `ethusdt`), recorded to append-only JSONL tapes.
- **History:** Binance REST `/api/v3/klines` for backtest/training corpus.
- **Analysis:** order-flow imbalance (taker buy/sell), per-tick Δprice/Δvolume — the `window.py` channel logic from `trading-engine`, adapted to spot.

## Setup

```bash
uv sync
cp .env.example .env   # fill in placeholders
```

## Run (paper mode)

```bash
uv run python scripts/run_engine.py <tape.jsonl> <model.onnx>
```

> **Warning:** Real orders are disabled by default. Live Binance trading is gated behind `LIVE_TRADING=1` plus API-key approval — same discipline as the parent engine. Crypto is extremely volatile; only risk what you can lose entirely.
