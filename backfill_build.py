"""Backfill a real training corpus from Binance bulk aggTrades (data.binance.vision).

Reads the downloaded daily aggTrades CSVs for BTCUSDT/ETHUSDT, derives the same
3 channels the live model uses (OFI / price-delta / volume-delta), slices
windows, z-scores, labels (forward velocity > 0 over horizon) and writes .npz
for training — directly, so ~1M ticks never materialise as Python objects.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np

SEQ, HORIZON, STRIDE, THRESHOLD = 300, 10, 25, 0.0
OUT = Path("/opt/data/corpus_backfill")
OUT.mkdir(parents=True, exist_ok=True)

JOBS = [
    ("/opt/data/backfill/btc.zip", "BTC-USD"),
    ("/opt/data/backfill/eth.zip", "ETH-USD"),
]


def load_symbol(zip_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ((n, 3) raw channels, prices) for one symbol from its aggTrades CSV."""
    zf = zipfile.ZipFile(zip_path)
    name = zf.namelist()[0]
    prices, qtys, buyer_maker = [], [], []
    with zf.open(name) as fh:
        reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8"))
        next(reader, None)  # header
        for row in reader:
            if len(row) < 7:
                continue
            prices.append(float(row[1]))
            qtys.append(float(row[2]))
            buyer_maker.append(row[6].strip().lower() == "true")
    prices = np.asarray(prices, dtype=np.float64)
    qtys = np.asarray(qtys, dtype=np.float64)
    bm = np.asarray(buyer_maker, dtype=bool)
    buy_cum = np.cumsum(np.where(~bm, qtys, 0.0))
    sell_cum = np.cumsum(np.where(bm, qtys, 0.0))
    vol_cum = np.cumsum(qtys)
    n = len(prices)
    c0 = np.zeros(n)
    db = np.diff(buy_cum)
    ds = np.diff(sell_cum)
    denom = np.abs(db) + np.abs(ds)
    np.divide(db - ds, denom, out=c0[1:], where=denom > 0)
    c1 = np.zeros(n)
    c1[1:] = np.diff(prices)
    c2 = np.zeros(n)
    c2[1:] = np.diff(vol_cum)
    return np.stack([c0, c1, c2], axis=1), prices


def build_windows(ch: np.ndarray, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = ch.shape[0]
    xs, ys = [], []
    start = 0
    while start + SEQ + HORIZON <= n:
        win = ch[start:start + SEQ]
        z = (win - win.mean(axis=0)) / (win.std(axis=0) + 1e-12)
        future = prices[start + SEQ:start + SEQ + HORIZON]
        label = 1 if float(future.mean() - prices[start + SEQ - 1]) > THRESHOLD else 0
        xs.append(z.astype(np.float32))
        ys.append(label)
        start += STRIDE
    if not xs:
        return np.zeros((0, SEQ, 3), np.float32), np.zeros((0,), np.int8)
    return np.stack(xs), np.asarray(ys, dtype=np.int8)


def main() -> None:
    all_x, all_y = [], []
    for zpath, sym in JOBS:
        print(f"loading {sym} from {zpath}", flush=True)
        ch, prices = load_symbol(zpath)
        X, y = build_windows(ch, prices)
        print(f"{sym}: {ch.shape[0]} ticks -> {X.shape[0]} windows ({int(y.sum())} pos)", flush=True)
        if X.shape[0]:
            np.savez(OUT / f"{sym}.npz", x=X, y=y, meta=json.dumps({"n": int(X.shape[0])}).encode())
            all_x.append(X)
            all_y.append(y)
    if all_x:
        X = np.concatenate(all_x)
        Y = np.concatenate(all_y)
        # cap training set to keep CPU training feasible
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X))[:20000]
        np.savez(OUT / "train_subset.npz", x=X[idx], y=Y[idx])
        print(f"combined {X.shape[0]} windows -> capped train subset {idx.shape[0]}", flush=True)


if __name__ == "__main__":
    main()
