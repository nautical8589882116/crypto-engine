"""Corpus builder — turn recorded crypto tick tapes into Mamba training windows.

    uv run python scripts/build_dataset.py --tapes /data/tapes/2026-09-09.jsonl --out /data/corpus/v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quant_crypto.data.binance_feed import Tick
from quant_crypto.data.dataset import CorpusParams, build_windows


def _load_tape(path: Path) -> list[Tick]:
    ticks = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                ticks.append(Tick.model_validate_json(line))
    return ticks


def main() -> None:
    p = argparse.ArgumentParser(description="Build model windows from tick tapes")
    p.add_argument("--tapes", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seq", type=int, default=300)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--gap-max", type=float, default=5.0)
    args = p.parse_args()

    params = CorpusParams(
        seq_len=args.seq, horizon=args.horizon,
        threshold=args.threshold, stride=args.stride, gap_max_s=args.gap_max,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"params": params.__dict__, "days": {}}
    total = 0
    for raw in args.tapes:
        tape = Path(raw)
        ticks = _load_tape(tape)
        if len(ticks) < params.seq_len + params.horizon:
            print(f"skip {tape.name}: only {len(ticks)} ticks")
            continue
        X, y, meta = build_windows(ticks, params)
        stem = tape.stem
        npz = out_dir / f"{stem}.npz"
        np.savez(npz, x=X, y=y, meta=json.dumps(meta).encode())
        manifest["days"][stem] = {"ticks": len(ticks), "windows": int(X.shape[0]), "pos": int(y.sum())}
        total += int(X.shape[0])
        print(f"{tape.name}: {len(ticks)} ticks -> {X.shape[0]} windows ({int(y.sum())} pos) -> {npz.name}")
    (out_dir / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"total windows: {total} -> {out_dir}")


if __name__ == "__main__":
    main()
