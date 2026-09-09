"""Retrain the Mamba model on the crypto corpus and ship a parity-gated ONNX.

    uv run python scripts/retrain.py --corpus /data/corpus/v1 --out-dir /data/models [--epochs 200]

Requires the [train] extra (torch + onnx). Exits non-zero if ONNX parity exceeds
1e-4 — a bad artifact never ships.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from quant_crypto.model.onnx_export import check_parity, export_onnx, validate_onnx
from quant_crypto.model.train import load_checkpoint, save_checkpoint, train_on_windows


def _load_corpus(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for p in paths:
        d = np.load(p)
        xs.append(d["x"])
        ys.append(d["y"])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    return x, y


def main() -> None:
    p = argparse.ArgumentParser(description="Retrain Mamba on the crypto corpus")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--parity-tol", type=float, default=1e-4)
    args = p.parse_args()

    npz = sorted(args.corpus.glob("*.npz"))
    if not npz:
        raise SystemExit(f"no .npz corpus files under {args.corpus}")
    x, y = _load_corpus(npz)
    print(f"corpus: {len(npz)} file(s), {x.shape[0]} windows, {int(y.sum())} positive")

    t0 = time.time()
    out = train_on_windows(x, y, epochs=args.epochs, lr=args.lr,
                           val_fraction=args.val_fraction, seed=args.seed)
    model = out["model"]
    print(f"train loss={out['loss']:.4f} val loss={out['val_loss']:.4f} ({time.time()-t0:.0f}s)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    seq_len, n_features = x.shape[1], x.shape[2]
    ckpt = args.out_dir / f"{stamp}.pt"
    save_checkpoint(model, str(ckpt))
    onnx = args.out_dir / f"{stamp}.onnx"
    export_onnx(model, str(onnx), seq_len=seq_len, n_features=n_features)
    validate_onnx(str(onnx))
    err = check_parity(model, str(onnx), seq_len=seq_len, n_features=n_features)
    print(f"onnx parity err={err:.2e} (tol {args.parity_tol})")
    report = {
        "artifact": onnx.name, "checkpoint": ckpt.name, "n_windows": int(x.shape[0]),
        "n_pos": int(y.sum()), "loss": out["loss"], "val_loss": out["val_loss"],
        "parity_err": err, "epochs": args.epochs, "seed": args.seed,
        "seq_len": seq_len, "n_features": n_features,
    }
    (args.out_dir / f"{stamp}.report.json").write_text(json.dumps(report, indent=2))
    if err > args.parity_tol:
        raise SystemExit(f"parity gate FAILED ({err:.2e}) — artifact not released")
    print(f"released: {onnx.name} (val_loss={out['val_loss']:.4f})")


if __name__ == "__main__":
    main()
