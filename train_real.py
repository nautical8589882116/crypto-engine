"""Train the crypto Mamba classifier on a backfilled real-trade corpus.

CPU-only and memory-capped (<1GB free), so: mini-batches, a small model, and a
bounded window subset. Exports ONNX at the runtime seq_len and checks parity.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from quant_crypto.model.mamba_model import MambaClassifier
from quant_crypto.model.onnx_export import export_onnx, validate_onnx

SEQ = 300


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="npz with x,y (or a dir of them)")
    ap.add_argument("--out-dir", default="/opt/data/models")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=4000, help="max windows to train on")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d-model", type=int, default=16)
    ap.add_argument("--n-layers", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(2)
    xs, ys = [], []
    p = Path(args.corpus)
    files = sorted(p.glob("*.npz")) if p.is_dir() else [p]
    for f in files:
        d = np.load(f)
        if "x" in d and d["x"].size:
            xs.append(d["x"])
            ys.append(d["y"])
    X = np.concatenate(xs).astype(np.float32)
    Y = np.concatenate(ys).astype(np.float32)
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(X))[: args.limit]
    X, Y = X[idx], Y[idx]
    print(f"training on {len(X)} windows, {int(Y.sum())} positive", flush=True)

    n_val = max(1, int(len(X) * 0.2))
    Xtr, Ytr, Xva, Yva = X[n_val:], Y[n_val:], X[:n_val], Y[:n_val]
    model = MambaClassifier(n_features=3, d_model=args.d_model, n_layers=args.n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()
    xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(Ytr)
    xv = torch.from_numpy(Xva)
    yv = torch.from_numpy(Yva)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), args.batch):
            sel = perm[i : i + args.batch]
            opt.zero_grad()
            out = model(xt[sel])
            loss = loss_fn(out, yt[sel])
            loss.backward()
            opt.step()
            tot += float(loss) * len(sel)
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(xv), yv))
        print(f"epoch {ep}: train {tot/len(xt):.4f} val {vloss:.4f} ({time.time()-t0:.0f}s)", flush=True)

    model.eval()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ckpt = out / f"{stamp}.pt"
    torch.save({"state_dict": model.state_dict(),
                "config": {"n_features": 3, "d_model": args.d_model, "n_layers": args.n_layers}}, ckpt)
    onnx_path = out / f"{stamp}.onnx"
    export_onnx(model, str(onnx_path), seq_len=SEQ)
    validate_onnx(str(onnx_path))
    with torch.no_grad():
        a = model(torch.from_numpy(Xva[:1])).numpy()  # export bakes batch=1
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    b = sess.run(None, {sess.get_inputs()[0].name: Xva[:1]})[0].reshape(-1)
    print(f"parity err={np.abs(a - b).max():.2e}", flush=True)
    (out / "mamba.onnx").write_bytes(onnx_path.read_bytes())
    report = {"windows": int(len(X)), "positive": int(Y.sum()), "val_loss": vloss,
              "d_model": args.d_model, "n_layers": args.n_layers, "epochs": args.epochs,
              "source": "binance-backfill", "trained_at": stamp}
    (out / f"{stamp}.report.json").write_text(json.dumps(report, indent=2))
    print("released:", onnx_path.name, "-> mamba.onnx", flush=True)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
