"""Memory-light trainer for the crypto Mamba classifier.

The container has <1GB free, so full-batch training over 300-step unrolled
sequences OOMs. This trains in mini-batches with a smaller model, then exports
a parity-checked ONNX.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from quant_crypto.model.mamba_model import MambaClassifier
from quant_crypto.model.onnx_export import check_parity, export_onnx, validate_onnx
from quant_crypto.model.train import save_checkpoint, split_windows

torch.set_num_threads(2)

SEQ, NF = 300, 3
OUT = Path("/opt/data/models")
OUT.mkdir(parents=True, exist_ok=True)

xs, ys = [], []
for p in sorted(Path("/opt/data/corpus_v1").glob("*.npz")):
    d = np.load(p)
    xs.append(d["x"]); ys.append(d["y"])
x = np.concatenate(xs); y = np.concatenate(ys)
print(f"corpus: {x.shape[0]} windows, {int(y.sum())} pos", flush=True)

x_tr, y_tr, x_va, y_va = split_windows(x, y, val_fraction=0.2, seed=0)
xt = torch.from_numpy(x_tr); yt = torch.from_numpy(y_tr.astype(np.float32))
xv = torch.from_numpy(x_va); yv = torch.from_numpy(y_va.astype(np.float32))

torch.manual_seed(0)
model = MambaClassifier(n_features=NF, d_model=16, d_state=8, n_layers=1)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.BCELoss()

BATCH, EPOCHS = 8, 20
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(xt))
    tot = 0.0
    for i in range(0, len(xt), BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        loss = loss_fn(model(xt[idx]), yt[idx])
        loss.backward(); opt.step()
        tot += float(loss.item())
    if ep % 10 == 0 or ep == EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(xv), yv).item())
        print(f"epoch {ep}: train {tot:.4f} val {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)

model.eval()
stamp = time.strftime("%Y%m%d-%H%M%S")
OUT.mkdir(parents=True, exist_ok=True)  # the tree can be reset mid-run
ckpt = OUT / f"{stamp}.pt"
save_checkpoint(model, str(ckpt))
onnx = OUT / f"{stamp}.onnx"
export_onnx(model, str(onnx), seq_len=SEQ, n_features=NF)
validate_onnx(str(onnx))
err = check_parity(model, str(onnx), seq_len=SEQ, n_features=NF)
report = {"artifact": onnx.name, "windows": int(x.shape[0]), "n_pos": int(y.sum()),
          "parity_err": err, "seq_len": SEQ, "n_features": NF,
          "d_model": 16, "n_layers": 1}
(OUT / f"{stamp}.report.json").write_text(json.dumps(report, indent=2))
print(f"parity err={err:.2e}", flush=True)
if err > 1e-4:
    raise SystemExit("parity gate FAILED")
# stable name the engine looks for
import shutil
shutil.copy(onnx, OUT / "mamba.onnx")
print("released:", onnx.name, "-> mamba.onnx", flush=True)
