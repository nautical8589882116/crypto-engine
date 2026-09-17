"""Finish the export step after train_real.py (parity must use batch 1).

The runtime inference feeds ONE window at a time, and export_onnx bakes a fixed
batch dimension of 1, so the parity check has to compare a single window — not
a 4-row slice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, "src")
from quant_crypto.model.mamba_model import MambaClassifier  # noqa: E402
from quant_crypto.model.onnx_export import validate_onnx  # noqa: E402

MODELS = Path("/opt/data/models")
CORPUS = Path("/opt/data/corpus_backfill/train_subset.npz")


def newest(suffix: str) -> Path:
    files = sorted(MODELS.glob(f"*{suffix}"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no {suffix} in {MODELS}")
    return files[-1]


def main() -> None:
    ckpt, onnx_path = newest(".pt"), newest(".onnx")
    # Safe: this checkpoint holds only tensors + a plain-int config dict.
    blob = torch.load(ckpt, weights_only=True)
    cfg = blob.get("config", {"n_features": 3, "d_model": 16, "n_layers": 1})
    model = MambaClassifier(**cfg)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    d = np.load(CORPUS)
    window = d["x"][:1].astype(np.float32)  # batch of exactly 1

    with torch.no_grad():
        ref = model(torch.from_numpy(window)).numpy().reshape(-1)

    validate_onnx(str(onnx_path))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: window})[0].reshape(-1)
    err = float(np.abs(ref - got).max())
    print(f"checkpoint: {ckpt.name}")
    print(f"onnx      : {onnx_path.name}")
    print(f"input     : {[i.shape for i in sess.get_inputs()]} -> {[o.shape for o in sess.get_outputs()]}")
    print(f"parity err= {err:.2e}  (torch {ref[0]:.6f} vs onnx {got[0]:.6f})")
    if err > 1e-4:
        raise SystemExit("parity check FAILED — refusing to release")

    (MODELS / "mamba.onnx").write_bytes(onnx_path.read_bytes())
    report = {
        "released_from": onnx_path.name,
        "parity_err": err,
        "windows_trained": 1500,
        "positive": 682,
        "val_loss": 0.6677,
        "random_baseline": 0.693,
        "config": cfg,
        "corpus": str(CORPUS),
        "source": "binance-backfill (2,129,835 trades -> 85,169 windows)",
        "note": "marginal signal (~0.025 below random); infrastructure demo, not a tradeable edge",
    }
    (MODELS / "mamba.report.json").write_text(json.dumps(report, indent=2))
    print(f"released -> {MODELS/'mamba.onnx'} ({onnx_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
