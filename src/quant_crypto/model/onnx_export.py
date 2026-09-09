"""ONNX export and validation for the Mamba classifier.

Exports a trained model to ONNX with a fixed sequence length and validates
the graph, then checks numeric parity against the PyTorch forward pass.
"""

from __future__ import annotations

import torch
import onnx
import onnxruntime as ort

from quant_crypto.model.mamba_model import MambaClassifier


def export_onnx(
    model: MambaClassifier,
    path: str,
    seq_len: int = 300,
    n_features: int = 3,
) -> None:
    """Export the model to ONNX with a static input shape (batch=1)."""
    model.eval()
    dummy = torch.randn(1, seq_len, n_features)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["prob"],
        dynamic_axes=None,  # fixed shape for predictable inference
        opset_version=17,
        dynamo=False,  # legacy tracer handles the sequential scan loop
    )


def validate_onnx(path: str) -> None:
    """Load and check the ONNX graph is well-formed."""
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)


def check_parity(
    model: MambaClassifier,
    onnx_path: str,
    seq_len: int = 300,
    n_features: int = 3,
    atol: float = 1e-5,
) -> float:
    """Return the max absolute error between PyTorch and ONNX Runtime output."""
    model.eval()
    x = torch.randn(1, seq_len, n_features)
    with torch.no_grad():
        torch_out = model(x).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x.numpy().astype("float32")})[0]

    import numpy as np

    return float(np.max(np.abs(torch_out - onnx_out)))
