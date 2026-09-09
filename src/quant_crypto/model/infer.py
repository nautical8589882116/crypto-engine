"""ONNX Runtime inference for the Mamba classifier.

Loads a trained ONNX model, runs CPU inference on a feature window, and emits
a tradeable Signal when P(velocity > threshold) clears the configured gate.
Supports atomic hot-swap of the model handle for MCP-driven weight updates.
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from quant_crypto.schemas.signal import Signal


class MambaInference:
    """ONNX Runtime wrapper with atomic hot-swap and threshold gating."""

    def __init__(
        self,
        onnx_path: str,
        seq_len: int = 300,
        n_features: int = 3,
        threshold: float = 0.85,
    ):
        self.onnx_path = onnx_path
        self.seq_len = seq_len
        self.n_features = n_features
        self.threshold = threshold
        self.sha256: str = ""
        self._lock = threading.Lock()
        self._session: ort.InferenceSession = self._load_session(onnx_path)

    @staticmethod
    def _load_session(path: str) -> ort.InferenceSession:
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def predict(self, window: np.ndarray) -> float:
        """Run inference on a (seq_len, n_features) window; returns P(velocity>threshold)."""
        if window.shape != (self.seq_len, self.n_features):
            raise ValueError(
                f"Expected window shape ({self.seq_len}, {self.n_features}), got {window.shape}"
            )
        x = window.astype("float32")[np.newaxis, ...]
        with self._lock:
            out = self._session.run(None, {"input": x})[0]
        return float(out.reshape(-1)[0])

    def emit_signal(
        self,
        window: np.ndarray,
        regime: str,
        direction: str = "flat",
        confidence: float = 1.0,
    ) -> Signal:
        prob = self.predict(window)
        if direction == "flat":
            # No direction -> never tradeable, but still report the probability.
            pass
        return Signal(
            regime=regime,  # type: ignore[arg-type]
            direction=direction,
            prob_velocity=prob,
            threshold=self.threshold,
            confidence=confidence,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def hot_swap(self, onnx_path: str) -> None:
        """Atomically replace the model handle (thread-safe).

        The candidate file must load as a valid ONNX graph AND produce a
        finite forward pass before it replaces the live session — a corrupt
        or truncated artifact never touches the running handle. The swapped
        artifact's sha256 is recorded for auditability.
        """
        new_session = self._load_session(onnx_path)  # raises if not a valid graph
        # Integrity smoke: one forward pass must stay finite.
        probe = np.zeros((self.seq_len, self.n_features), dtype="float32")
        out = new_session.run(None, {"input": probe[np.newaxis, ...]})[0]
        if not np.isfinite(float(np.asarray(out).reshape(-1)[0])):
            raise ValueError(f"hot-swap rejected: non-finite output from {onnx_path}")
        digest = hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()
        with self._lock:
            self._session = new_session
            self.onnx_path = onnx_path
            self.sha256 = digest

    def benchmark(self, window: np.ndarray, n_runs: int = 100) -> float:
        """Return average inference latency in milliseconds."""
        self.predict(window)  # warm-up
        start = time.perf_counter()
        for _ in range(n_runs):
            self.predict(window)
        elapsed = time.perf_counter() - start
        return (elapsed / n_runs) * 1000.0
