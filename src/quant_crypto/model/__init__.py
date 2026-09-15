"""Crypto model package.

`MambaInference` (ONNX Runtime) has no torch dependency and is what the runtime
engine worker imports. `MambaClassifier` (torch) is only needed for TRAINING, so
it is exposed lazily — importing it here eagerly would drag torch into the lean
runtime image and break the engine worker with ModuleNotFoundError.
"""

from quant_crypto.model.infer import MambaInference

__all__ = ["MambaClassifier", "MambaInference"]


def __getattr__(name: str):
    if name == "MambaClassifier":
        from quant_crypto.model.mamba_model import MambaClassifier

        return MambaClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
