"""Crypto model package."""

from quant_crypto.model.mamba_model import MambaClassifier
from quant_crypto.model.infer import MambaInference

__all__ = ["MambaClassifier", "MambaInference"]
