"""Mamba (Selective State Space Model) for sequence classification.

A compact, dependency-free SSM that ingests a (batch, seq_len, n_features)
window and outputs P(velocity > threshold). Implemented in pure PyTorch so it
exports cleanly to ONNX for CPU inference.

The selective scan (S6) is implemented as a sequential recurrence over the
sequence dimension — simple, correct, and ONNX-traceable (as opposed to the
parallel-scan formulation which uses operators ONNX does not support).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SelectiveScan(nn.Module):
    """Discretized selective SSM core (Mamba S6)."""

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        dt_rank = dt_rank or d_model

        # Projects input -> (delta, B, C)
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        # A is stored in log-space and kept negative for stable decay.
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape
        A = -torch.exp(self.A_log)  # (d_model, d_state), negative -> decay

        x_proj = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*d_state)
        delta, B, C = x_proj.split(
            [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1
        )
        delta = self.dt_proj(delta)  # (batch, seq_len, d_model)
        delta = torch.nn.functional.softplus(delta)

        # B, C: (batch, seq_len, d_state)
        # Discretize: A_bar = exp(delta * A), B_bar = delta * B
        A_bar = torch.exp(delta.unsqueeze(-1) * A)  # (batch, seq_len, d_model, d_state)
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seq_len, d_model, d_state)

        # Sequential scan
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            y_t = (C[:, t].unsqueeze(1) * h).sum(dim=-1)  # (batch, d_model)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        return y + x * self.D.unsqueeze(0).unsqueeze(0)


class MambaBlock(nn.Module):
    """A single Mamba block with residual connection and layer norm."""

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveScan(d_model, d_state)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.out_proj(x)
        return x + residual


class MambaClassifier(nn.Module):
    """Sequence classifier: (batch, seq_len, n_features) -> P(velocity > threshold)."""

    def __init__(self, n_features: int = 3, d_model: int = 32, d_state: int = 16, n_layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(n_features, d_model, bias=False)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # pool over time
        logits = self.head(x).squeeze(-1)
        return torch.sigmoid(logits)
