"""Training loop for the Mamba classifier.

Trains on synthetic and/or historical tick data to predict whether velocity
over the next K ticks exceeds a threshold. The overfit test proves model
capacity (loss -> ~0 on a small deterministic dataset). Stage B adds the
real-corpus path: numpy windows from quant_crypto.data.dataset -> train/val split.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from quant_crypto.model.mamba_model import MambaClassifier

# Tiny per-op matmuls thrash under OpenMP's default (48 threads here) —
# cap it so CPU training runs at sane speed.
torch.set_num_threads(min(8, max(1, (torch.get_num_threads() or 8))))


def split_windows(
    x: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically split windowed samples into train/val by row index."""
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def train_on_windows(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 100,
    lr: float = 3e-3,
    val_fraction: float = 0.2,
    seed: int = 0,
    model_kwargs: dict | None = None,
) -> dict:
    """Train a Mamba classifier on numpy windows from the corpus builder.

    x: (n, seq_len, n_features) float32. Returns the trained model plus final
    train/val BCE loss. Deterministic for a given seed.
    """
    mk = {"d_model": 32, "d_state": 16, "n_layers": 2}
    if model_kwargs:
        mk.update(model_kwargs)
    torch.manual_seed(seed)
    model = MambaClassifier(n_features=x.shape[2], **mk)

    # Accept either numpy or torch inputs.
    if torch.is_tensor(x):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)
    if torch.is_tensor(y):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.asarray(y)

    x_tr, y_tr, x_va, y_va = split_windows(x_np, y_np, val_fraction=val_fraction, seed=seed)
    x_t = torch.from_numpy(x_tr)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    x_v = torch.from_numpy(x_va)
    y_v = torch.from_numpy(y_va.astype(np.float32))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    final_loss = float("nan")
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
        final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        val_loss = float(loss_fn(model(x_v), y_v).item())

    return {"model": model, "loss": final_loss, "val_loss": val_loss}


def make_synthetic_dataset(
    n_samples: int = 64,
    seq_len: int = 300,
    n_features: int = 3,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a small, learnable synthetic dataset.

    Label = 1 when the mean OFI (feature 0) over the window is positive,
    i.e. the 'velocity' signal is deterministic in the input.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, seq_len, n_features, generator=g)
    # Make the label a function of feature 0 so it's learnable.
    ofi_mean = x[:, :, 0].mean(dim=1)
    y = (ofi_mean > 0).float()
    return x, y


def train(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 300,
    lr: float = 3e-3,
) -> float:
    """Train to overfit a small dataset; returns the final BCE loss."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
    return float(loss.item())


def save_checkpoint(model: nn.Module, path: str) -> None:
    torch.save({"state_dict": model.state_dict()}, path)


def load_checkpoint(model: nn.Module, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
