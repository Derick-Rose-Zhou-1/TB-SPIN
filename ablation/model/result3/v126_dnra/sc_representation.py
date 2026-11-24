"""Single-cell VAE construction for isolated de novo V126 runs."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


DEFAULT_CANONICAL_SOURCE = Path(__file__).resolve().parents[1] / "tbstate_v12_models.py"


def _canonical_module(source: Path = DEFAULT_CANONICAL_SOURCE):
    spec = importlib.util.spec_from_file_location("v126_dnra_canonical", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"CANONICAL_MODEL_IMPORT_FAILED:{source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_sc_model(
    n_genes: int,
    latent_dim: int = 32,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    stage_loss_weight: float = 25.0,
):
    canonical = _canonical_module()
    config = canonical.SingleCellVAEConfig(
        n_genes=n_genes,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=2,
        dropout=dropout,
        likelihood="zinb",
        stage_loss_weight=float(stage_loss_weight),
        kl_max_weight=1.0,
        kl_warmup_epochs=4,
    )
    return canonical.SingleCellOrdinalVAE(config)


def loss_weight_for_variant(variant: str, full_weight: float) -> float:
    if variant == "NO_ORDINAL_SINGLE_CELL_SUPERVISION":
        return 0.0
    if variant == "FULL":
        return float(full_weight)
    raise ValueError(f"UNKNOWN_SC_VARIANT:{variant}")


def train_minibatch(model, counts: torch.Tensor, stages: torch.Tensor, optimizer, kl_weight: float) -> dict[str, float | bool]:
    canonical = _canonical_module()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(counts)
    loss, terms = canonical.count_vae_loss(model, counts, stages, output, kl_weight=kl_weight)
    if not torch.isfinite(loss):
        raise RuntimeError("NONFINITE_SC_LOSS")
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
    if not torch.isfinite(torch.tensor(gradient_norm)):
        raise RuntimeError("NONFINITE_SC_GRADIENT")
    optimizer.step()
    return {
        "total_loss": float(terms["total_loss"]),
        "reconstruction_loss": float(terms["reconstruction_loss"]),
        "kl_loss": float(terms["kl_loss"]),
        "stage_loss": float(terms["stage_loss"]),
        "gradient_norm": gradient_norm,
        "finite": True,
    }
