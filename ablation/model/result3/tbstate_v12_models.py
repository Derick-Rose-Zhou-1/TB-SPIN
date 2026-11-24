"""PyTorch model components shared by the TB-STATE V12 pipeline."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class SingleCellVAEConfig:
    n_genes: int
    latent_dim: int = 32
    hidden_dim: int = 256
    n_hidden_layers: int = 2
    dropout: float = 0.1
    likelihood: str = "zinb"
    stage_loss_weight: float = 25.0
    kl_max_weight: float = 1.0
    kl_warmup_epochs: int = 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SingleCellOrdinalVAE(nn.Module):
    """Count VAE with ZINB/NB likelihood and an ordinal HC/LTBI/ATB head."""

    def __init__(self, config: SingleCellVAEConfig):
        super().__init__()
        if config.likelihood not in {"zinb", "nb"}:
            raise ValueError(f"Unsupported likelihood: {config.likelihood}")
        self.config = config
        encoder_layers: list[nn.Module] = []
        input_dim = config.n_genes
        for _ in range(config.n_hidden_layers):
            encoder_layers.extend(
                [
                    nn.Linear(input_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            input_dim = config.hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.z_mean = nn.Linear(config.hidden_dim, config.latent_dim)
        self.z_logvar = nn.Linear(config.hidden_dim, config.latent_dim)

        decoder_layers: list[nn.Module] = []
        input_dim = config.latent_dim
        for _ in range(config.n_hidden_layers):
            decoder_layers.extend(
                [
                    nn.Linear(input_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            input_dim = config.hidden_dim
        self.decoder = nn.Sequential(*decoder_layers)
        self.scale_logits = nn.Linear(config.hidden_dim, config.n_genes)
        self.dropout_logits = (
            nn.Linear(config.hidden_dim, config.n_genes)
            if config.likelihood == "zinb"
            else None
        )
        self.log_inverse_dispersion = nn.Parameter(
            torch.zeros(config.n_genes, dtype=torch.float32)
        )

        self.stage_score = nn.Linear(config.latent_dim, 1)
        self.stage_threshold_base = nn.Parameter(torch.tensor(-0.5))
        self.stage_threshold_gap_raw = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def encoder_input(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        library = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        normalized = torch.log1p(counts / library * 1.0e4)
        return normalized, library

    def encode(self, counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized, _ = self.encoder_input(counts)
        hidden = self.encoder(normalized)
        return self.z_mean(hidden), self.z_logvar(hidden).clamp(-12.0, 12.0)

    def reparameterize(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if not self.training:
            return mean
        standard_deviation = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(standard_deviation) * standard_deviation

    def ordinal_logits(self, latent: torch.Tensor) -> torch.Tensor:
        score = self.stage_score(latent)
        threshold_1 = self.stage_threshold_base
        threshold_2 = threshold_1 + F.softplus(self.stage_threshold_gap_raw)
        return torch.cat([score - threshold_1, score - threshold_2], dim=1)

    def forward(self, counts: torch.Tensor) -> dict[str, torch.Tensor | None]:
        normalized, library = self.encoder_input(counts)
        hidden = self.encoder(normalized)
        mean = self.z_mean(hidden)
        logvar = self.z_logvar(hidden).clamp(-12.0, 12.0)
        latent = self.reparameterize(mean, logvar)
        decoded = self.decoder(latent)
        proportions = torch.softmax(self.scale_logits(decoded), dim=1)
        rate = proportions * library
        dropout_logits = (
            self.dropout_logits(decoded)
            if self.dropout_logits is not None
            else None
        )
        return {
            "z_mean": mean,
            "z_logvar": logvar,
            "z": latent,
            "rate": rate,
            "dropout_logits": dropout_logits,
            "ordinal_logits": self.ordinal_logits(latent),
        }


def negative_binomial_log_probability(
    counts: torch.Tensor,
    mean: torch.Tensor,
    inverse_dispersion: torch.Tensor,
) -> torch.Tensor:
    counts = counts.float()
    mean = mean.float().clamp_min(1.0e-8)
    theta = inverse_dispersion.float().clamp_min(1.0e-4)
    log_theta_mu = torch.log(theta + mean)
    return (
        torch.lgamma(counts + theta)
        - torch.lgamma(theta)
        - torch.lgamma(counts + 1.0)
        + theta * (torch.log(theta) - log_theta_mu)
        + counts * (torch.log(mean) - log_theta_mu)
    )


def count_vae_loss(
    model: SingleCellOrdinalVAE,
    counts: torch.Tensor,
    stages: torch.Tensor,
    output: dict[str, torch.Tensor | None],
    kl_weight: float,
    stage_positive_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rate = output["rate"]
    mean = output["z_mean"]
    logvar = output["z_logvar"]
    ordinal_logits = output["ordinal_logits"]
    if not isinstance(rate, torch.Tensor):
        raise TypeError("Missing rate tensor.")
    if not isinstance(mean, torch.Tensor) or not isinstance(logvar, torch.Tensor):
        raise TypeError("Missing latent tensors.")
    if not isinstance(ordinal_logits, torch.Tensor):
        raise TypeError("Missing ordinal logits.")
    theta = F.softplus(model.log_inverse_dispersion) + 1.0e-4
    nb_log_probability = negative_binomial_log_probability(counts, rate, theta)
    if model.config.likelihood == "zinb":
        dropout_logits = output["dropout_logits"]
        if not isinstance(dropout_logits, torch.Tensor):
            raise TypeError("ZINB model lacks dropout logits.")
        dropout_probability = torch.sigmoid(dropout_logits.float())
        zero_nb_probability = torch.exp(
            negative_binomial_log_probability(
                torch.zeros_like(counts), rate, theta
            )
        )
        zero_log_probability = torch.log(
            dropout_probability
            + (1.0 - dropout_probability) * zero_nb_probability
            + 1.0e-8
        )
        nonzero_log_probability = (
            F.logsigmoid(-dropout_logits.float()) + nb_log_probability
        )
        count_log_probability = torch.where(
            counts <= 0.0, zero_log_probability, nonzero_log_probability
        )
    else:
        count_log_probability = nb_log_probability
    reconstruction = -count_log_probability.sum(dim=1).mean()
    kl = -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).sum(dim=1).mean()
    ordinal_targets = torch.stack(
        [(stages >= 1).float(), (stages >= 2).float()], dim=1
    )
    stage_loss = F.binary_cross_entropy_with_logits(
        ordinal_logits.float(),
        ordinal_targets,
        pos_weight=stage_positive_weights,
    )
    total = (
        reconstruction
        + float(kl_weight) * kl
        + model.config.stage_loss_weight * stage_loss
    )
    return total, {
        "total_loss": total.detach(),
        "reconstruction_loss": reconstruction.detach(),
        "kl_loss": kl.detach(),
        "stage_loss": stage_loss.detach(),
    }


def cox_efron_negative_partial_log_likelihood(
    risk_score: torch.Tensor,
    duration: torch.Tensor,
    event: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Breslow partial likelihood, normalized by weighted event count."""
    order = torch.argsort(duration, descending=True)
    risk = risk_score[order].reshape(-1)
    observed_event = event[order].float().reshape(-1)
    if sample_weight is None:
        weight = torch.ones_like(observed_event)
    else:
        weight = sample_weight[order].float().reshape(-1)
    log_cumulative_risk = torch.logcumsumexp(risk, dim=0)
    contributions = (risk - log_cumulative_risk) * observed_event * weight
    denominator = (observed_event * weight).sum().clamp_min(1.0)
    return -contributions.sum() / denominator


class BulkAdapterEncoder(nn.Module):
    """Bulk input adapter feeding the single-cell encoder's shared core."""

    def __init__(
        self,
        n_genes: int,
        sc_hidden_dim: int,
        sc_latent_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gene_scale = nn.Parameter(torch.ones(n_genes))
        self.gene_shift = nn.Parameter(torch.zeros(n_genes))
        self.input_projection = nn.Linear(n_genes, sc_hidden_dim)
        self.shared_core = nn.Sequential(
            nn.LayerNorm(sc_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sc_hidden_dim, sc_hidden_dim),
            nn.LayerNorm(sc_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.latent_projection = nn.Linear(sc_hidden_dim, sc_latent_dim)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        adapted = expression * self.gene_scale + self.gene_shift
        hidden = self.input_projection(adapted)
        hidden = self.shared_core(hidden)
        return self.latent_projection(hidden)


class PseudobulkAlignmentModel(nn.Module):
    """Two domain-specific gene adapters with one single-cell-initialized core."""

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.sc_gene_scale = nn.Parameter(torch.ones(n_genes))
        self.sc_gene_shift = nn.Parameter(torch.zeros(n_genes))
        self.bulk_gene_scale = nn.Parameter(torch.ones(n_genes))
        self.bulk_gene_shift = nn.Parameter(torch.zeros(n_genes))
        self.input_projection = nn.Linear(n_genes, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.shared_linear = nn.Linear(hidden_dim, hidden_dim)
        self.shared_norm = nn.LayerNorm(hidden_dim)
        self.latent_projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, expression: torch.Tensor, domain: str) -> torch.Tensor:
        if domain == "single_cell_pseudobulk":
            adapted = expression * self.sc_gene_scale + self.sc_gene_shift
        elif domain == "bulk":
            adapted = expression * self.bulk_gene_scale + self.bulk_gene_shift
        else:
            raise ValueError(f"Unknown alignment domain: {domain}")
        hidden = F.gelu(self.input_norm(self.input_projection(adapted)))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = F.gelu(self.shared_norm(self.shared_linear(hidden)))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.latent_projection(hidden)

    def initialize_from_single_cell(
        self, single_cell_model: SingleCellOrdinalVAE
    ) -> None:
        if single_cell_model.config.n_hidden_layers != 2:
            raise ValueError("V12 alignment expects the locked two-layer SC encoder.")
        with torch.no_grad():
            self.input_projection.weight.copy_(
                single_cell_model.encoder[0].weight
            )
            self.input_projection.bias.copy_(single_cell_model.encoder[0].bias)
            self.input_norm.weight.copy_(single_cell_model.encoder[1].weight)
            self.input_norm.bias.copy_(single_cell_model.encoder[1].bias)
            self.shared_linear.weight.copy_(single_cell_model.encoder[4].weight)
            self.shared_linear.bias.copy_(single_cell_model.encoder[4].bias)
            self.shared_norm.weight.copy_(single_cell_model.encoder[5].weight)
            self.shared_norm.bias.copy_(single_cell_model.encoder[5].bias)
            self.latent_projection.weight.copy_(single_cell_model.z_mean.weight)
            self.latent_projection.bias.copy_(single_cell_model.z_mean.bias)


class MultiEndpointRiskHead(nn.Module):
    """Low-complexity masked logistic and Cox heads over a shared latent."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim > 0:
            self.body = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_input = hidden_dim
        else:
            self.body = nn.Identity()
            head_input = latent_dim
        self.logistic_heads = nn.ModuleDict(
            {endpoint: nn.Linear(head_input, 1) for endpoint in ("H1", "H2", "H3")}
        )
        self.cox_head = nn.Linear(head_input, 1)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.body(latent)
        result = {
            endpoint: head(hidden).reshape(-1)
            for endpoint, head in self.logistic_heads.items()
        }
        result["TIME_TO_TB"] = self.cox_head(hidden).reshape(-1)
        return result


class FeatureRiskModel(nn.Module):
    """Multitask risk model with optional near-identity latent fine-tuning."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 0,
        dropout: float = 0.1,
        finetune_last_layer: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.finetune_last_layer = finetune_last_layer
        if finetune_last_layer:
            self.feature_transform = nn.Linear(n_features, n_features)
            with torch.no_grad():
                self.feature_transform.weight.copy_(torch.eye(n_features))
                self.feature_transform.bias.zero_()
        else:
            self.feature_transform = nn.Identity()
        self.head = MultiEndpointRiskHead(
            latent_dim=n_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        transformed = self.feature_transform(features)
        return self.head(transformed)

    def identity_penalty(self) -> torch.Tensor:
        if not self.finetune_last_layer:
            parameter = next(self.parameters())
            return parameter.new_tensor(0.0)
        if not isinstance(self.feature_transform, nn.Linear):
            raise TypeError("Expected a linear feature transform.")
        identity = torch.eye(
            self.n_features,
            device=self.feature_transform.weight.device,
            dtype=self.feature_transform.weight.dtype,
        )
        return (
            (self.feature_transform.weight - identity).square().mean()
            + self.feature_transform.bias.square().mean()
        )
