#!/usr/bin/env python3
"""Core count model and participant-level heads for TB-STATE-SC V3."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def zinb_log_prob(
    counts: torch.Tensor,
    mean: torch.Tensor,
    inverse_dispersion: torch.Tensor,
    dropout_logits: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Elementwise log probability under a mean/dispersion ZINB model."""
    x = counts
    mu = mean.clamp_min(eps)
    theta = inverse_dispersion.clamp_min(eps)
    log_theta_mu = torch.log(theta + mu)
    log_nb = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - log_theta_mu)
        + x * (torch.log(mu) - log_theta_mu)
    )
    log_nb_zero = theta * (torch.log(theta) - log_theta_mu)
    log_dropout = F.logsigmoid(dropout_logits)
    log_not_dropout = F.logsigmoid(-dropout_logits)
    zero_log_prob = torch.logaddexp(log_dropout, log_not_dropout + log_nb_zero)
    nonzero_log_prob = log_not_dropout + log_nb
    return torch.where(x <= eps, zero_log_prob, nonzero_log_prob)


def nb_log_prob(
    counts: torch.Tensor,
    mean: torch.Tensor,
    inverse_dispersion: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    x = counts
    mu = mean.clamp_min(eps)
    theta = inverse_dispersion.clamp_min(eps)
    log_theta_mu = torch.log(theta + mu)
    return (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - log_theta_mu)
        + x * (torch.log(mu) - log_theta_mu)
    )


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL(q||N(0,I)) across cells, summed across latent dimensions."""
    return -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()


def cross_covariance_penalty(z_id: torch.Tensor, z_state: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius norm of the empirical cross-covariance matrix."""
    if z_id.shape[0] < 2:
        return z_id.new_zeros(())
    id_centered = z_id - z_id.mean(dim=0, keepdim=True)
    state_centered = z_state - z_state.mean(dim=0, keepdim=True)
    covariance = id_centered.T @ state_centered / (z_id.shape[0] - 1)
    return covariance.square().sum()


def participant_pool(
    scores: torch.Tensor,
    participant_index: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Participant mean and smooth upper-tail pooling with equal participant units."""
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    unique_participants = torch.unique(participant_index, sorted=True)
    means: list[torch.Tensor] = []
    tails: list[torch.Tensor] = []
    for participant in unique_participants:
        values = scores[participant_index == participant]
        means.append(values.mean())
        tails.append(tau * (torch.logsumexp(values / tau, dim=0) - math.log(values.numel())))
    return unique_participants, torch.stack(means), torch.stack(tails)


class MonotoneMILHead(nn.Module):
    """Linear cell score plus monotone participant mean/tail aggregation."""

    def __init__(self, state_dim: int, tau: float) -> None:
        super().__init__()
        self.cell_score = nn.Linear(state_dim, 1)
        self.participant_bias = nn.Parameter(torch.zeros(()))
        self.raw_mean_weight = nn.Parameter(torch.zeros(()))
        self.raw_tail_weight = nn.Parameter(torch.zeros(()))
        self.tau = float(tau)

    def forward(
        self, z_state: torch.Tensor, participant_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eta = self.cell_score(z_state).squeeze(-1)
        participants, mean_score, tail_score = participant_pool(eta, participant_index, self.tau)
        logits = (
            self.participant_bias
            + F.softplus(self.raw_mean_weight) * mean_score
            + F.softplus(self.raw_tail_weight) * tail_score
        )
        return participants, logits, eta


class OrdinalMILHead(nn.Module):
    """Participant-level CORAL-style auxiliary head with ordered cut points."""

    def __init__(self, state_dim: int, n_thresholds: int, tau: float) -> None:
        super().__init__()
        self.cell_score = nn.Linear(state_dim, 1)
        self.raw_mean_weight = nn.Parameter(torch.zeros(()))
        self.raw_tail_weight = nn.Parameter(torch.zeros(()))
        self.base_threshold = nn.Parameter(torch.zeros(()))
        self.threshold_steps = nn.Parameter(torch.zeros(max(0, n_thresholds - 1)))
        self.n_thresholds = int(n_thresholds)
        self.tau = float(tau)

    def ordered_thresholds(self) -> torch.Tensor:
        if self.n_thresholds == 1:
            return self.base_threshold.reshape(1)
        steps = F.softplus(self.threshold_steps)
        return torch.cat(
            [self.base_threshold.reshape(1), self.base_threshold + torch.cumsum(steps, dim=0)]
        )

    def forward(
        self, z_state: torch.Tensor, participant_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eta = self.cell_score(z_state).squeeze(-1)
        participants, mean_score, tail_score = participant_pool(eta, participant_index, self.tau)
        severity = (
            F.softplus(self.raw_mean_weight) * mean_score
            + F.softplus(self.raw_tail_weight) * tail_score
        )
        logits = severity[:, None] - self.ordered_thresholds()[None, :]
        return participants, logits


class TBStateSCV3(nn.Module):
    """Count-aware identity/state VAE with identity-conditioned state modulation."""

    def __init__(
        self,
        n_genes: int,
        n_major_celltypes: int,
        n_technical_batches: int,
        transition_names: Iterable[str],
        state_dim: int,
        tau: float,
        identity_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.10,
        technical_embedding_dim: int = 8,
        likelihood: str = "zinb",
        n_ordinal_thresholds: int = 2,
    ) -> None:
        super().__init__()
        if likelihood not in {"zinb", "nb"}:
            raise ValueError("likelihood must be 'zinb' or 'nb'")
        self.n_genes = int(n_genes)
        self.identity_dim = int(identity_dim)
        self.state_dim = int(state_dim)
        self.likelihood = likelihood
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.id_mu = nn.Linear(hidden_dim, identity_dim)
        self.id_logvar = nn.Linear(hidden_dim, identity_dim)
        self.state_mu = nn.Linear(hidden_dim, state_dim)
        self.state_logvar = nn.Linear(hidden_dim, state_dim)

        self.identity_hidden = nn.Sequential(nn.Linear(identity_dim, hidden_dim), nn.GELU())
        self.state_gamma = nn.Linear(state_dim, hidden_dim)
        self.state_beta = nn.Linear(state_dim, hidden_dim)
        self.technical_embedding = nn.Embedding(n_technical_batches, technical_embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + technical_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gene_logits = nn.Linear(hidden_dim, n_genes)
        self.dropout_logits = nn.Linear(hidden_dim, n_genes) if likelihood == "zinb" else None
        self.raw_inverse_dispersion = nn.Parameter(torch.zeros(n_genes))
        self.identity_head = nn.Linear(identity_dim, n_major_celltypes)
        self.transition_heads = nn.ModuleDict(
            {name: MonotoneMILHead(state_dim, tau) for name in transition_names}
        )
        self.ordinal_head = OrdinalMILHead(state_dim, n_ordinal_thresholds, tau)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def encode(self, encoder_input: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(encoder_input)
        id_mu = self.id_mu(hidden)
        id_logvar = self.id_logvar(hidden).clamp(-12.0, 8.0)
        state_mu = self.state_mu(hidden)
        state_logvar = self.state_logvar(hidden).clamp(-12.0, 8.0)
        return {
            "id_mu": id_mu,
            "id_logvar": id_logvar,
            "state_mu": state_mu,
            "state_logvar": state_logvar,
            "z_id": self.reparameterize(id_mu, id_logvar),
            "z_state": self.reparameterize(state_mu, state_logvar),
        }

    def decode(
        self,
        z_id: torch.Tensor,
        z_state: torch.Tensor,
        technical_batch: torch.Tensor,
        library_size: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        identity = self.identity_hidden(z_id)
        gamma = 0.5 * torch.tanh(self.state_gamma(z_state))
        beta = self.state_beta(z_state)
        modulated = identity * (1.0 + gamma) + beta
        technical = self.technical_embedding(technical_batch)
        hidden = self.decoder(torch.cat([modulated, technical], dim=1))
        proportions = F.softmax(self.gene_logits(hidden), dim=1)
        mean = proportions * library_size.reshape(-1, 1).clamp_min(1.0)
        inverse_dispersion = F.softplus(self.raw_inverse_dispersion).clamp_min(1e-4)
        output = {"mean": mean, "inverse_dispersion": inverse_dispersion}
        if self.dropout_logits is not None:
            output["dropout_logits"] = self.dropout_logits(hidden)
        return output

    def forward(
        self,
        encoder_input: torch.Tensor,
        technical_batch: torch.Tensor,
        library_size: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent = self.encode(encoder_input)
        decoded = self.decode(
            latent["z_id"], latent["z_state"], technical_batch, library_size
        )
        return {
            **latent,
            **decoded,
            "identity_logits": self.identity_head(latent["z_id"]),
        }

    def reconstruction_loss(self, counts: torch.Tensor, output: dict[str, torch.Tensor]) -> torch.Tensor:
        counts_fp32 = counts.float()
        mean_fp32 = output["mean"].float()
        dispersion_fp32 = output["inverse_dispersion"].float()
        if self.likelihood == "zinb":
            log_prob = zinb_log_prob(
                counts_fp32,
                mean_fp32,
                dispersion_fp32,
                output["dropout_logits"].float(),
            )
        else:
            log_prob = nb_log_prob(counts_fp32, mean_fp32, dispersion_fp32)
        return -log_prob.sum(dim=1).mean()


class VanillaCountVAE(nn.Module):
    """Prespecified 24D vanilla count-VAE representation baseline."""

    def __init__(
        self,
        n_genes: int,
        n_technical_batches: int = 1,
        latent_dim: int = 24,
        hidden_dim: int = 256,
        dropout: float = 0.10,
        technical_embedding_dim: int = 8,
        likelihood: str = "zinb",
    ) -> None:
        super().__init__()
        if likelihood not in {"zinb", "nb"}:
            raise ValueError("likelihood must be 'zinb' or 'nb'")
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.likelihood = likelihood
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.technical_embedding = nn.Embedding(n_technical_batches, technical_embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + technical_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gene_logits = nn.Linear(hidden_dim, n_genes)
        self.dropout_logits = nn.Linear(hidden_dim, n_genes) if likelihood == "zinb" else None
        self.raw_inverse_dispersion = nn.Parameter(torch.zeros(n_genes))

    def encode(self, encoder_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(encoder_input)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(-12.0, 8.0)
        if torch.is_grad_enabled():
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu
        return mu, logvar, z

    def forward(
        self,
        encoder_input: torch.Tensor,
        technical_batch: torch.Tensor,
        library_size: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mu, logvar, z = self.encode(encoder_input)
        hidden = self.decoder(torch.cat([z, self.technical_embedding(technical_batch)], dim=1))
        mean = F.softmax(self.gene_logits(hidden), dim=1) * library_size[:, None].clamp_min(1.0)
        output = {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "mean": mean,
            "inverse_dispersion": F.softplus(self.raw_inverse_dispersion).clamp_min(1e-4),
        }
        if self.dropout_logits is not None:
            output["dropout_logits"] = self.dropout_logits(hidden)
        return output

    def reconstruction_loss(self, counts: torch.Tensor, output: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.likelihood == "zinb":
            log_prob = zinb_log_prob(
                counts.float(),
                output["mean"].float(),
                output["inverse_dispersion"].float(),
                output["dropout_logits"].float(),
            )
        else:
            log_prob = nb_log_prob(
                counts.float(),
                output["mean"].float(),
                output["inverse_dispersion"].float(),
            )
        return -log_prob.sum(dim=1).mean()

