#!/usr/bin/env python3
"""Training and evaluation utilities for participant-supervised TB-STATE-SC V3."""

from __future__ import annotations

import copy
import gc
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.special import logsumexp
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F

from tbstate_sc_v3_model import (
    TBStateSCV3,
    VanillaCountVAE,
    cross_covariance_penalty,
    gaussian_kl,
)


R2_H5AD = Path("data/train_new.h5ad")
TASKS = {
    "T1": {"negative": "HC", "positive": "TBINF"},
    "T2": {"negative": "TBINF", "positive": "ATB"},
}
STAGE_ORDINAL = {"HC": 0, "TBINF": 1, "ATB": 2}


@dataclass(frozen=True)
class TrainConfig:
    state_dim: int
    tau: float
    lambda_cross: float
    lambda_id: float
    beta_kl: float
    lr_finetune_encoder: float
    likelihood: str = "zinb"
    identity_dim: int = 16
    hidden_dim: int = 256
    lambda_ordinal: float = 0.15
    lr_pretrain: float = 5e-4
    lr_heads: float = 5e-4
    weight_decay: float = 1e-4
    max_pretrain_epochs: int = 100
    max_finetune_epochs: int = 150
    participants_per_batch: int = 8
    cells_per_participant: int = 512
    gradient_clip: float = 5.0
    eval_every: int = 5
    early_stop_checks: int = 4


@dataclass
class PreparedData:
    counts: sparse.csr_matrix
    metadata: pd.DataFrame
    gene_ids: list[str]
    participant_col: str
    participant_to_cells: dict[str, np.ndarray]
    participant_to_stage: dict[str, str]
    celltype_codes: np.ndarray
    celltype_levels: list[str]
    identity_weights: np.ndarray


def dataframe_index(group: h5py.Group) -> pd.Series:
    name = group.attrs["_index"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    values = group[str(name)][()]
    return pd.Series(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def load_csr(group: h5py.Group) -> sparse.csr_matrix:
    shape = tuple(int(value) for value in group.attrs["shape"])
    return sparse.csr_matrix(
        (
            np.asarray(group["data"][()]),
            np.asarray(group["indices"][()]),
            np.asarray(group["indptr"][()]),
        ),
        shape=shape,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_result2_fold(root: Path, outer_fold: int) -> PreparedData:
    metadata = pd.read_csv(
        root / "02_INPUT_AUDIT/MODEL_READY_METADATA.tsv.gz", sep="\t", dtype=str
    )
    forbidden = [
        column
        for column in metadata.columns
        if any(
            token in column.lower()
            for token in ("celltype_fine", "fine_celltype", "celltype_refined", "leiden", "cluster")
        )
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden annotation columns reached training: {forbidden}")
    hvg = pd.read_csv(
        root / f"05_FEATURES/FOLD_{outer_fold}_HVG5000.tsv",
        sep="\t",
        dtype={"gene_id": str},
    ).sort_values("feature_rank")
    selected_genes = hvg["gene_id"].astype(str).tolist()
    all_genes = pd.read_csv(
        root / "02_INPUT_AUDIT/GENE_MANIFEST.tsv.gz", sep="\t", dtype=str
    )["gene_id"].astype(str)
    gene_position = {gene: index for index, gene in enumerate(all_genes)}
    selected_positions = np.asarray([gene_position[gene] for gene in selected_genes], dtype=int)
    with h5py.File(R2_H5AD, "r") as handle:
        h5ad_cell_ids = dataframe_index(handle["obs"]).astype(str).to_numpy()
        if not np.array_equal(h5ad_cell_ids, metadata["cell_id"].astype(str).to_numpy()):
            raise RuntimeError("Result2 metadata/H5AD row alignment failed")
        full_counts = load_csr(handle["layers"]["counts"])
    counts = full_counts[:, selected_positions].tocsr()
    del full_counts
    gc.collect()
    if counts.shape != (356_949, 5_000):
        raise RuntimeError(f"Unexpected fold matrix shape: {counts.shape}")
    if counts.data.size and not np.allclose(counts.data, np.rint(counts.data), atol=1e-6):
        raise RuntimeError("Non-integer values reached the count target")

    participant_col = "sample_id_std"
    participant_to_cells = {
        str(participant): np.asarray(index, dtype=np.int64)
        for participant, index in metadata.groupby(participant_col, observed=True).indices.items()
    }
    participant_to_stage = (
        metadata.groupby(participant_col, observed=True)["analysis_stage"].first().astype(str).to_dict()
    )
    celltype = pd.Categorical(metadata["celltype_raw"].astype(str))
    if (celltype.codes < 0).any():
        raise RuntimeError("Missing celltype_raw labels")
    frequency = np.bincount(celltype.codes, minlength=len(celltype.categories)).astype(float)
    inverse = frequency.sum() / np.maximum(frequency, 1.0)
    inverse /= inverse.mean()
    identity_weights = np.minimum(inverse, 5.0)
    identity_weights /= identity_weights.mean()
    return PreparedData(
        counts=counts,
        metadata=metadata,
        gene_ids=selected_genes,
        participant_col=participant_col,
        participant_to_cells=participant_to_cells,
        participant_to_stage=participant_to_stage,
        celltype_codes=celltype.codes.astype(np.int64),
        celltype_levels=[str(value) for value in celltype.categories],
        identity_weights=identity_weights.astype(np.float32),
    )


def participant_batches(
    data: PreparedData,
    participants: list[str],
    config: TrainConfig,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    order = np.asarray(participants, dtype=object)
    rng.shuffle(order)
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for start in range(0, len(order), config.participants_per_batch):
        selected = order[start : start + config.participants_per_batch]
        cell_parts: list[np.ndarray] = []
        local_parts: list[np.ndarray] = []
        stages: list[int] = []
        for local_code, participant in enumerate(selected):
            available = data.participant_to_cells[str(participant)]
            n_take = min(config.cells_per_participant, len(available))
            chosen = rng.choice(available, size=n_take, replace=False)
            cell_parts.append(np.asarray(chosen, dtype=np.int64))
            local_parts.append(np.full(n_take, local_code, dtype=np.int64))
            stages.append(STAGE_ORDINAL[data.participant_to_stage[str(participant)]])
        batches.append(
            (
                np.concatenate(cell_parts),
                np.concatenate(local_parts),
                np.asarray(stages, dtype=np.int64),
            )
        )
    return batches


def dense_batch(data: PreparedData, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    counts_np = data.counts[indices].toarray().astype(np.float32, copy=False)
    counts = torch.from_numpy(counts_np).to(device, non_blocking=True)
    library = counts.sum(dim=1).clamp_min(1.0)
    encoder_input = torch.log1p(counts / library[:, None] * 10_000.0)
    technical_batch = torch.zeros(len(indices), dtype=torch.long, device=device)
    celltype = torch.as_tensor(data.celltype_codes[indices], dtype=torch.long, device=device)
    return counts, library, encoder_input, technical_batch, celltype


def phase_loss(
    model: TBStateSCV3,
    data: PreparedData,
    indices: np.ndarray,
    local_participants: np.ndarray,
    local_stage_codes: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    beta_kl: float,
    supervised: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    counts, library, encoder_input, technical_batch, celltype = dense_batch(data, indices, device)
    participant_index = torch.as_tensor(local_participants, dtype=torch.long, device=device)
    stage_codes = torch.as_tensor(local_stage_codes, dtype=torch.long, device=device)
    output = model(encoder_input, technical_batch, library)
    reconstruction = model.reconstruction_loss(counts, output) / data.counts.shape[1]
    kl = gaussian_kl(output["id_mu"], output["id_logvar"]) + gaussian_kl(
        output["state_mu"], output["state_logvar"]
    )
    identity_weight = torch.as_tensor(data.identity_weights, dtype=torch.float32, device=device)
    identity = F.cross_entropy(output["identity_logits"], celltype, weight=identity_weight)
    crosscov = cross_covariance_penalty(output["z_id"], output["z_state"])
    loss = reconstruction + beta_kl * kl + config.lambda_id * identity
    loss = loss + config.lambda_cross * crosscov
    transition_loss = reconstruction.new_zeros(())
    ordinal_loss = reconstruction.new_zeros(())
    if supervised:
        for task, definition in TASKS.items():
            participants, logits, _ = model.transition_heads[task](
                output["z_state"], participant_index
            )
            participant_stage = stage_codes[participants]
            negative_code = STAGE_ORDINAL[definition["negative"]]
            positive_code = STAGE_ORDINAL[definition["positive"]]
            eligible = (participant_stage == negative_code) | (participant_stage == positive_code)
            if eligible.any():
                target = (participant_stage[eligible] == positive_code).float()
                transition_loss = transition_loss + F.binary_cross_entropy_with_logits(
                    logits[eligible], target
                )
        ordinal_participants, ordinal_logits = model.ordinal_head(
            output["z_state"], participant_index
        )
        ordinal_stage = stage_codes[ordinal_participants]
        ordinal_target = torch.stack(
            [(ordinal_stage > threshold).float() for threshold in range(2)], dim=1
        )
        ordinal_loss = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_target)
        loss = loss + transition_loss + config.lambda_ordinal * ordinal_loss
    components = {
        "loss": float(loss.detach().cpu()),
        "reconstruction_per_gene": float(reconstruction.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "identity": float(identity.detach().cpu()),
        "crosscov": float(crosscov.detach().cpu()),
        "transition": float(transition_loss.detach().cpu()),
        "ordinal": float(ordinal_loss.detach().cpu()),
    }
    return loss, components


def amp_safe_phase_loss(
    model: TBStateSCV3,
    data: PreparedData,
    indices: np.ndarray,
    local_participants: np.ndarray,
    local_stage_codes: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    beta_kl: float,
    supervised: bool,
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.cuda.amp.autocast(enabled=use_amp):
        loss, components = phase_loss(
            model,
            data,
            indices,
            local_participants,
            local_stage_codes,
            config,
            device,
            beta_kl,
            supervised,
        )
    if torch.isfinite(loss):
        components["amp_safe_fallback"] = 0.0
        return loss, components
    print("AMP_SAFE_FALLBACK=FP32_LOSS", flush=True)
    with torch.cuda.amp.autocast(enabled=False):
        loss, components = phase_loss(
            model,
            data,
            indices,
            local_participants,
            local_stage_codes,
            config,
            device,
            beta_kl,
            supervised,
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("FP32 loss remained NaN/Inf after AMP_SAFE_FALLBACK")
    components["amp_safe_fallback"] = 1.0
    return loss, components


def aggregate_numpy(scores: np.ndarray, participants: np.ndarray, tau: float) -> pd.DataFrame:
    frame = pd.DataFrame({"participant": participants.astype(str), "score": scores})
    rows = []
    for participant, sub in frame.groupby("participant", sort=True):
        values = sub["score"].to_numpy(float)
        rows.append(
            {
                "participant": participant,
                "mean_eta": float(values.mean()),
                "tail_eta": float(tau * (logsumexp(values / tau) - math.log(len(values)))),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_model(
    model: TBStateSCV3,
    data: PreparedData,
    participants: list[str],
    config: TrainConfig,
    device: torch.device,
    return_cell_scores: bool,
    chunk_size: int = 2048,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, dict[str, float]]:
    model.eval()
    selected_indices = np.concatenate([data.participant_to_cells[p] for p in participants])
    selected_indices.sort()
    eta = {task: [] for task in TASKS}
    id_prediction: list[np.ndarray] = []
    id_truth: list[np.ndarray] = []
    z_state_values: list[np.ndarray] = []
    reconstruction_sum = 0.0
    reconstruction_cells = 0
    for start in range(0, len(selected_indices), chunk_size):
        indices = selected_indices[start : start + chunk_size]
        counts, library, encoder_input, technical_batch, celltype = dense_batch(data, indices, device)
        output = model(encoder_input, technical_batch, library)
        reconstruction_sum += float(model.reconstruction_loss(counts, output).cpu()) * len(indices)
        reconstruction_cells += len(indices)
        state_mu = output["state_mu"]
        z_state_values.append(state_mu.cpu().numpy())
        id_prediction.append(output["identity_logits"].argmax(dim=1).cpu().numpy())
        id_truth.append(celltype.cpu().numpy())
        for task in TASKS:
            eta[task].append(model.transition_heads[task].cell_score(state_mu).squeeze(-1).cpu().numpy())
    eta = {task: np.concatenate(values) for task, values in eta.items()}
    selected_meta = data.metadata.iloc[selected_indices].reset_index(drop=True)
    participant_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | str | int]] = []
    for task, definition in TASKS.items():
        aggregate = aggregate_numpy(
            eta[task], selected_meta[data.participant_col].to_numpy(str), config.tau
        )
        head = model.transition_heads[task]
        bias = float(head.participant_bias.detach().cpu())
        mean_weight = float(F.softplus(head.raw_mean_weight).detach().cpu())
        tail_weight = float(F.softplus(head.raw_tail_weight).detach().cpu())
        aggregate["logit"] = (
            bias + mean_weight * aggregate["mean_eta"] + tail_weight * aggregate["tail_eta"]
        )
        aggregate["probability"] = 1.0 / (1.0 + np.exp(-aggregate["logit"].clip(-40, 40)))
        aggregate["analysis_stage"] = aggregate["participant"].map(data.participant_to_stage)
        aggregate = aggregate.loc[
            aggregate["analysis_stage"].isin([definition["negative"], definition["positive"]])
        ].copy()
        aggregate["y_true"] = aggregate["analysis_stage"].eq(definition["positive"]).astype(int)
        aggregate.insert(0, "task", task)
        participant_predictions.append(aggregate)
        if aggregate["y_true"].nunique() == 2:
            y_true = aggregate["y_true"].to_numpy(int)
            probability = aggregate["probability"].to_numpy(float)
            predicted = (probability >= 0.5).astype(int)
            metric_rows.extend(
                [
                    {"task": task, "metric": "AUPRC", "value": average_precision_score(y_true, probability), "n_participants": len(aggregate)},
                    {"task": task, "metric": "AUROC", "value": roc_auc_score(y_true, probability), "n_participants": len(aggregate)},
                    {"task": task, "metric": "balanced_accuracy", "value": balanced_accuracy_score(y_true, predicted), "n_participants": len(aggregate)},
                    {"task": task, "metric": "Brier", "value": brier_score_loss(y_true, probability), "n_participants": len(aggregate)},
                ]
            )
    id_prediction_array = np.concatenate(id_prediction)
    id_truth_array = np.concatenate(id_truth)
    z_state_array = np.concatenate(z_state_values)
    quality = {
        "reconstruction_nll_per_cell_per_gene": reconstruction_sum
        / max(1, reconstruction_cells)
        / data.counts.shape[1],
        "identity_macro_f1": f1_score(id_truth_array, id_prediction_array, average="macro"),
        "state_variance_min": float(np.var(z_state_array, axis=0).min()),
        "state_variance_median": float(np.median(np.var(z_state_array, axis=0))),
        "state_finite": float(np.isfinite(z_state_array).all()),
    }
    cell_scores = None
    if return_cell_scores:
        cell_scores = selected_meta[
            ["cell_id", data.participant_col, "analysis_stage", "celltype_raw"]
        ].copy()
        for task in TASKS:
            cell_scores[f"eta_{task}"] = eta[task]
    return (
        pd.concat(participant_predictions, ignore_index=True),
        cell_scores,
        pd.DataFrame(metric_rows),
        quality,
    )


@torch.no_grad()
def encode_cell_states(
    model: TBStateSCV3,
    data: PreparedData,
    participants: list[str],
    device: torch.device,
    chunk_size: int = 2048,
) -> pd.DataFrame:
    """Export frozen state posterior means without any forbidden annotation."""
    model.eval()
    selected_indices = np.concatenate([data.participant_to_cells[p] for p in participants])
    selected_indices.sort()
    identity_values = []
    state_values = []
    eta_values = {task: [] for task in TASKS}
    for start in range(0, len(selected_indices), chunk_size):
        indices = selected_indices[start : start + chunk_size]
        _, library, encoder_input, technical_batch, _ = dense_batch(data, indices, device)
        output = model(encoder_input, technical_batch, library)
        identity_values.append(output["id_mu"].cpu().numpy())
        state_values.append(output["state_mu"].cpu().numpy())
        for task in TASKS:
            eta_values[task].append(
                model.transition_heads[task]
                .cell_score(output["state_mu"])
                .squeeze(-1)
                .cpu()
                .numpy()
            )
    identities = np.concatenate(identity_values)
    states = np.concatenate(state_values)
    result = data.metadata.iloc[selected_indices][
        ["cell_id", data.participant_col, "analysis_stage", "celltype_raw"]
    ].reset_index(drop=True)
    for dimension in range(identities.shape[1]):
        result[f"z_id_{dimension + 1:02d}"] = identities[:, dimension]
    for dimension in range(states.shape[1]):
        result[f"z_state_{dimension + 1:02d}"] = states[:, dimension]
    for task in TASKS:
        result[f"eta_{task}"] = np.concatenate(eta_values[task])
    return result


@torch.no_grad()
def encode_identity_for_probe(
    model: TBStateSCV3,
    data: PreparedData,
    participants: list[str],
    device: torch.device,
    max_cells: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    indices = np.concatenate([data.participant_to_cells[p] for p in participants])
    if len(indices) > max_cells:
        indices = np.random.default_rng(seed).choice(indices, max_cells, replace=False)
    indices = np.asarray(indices, dtype=np.int64)
    latent = []
    labels = []
    for start in range(0, len(indices), 2048):
        chunk = indices[start : start + 2048]
        _, library, encoder_input, technical_batch, celltype = dense_batch(data, chunk, device)
        output = model(encoder_input, technical_batch, library)
        latent.append(output["id_mu"].cpu().numpy())
        labels.append(celltype.cpu().numpy())
    return np.concatenate(latent), np.concatenate(labels)


def identity_linear_probe_quality(
    model: TBStateSCV3,
    data: PreparedData,
    train_participants: list[str],
    validation_participants: list[str],
    device: torch.device,
    seed: int,
) -> float:
    train_latent, train_labels = encode_identity_for_probe(
        model, data, train_participants, device, 100_000, seed + 30_000
    )
    validation_latent, validation_labels = encode_identity_for_probe(
        model, data, validation_participants, device, 100_000, seed + 40_000
    )
    probe = LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=seed, n_jobs=1
    )
    probe.fit(train_latent, train_labels)
    return float(
        f1_score(validation_labels, probe.predict(validation_latent), average="macro")
    )


@torch.no_grad()
def validation_reconstruction(
    model: TBStateSCV3,
    data: PreparedData,
    participants: list[str],
    device: torch.device,
    seed: int,
    max_cells: int = 20_000,
) -> float:
    model.eval()
    indices = np.concatenate([data.participant_to_cells[p] for p in participants])
    if len(indices) > max_cells:
        indices = np.random.default_rng(seed).choice(indices, size=max_cells, replace=False)
    values = []
    weights = []
    for start in range(0, len(indices), 2048):
        chunk = np.asarray(indices[start : start + 2048], dtype=np.int64)
        counts, library, encoder_input, technical_batch, _ = dense_batch(data, chunk, device)
        output = model(encoder_input, technical_batch, library)
        values.append(float(model.reconstruction_loss(counts, output).cpu()) / data.counts.shape[1])
        weights.append(len(chunk))
    return float(np.average(values, weights=weights))


def train_model(
    data: PreparedData,
    train_participants: list[str],
    validation_participants: list[str],
    config: TrainConfig,
    seed: int,
    device: torch.device,
    output_dir: Path | None = None,
) -> tuple[TBStateSCV3, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    seed_everything(seed)
    model = TBStateSCV3(
        n_genes=data.counts.shape[1],
        n_major_celltypes=len(data.celltype_levels),
        n_technical_batches=1,
        transition_names=TASKS,
        state_dim=config.state_dim,
        tau=config.tau,
        identity_dim=config.identity_dim,
        hidden_dim=config.hidden_dim,
        likelihood=config.likelihood,
    ).to(device)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr_pretrain, weight_decay=config.weight_decay
    )
    best_pretrain = math.inf
    best_pretrain_state = None
    stale_checks = 0
    for epoch in range(1, config.max_pretrain_epochs + 1):
        model.train()
        epoch_components = []
        beta = config.beta_kl * min(1.0, epoch / 30.0)
        for indices, local_participants, stage_codes in participant_batches(
            data, train_participants, config, rng
        ):
            optimizer.zero_grad(set_to_none=True)
            loss, components = amp_safe_phase_loss(
                model, data, indices, local_participants, stage_codes,
                config, device, beta, False, use_amp,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_components.append(components)
        history.append(
            {
                "phase": "pretrain",
                "epoch": epoch,
                **{key: float(np.mean([row[key] for row in epoch_components])) for key in epoch_components[0]},
            }
        )
        if epoch % config.eval_every == 0:
            validation_loss = validation_reconstruction(
                model, data, validation_participants, device, seed + epoch
            )
            history[-1]["validation_reconstruction"] = validation_loss
            if validation_loss < best_pretrain - 1e-4:
                best_pretrain = validation_loss
                best_pretrain_state = copy.deepcopy(model.state_dict())
                stale_checks = 0
            else:
                stale_checks += 1
                if stale_checks >= config.early_stop_checks:
                    break
    if best_pretrain_state is not None:
        model.load_state_dict(best_pretrain_state)

    head_parameters = list(model.transition_heads.parameters()) + list(model.ordinal_head.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    base_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": config.lr_finetune_encoder},
            {"params": head_parameters, "lr": config.lr_heads},
        ],
        weight_decay=config.weight_decay,
    )
    best_objective = -math.inf
    best_state = None
    stale_checks = 0
    best_validation_predictions = pd.DataFrame()
    best_validation_metrics = pd.DataFrame()
    best_quality: dict[str, float] = {}
    for epoch in range(1, config.max_finetune_epochs + 1):
        model.train()
        epoch_components = []
        for indices, local_participants, stage_codes in participant_batches(
            data, train_participants, config, rng
        ):
            optimizer.zero_grad(set_to_none=True)
            loss, components = amp_safe_phase_loss(
                model, data, indices, local_participants, stage_codes,
                config, device, config.beta_kl, True, use_amp,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_components.append(components)
        history.append(
            {
                "phase": "finetune",
                "epoch": epoch,
                **{key: float(np.mean([row[key] for row in epoch_components])) for key in epoch_components[0]},
            }
        )
        if epoch % config.eval_every == 0:
            predictions, _, metrics, quality = evaluate_model(
                model,
                data,
                validation_participants,
                config,
                device,
                return_cell_scores=False,
            )
            t2 = metrics.loc[
                metrics["task"].eq("T2") & metrics["metric"].eq("AUPRC"), "value"
            ]
            objective = float(t2.iloc[0]) if len(t2) else -math.inf
            history[-1]["validation_T2_AUPRC"] = objective
            if objective > best_objective + 1e-4:
                best_objective = objective
                best_state = copy.deepcopy(model.state_dict())
                best_validation_predictions = predictions
                best_validation_metrics = metrics
                best_quality = quality
                stale_checks = 0
            else:
                stale_checks += 1
                if stale_checks >= config.early_stop_checks:
                    break
    if best_state is None:
        raise RuntimeError("Fine-tuning produced no valid T2 validation objective")
    model.load_state_dict(best_state)
    best_quality["identity_head_macro_f1"] = best_quality["identity_macro_f1"]
    best_quality["identity_macro_f1"] = identity_linear_probe_quality(
        model,
        data,
        train_participants,
        validation_participants,
        device,
        seed,
    )
    best_quality["identity_quality_definition"] = "frozen_z_id_logistic_linear_probe"
    history_frame = pd.DataFrame(history)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        history_frame.to_csv(output_dir / "training_history.tsv", sep="\t", index=False)
        best_validation_predictions.to_csv(
            output_dir / "validation_participant_predictions.tsv", sep="\t", index=False
        )
        best_validation_metrics.to_csv(
            output_dir / "validation_metrics.tsv", sep="\t", index=False
        )
        (output_dir / "validation_quality.json").write_text(
            json.dumps(best_quality, indent=2), encoding="utf-8"
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "train_config": asdict(config),
                "seed": seed,
                "gene_ids": data.gene_ids,
                "celltype_levels": data.celltype_levels,
                "tasks": TASKS,
            },
            output_dir / "model.pt",
        )
    return model, history_frame, best_validation_metrics, best_quality


def fit_model_fixed_epochs(
    data: PreparedData,
    participants: list[str],
    config: TrainConfig,
    seed: int,
    device: torch.device,
    pretrain_epochs: int,
    finetune_epochs: int,
    output_dir: Path,
) -> tuple[TBStateSCV3, pd.DataFrame]:
    """Refit a selected configuration on every outer-training participant."""
    if pretrain_epochs < 1 or finetune_epochs < 1:
        raise ValueError("Fixed epoch counts must be positive")
    seed_everything(seed)
    model = TBStateSCV3(
        n_genes=data.counts.shape[1],
        n_major_celltypes=len(data.celltype_levels),
        n_technical_batches=1,
        transition_names=TASKS,
        state_dim=config.state_dim,
        tau=config.tau,
        identity_dim=config.identity_dim,
        hidden_dim=config.hidden_dim,
        likelihood=config.likelihood,
    ).to(device)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int | str]] = []

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr_pretrain, weight_decay=config.weight_decay
    )
    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        rows = []
        beta = config.beta_kl * min(1.0, epoch / 30.0)
        for indices, local_participants, stage_codes in participant_batches(
            data, participants, config, rng
        ):
            optimizer.zero_grad(set_to_none=True)
            loss, components = amp_safe_phase_loss(
                model, data, indices, local_participants, stage_codes,
                config, device, beta, False, use_amp,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            rows.append(components)
        history.append(
            {
                "phase": "pretrain_full_outer_training",
                "epoch": epoch,
                **{key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
            }
        )

    head_parameters = list(model.transition_heads.parameters()) + list(model.ordinal_head.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    base_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": config.lr_finetune_encoder},
            {"params": head_parameters, "lr": config.lr_heads},
        ],
        weight_decay=config.weight_decay,
    )
    for epoch in range(1, finetune_epochs + 1):
        model.train()
        rows = []
        for indices, local_participants, stage_codes in participant_batches(
            data, participants, config, rng
        ):
            optimizer.zero_grad(set_to_none=True)
            loss, components = amp_safe_phase_loss(
                model, data, indices, local_participants, stage_codes,
                config, device, config.beta_kl, True, use_amp,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            rows.append(components)
        history.append(
            {
                "phase": "finetune_full_outer_training",
                "epoch": epoch,
                **{key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.tsv", sep="\t", index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "train_config": asdict(config),
            "seed": seed,
            "gene_ids": data.gene_ids,
            "celltype_levels": data.celltype_levels,
            "tasks": TASKS,
            "pretrain_epochs": pretrain_epochs,
            "finetune_epochs": finetune_epochs,
        },
        output_dir / "model.pt",
    )
    return model, history_frame


@torch.no_grad()
def encode_vanilla(
    model: VanillaCountVAE,
    data: PreparedData,
    participants: list[str],
    device: torch.device,
    max_cells: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    indices = np.concatenate([data.participant_to_cells[p] for p in participants])
    if len(indices) > max_cells:
        indices = np.random.default_rng(seed).choice(indices, size=max_cells, replace=False)
    indices = np.asarray(indices, dtype=np.int64)
    latent = []
    labels = []
    reconstruction_total = 0.0
    n_cells = 0
    for start in range(0, len(indices), 2048):
        chunk = indices[start : start + 2048]
        counts, library, encoder_input, technical_batch, celltype = dense_batch(data, chunk, device)
        output = model(encoder_input, technical_batch, library)
        latent.append(output["mu"].cpu().numpy())
        labels.append(celltype.cpu().numpy())
        reconstruction_total += float(model.reconstruction_loss(counts, output).cpu()) * len(chunk)
        n_cells += len(chunk)
    return (
        np.concatenate(latent),
        np.concatenate(labels),
        reconstruction_total / n_cells / data.counts.shape[1],
    )


def train_vanilla_representation_baseline(
    data: PreparedData,
    train_participants: list[str],
    validation_participants: list[str],
    config: TrainConfig,
    seed: int,
    device: torch.device,
    output_dir: Path | None = None,
) -> dict[str, float]:
    """Fit the unsupervised baseline used by the preregistered representation gates."""
    seed_everything(seed)
    model = VanillaCountVAE(
        n_genes=data.counts.shape[1],
        latent_dim=24,
        hidden_dim=config.hidden_dim,
        likelihood=config.likelihood,
    ).to(device)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr_pretrain, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(seed)
    best_validation = math.inf
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, config.max_pretrain_epochs + 1):
        model.train()
        losses = []
        beta = config.beta_kl * min(1.0, epoch / 30.0)
        for indices, _, _ in participant_batches(data, train_participants, config, rng):
            optimizer.zero_grad(set_to_none=True)
            counts, library, encoder_input, technical_batch, _ = dense_batch(data, indices, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(encoder_input, technical_batch, library)
                reconstruction = model.reconstruction_loss(counts, output) / data.counts.shape[1]
                kl = gaussian_kl(output["mu"], output["logvar"])
                loss = reconstruction + beta * kl
            if not torch.isfinite(loss):
                print("AMP_SAFE_FALLBACK=FP32_LOSS baseline=vanilla", flush=True)
                with torch.cuda.amp.autocast(enabled=False):
                    output = model(encoder_input.float(), technical_batch, library.float())
                    reconstruction = (
                        model.reconstruction_loss(counts.float(), output) / data.counts.shape[1]
                    )
                    kl = gaussian_kl(output["mu"].float(), output["logvar"].float())
                    loss = reconstruction + beta * kl
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Vanilla FP32 loss remained NaN/Inf after AMP_SAFE_FALLBACK"
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if epoch % config.eval_every == 0:
            _, _, validation_reconstruction_value = encode_vanilla(
                model,
                data,
                validation_participants,
                device,
                max_cells=20_000,
                seed=seed + epoch,
            )
            row["validation_reconstruction"] = validation_reconstruction_value
            if validation_reconstruction_value < best_validation - 1e-4:
                best_validation = validation_reconstruction_value
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= config.early_stop_checks:
                    history.append(row)
                    break
        history.append(row)
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    train_latent, train_labels, _ = encode_vanilla(
        model, data, train_participants, device, max_cells=100_000, seed=seed + 10_000
    )
    validation_latent, validation_labels, validation_reconstruction_value = encode_vanilla(
        model,
        data,
        validation_participants,
        device,
        max_cells=100_000,
        seed=seed + 20_000,
    )
    probe = LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=seed, n_jobs=1
    )
    probe.fit(train_latent, train_labels)
    identity_macro_f1 = f1_score(
        validation_labels, probe.predict(validation_latent), average="macro"
    )
    metrics = {
        "validation_reconstruction_nll_per_cell_per_gene": float(
            validation_reconstruction_value
        ),
        "identity_linear_probe_macro_f1": float(identity_macro_f1),
        "latent_variance_min": float(np.var(validation_latent, axis=0).min()),
        "latent_variance_median": float(np.median(np.var(validation_latent, axis=0))),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(output_dir / "training_history.tsv", sep="\t", index=False)
        (output_dir / "representation_gate_baseline.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "gene_ids": data.gene_ids,
                "seed": seed,
                "metrics": metrics,
            },
            output_dir / "vanilla_count_vae.pt",
        )
    return metrics

