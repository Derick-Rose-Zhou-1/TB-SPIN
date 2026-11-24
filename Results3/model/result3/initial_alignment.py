#!/usr/bin/env python
"""Build Tian pseudobulk anchors and fold-specific leakage-safe bulk alignment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.nn import functional as F

from tbstate_v12_models import (
    PseudobulkAlignmentModel,
    SingleCellOrdinalVAE,
    SingleCellVAEConfig,
    seed_everything,
)


ROOT = Path(os.environ.get("TBSPIN_PROJECT", "."))
V12_ROOT = ROOT / "Results3" / "result3_tbstate_end2end_v12_final"
SC_H5AD = (
    ROOT
    / "tian_16986_HVG3000_QCclean_20260602"
    / "h5ad"
    / "Tian_PBMC_16986genes_HVG3000_res2_QCclean_fullgenes.h5ad"
)
SC_STATUS = V12_ROOT / "02_SC_ENCODER_STATUS.json"
SC_LATENT = V12_ROOT / "02_SC_CELL_LATENT.tsv"
GENE_LOOKUP = V12_ROOT / "01_SHARED_GENE_LOOKUP.json"
BULK_SHARED = V12_ROOT / "01_SHARED_BULK_LOG2CPM.joblib"
SPLIT_LOCK = V12_ROOT / "00_SPLIT_LOCK.tsv"

METADATA_PATH = V12_ROOT / "03_PSEUDOBULK_ANCHOR_METADATA.tsv"
EXPRESSION_PATH = V12_ROOT / "03_PSEUDOBULK_LOG1P_CPM.joblib"
LATENT_PATH = V12_ROOT / "03_PSEUDOBULK_MEAN_SC_LATENT.tsv"
COMPOSITION_PATH = V12_ROOT / "03_PSEUDOBULK_CELL_COMPOSITION.tsv"
ALIGNMENT_CONFIG = V12_ROOT / "03_PSEUDOBULK_ALIGNMENT_CONFIG.json"
ALIGNMENT_AUDIT = V12_ROOT / "03_PSEUDOBULK_ALIGNMENT_AUDIT.tsv"
STATUS_PATH = V12_ROOT / "03_PSEUDOBULK_ALIGNMENT_STATUS.json"

SEEDS = (20260727, 20260728, 20260729)
FOLDS = (
    "heldout_GSE79362",
    "heldout_GSE94438",
    "heldout_GSE107993_GSE107994",
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-step", action="store_true")
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.8)
    parser.add_argument("--alignment-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_tsv(frame: pd.DataFrame, path: Path, float_format: str | None = None) -> None:
    temporary = path.with_suffix(path.suffix + ".building")
    frame.to_csv(
        temporary,
        sep="\t",
        index=False,
        lineterminator="\n",
        float_format=float_format,
    )
    os.replace(temporary, path)


def write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".building")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def hash_values(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def alignment_checkpoint(fold: str, seed: int) -> Path:
    return V12_ROOT / f"03_PSEUDOBULK_ALIGNMENT_{fold}_seed{seed}.pt"


def sc_checkpoint(seed: int) -> Path:
    return V12_ROOT / f"02_SC_ENCODER_CHECKPOINT_seed{seed}.pt"


def preflight(args: argparse.Namespace) -> int:
    sc_pass = False
    if SC_STATUS.is_file():
        sc_pass = (
            json.loads(SC_STATUS.read_text(encoding="utf-8")).get("status")
            == "PASS"
        )
    checks = {
        "step03_sc_encoder_pass": sc_pass,
        "sc_h5ad_exists": SC_H5AD.is_file(),
        "sc_latent_exists": SC_LATENT.is_file(),
        "gene_lookup_exists": GENE_LOOKUP.is_file(),
        "bulk_shared_exists": BULK_SHARED.is_file(),
        "split_lock_exists": SPLIT_LOCK.is_file(),
        "cuda_available": torch.cuda.is_available(),
        "min_cells_valid": args.min_cells >= 5,
        "bootstrap_fraction_valid": 0 < args.bootstrap_fraction <= 1,
    }
    passed = all(checks.values())
    print(
        json.dumps(
            {
                "step": "04_build_pseudobulk_alignment",
                "status": "PASS" if passed else "FAIL",
                "checks": checks,
            },
            indent=2,
        )
    )
    return 0 if passed else 2


def load_sc_model(seed: int) -> SingleCellOrdinalVAE:
    payload = torch.load(sc_checkpoint(seed), map_location="cpu", weights_only=False)
    model = SingleCellOrdinalVAE(SingleCellVAEConfig(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval()


def collapse_aggregated_counts(
    aggregated_full: sp.spmatrix | np.ndarray,
    gene_order: list[str],
    sc_lookup: dict[str, list[int]],
) -> np.ndarray:
    groups = [sc_lookup[gene] for gene in gene_order]
    if all(len(indices) == 1 for indices in groups):
        indices = np.asarray([entry[0] for entry in groups], dtype=np.int64)
        selected = aggregated_full[:, indices]
        return (
            selected.toarray()
            if sp.issparse(selected)
            else np.asarray(selected)
        ).astype(np.float32, copy=False)
    output = np.empty((aggregated_full.shape[0], len(groups)), dtype=np.float32)
    for output_index, indices in enumerate(groups):
        selected = aggregated_full[:, indices]
        if sp.issparse(selected):
            values = np.asarray(selected.sum(axis=1)).reshape(-1)
        else:
            values = np.asarray(selected).sum(axis=1)
        output[:, output_index] = values
    return output


def indicator_from_groups(
    group_codes: np.ndarray,
    n_groups: int,
    cell_indices: np.ndarray | None = None,
) -> sp.csr_matrix:
    if cell_indices is None:
        cell_indices = np.arange(len(group_codes), dtype=np.int64)
        codes = group_codes
    else:
        codes = group_codes[cell_indices]
    data = np.ones(len(cell_indices), dtype=np.float32)
    return sp.csr_matrix(
        (data, (codes, cell_indices)),
        shape=(n_groups, len(group_codes)),
        dtype=np.float32,
    )


def aggregate_anchor_family(
    obs: pd.DataFrame,
    counts: Any,
    cell_latent: np.ndarray,
    gene_order: list[str],
    sc_lookup: dict[str, list[int]],
    group_columns: list[str],
    anchor_type: str,
    min_cells: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    keys = obs[group_columns].astype(str).agg("||".join, axis=1)
    codes, unique_keys = pd.factorize(keys, sort=True)
    group_sizes = np.bincount(codes, minlength=len(unique_keys))
    keep_codes = np.where(group_sizes >= min_cells)[0]
    remap = np.full(len(unique_keys), -1, dtype=np.int64)
    remap[keep_codes] = np.arange(len(keep_codes), dtype=np.int64)
    keep_cells = remap[codes] >= 0
    filtered_codes = remap[codes]
    indicator = sp.csr_matrix(
        (
            np.ones(int(keep_cells.sum()), dtype=np.float32),
            (
                filtered_codes[keep_cells],
                np.flatnonzero(keep_cells),
            ),
        ),
        shape=(len(keep_codes), len(obs)),
    )
    aggregated_full = indicator @ counts
    aggregated = collapse_aggregated_counts(
        aggregated_full, gene_order, sc_lookup
    )
    mean_latent = (
        indicator @ cell_latent
    ) / group_sizes[keep_codes].reshape(-1, 1)
    records: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    for new_code, original_code in enumerate(keep_codes):
        member_indices = np.flatnonzero(codes == original_code)
        first = obs.iloc[member_indices[0]]
        anchor_id = f"{anchor_type}::{new_code:05d}"
        records.append(
            {
                "anchor_id": anchor_id,
                "anchor_type": anchor_type,
                "source_sample": str(first["sample_id_std"]),
                "stage": str(first["group"]),
                "celltype_major": (
                    str(first["celltype_major"])
                    if "celltype_major" in group_columns
                    else "ALL"
                ),
                "bootstrap_seed": "",
                "bootstrap_fraction": 1.0,
                "n_cells": len(member_indices),
                "min_cell_threshold": min_cells,
            }
        )
        composition = (
            obs.iloc[member_indices]["celltype_major"]
            .astype(str)
            .value_counts(normalize=True)
        )
        for celltype, fraction in composition.items():
            composition_rows.append(
                {
                    "anchor_id": anchor_id,
                    "celltype_major": celltype,
                    "cell_fraction": float(fraction),
                    "n_anchor_cells": len(member_indices),
                }
            )
    return (
        pd.DataFrame(records),
        aggregated,
        np.asarray(mean_latent, dtype=np.float32),
        pd.DataFrame(composition_rows),
    )


def bootstrap_anchors(
    obs: pd.DataFrame,
    counts: Any,
    cell_latent: np.ndarray,
    gene_order: list[str],
    sc_lookup: dict[str, list[int]],
    min_cells: int,
    fraction: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    base_keys = obs[["sample_id_std", "group", "celltype_major"]].astype(str)
    key_series = base_keys.agg("||".join, axis=1)
    grouped: dict[str, np.ndarray] = {
        key: np.asarray(indices, dtype=np.int64)
        for key, indices in key_series.groupby(key_series).indices.items()
        if len(indices) >= min_cells
    }
    selections: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for key in sorted(grouped):
            indices = grouped[key]
            take = max(min_cells, int(round(len(indices) * fraction)))
            take = min(take, len(indices))
            selected = np.sort(rng.choice(indices, size=take, replace=False))
            selections.append(selected)
            sample, stage, celltype = key.split("||")
            records.append(
                {
                    "anchor_id": f"BOOTSTRAP_SAMPLE_CELLTYPE::{len(records):05d}",
                    "anchor_type": "BOOTSTRAP_SAMPLE_CELLTYPE",
                    "source_sample": sample,
                    "stage": stage,
                    "celltype_major": celltype,
                    "bootstrap_seed": seed,
                    "bootstrap_fraction": fraction,
                    "n_cells": take,
                    "min_cell_threshold": min_cells,
                }
            )
    row_indices: list[int] = []
    col_indices: list[int] = []
    for anchor_index, indices in enumerate(selections):
        row_indices.extend([anchor_index] * len(indices))
        col_indices.extend(indices.tolist())
    indicator = sp.csr_matrix(
        (
            np.ones(len(row_indices), dtype=np.float32),
            (row_indices, col_indices),
        ),
        shape=(len(selections), len(obs)),
    )
    aggregated_full = indicator @ counts
    aggregated = collapse_aggregated_counts(
        aggregated_full, gene_order, sc_lookup
    )
    sizes = np.asarray([len(indices) for indices in selections], dtype=np.float32)
    mean_latent = (indicator @ cell_latent) / sizes.reshape(-1, 1)
    composition = pd.DataFrame(
        {
            "anchor_id": [record["anchor_id"] for record in records],
            "celltype_major": [record["celltype_major"] for record in records],
            "cell_fraction": 1.0,
            "n_anchor_cells": sizes.astype(int),
        }
    )
    return (
        pd.DataFrame(records),
        aggregated,
        np.asarray(mean_latent, dtype=np.float32),
        composition,
    )


def counts_to_log1p_cpm(aggregated: np.ndarray) -> np.ndarray:
    library = aggregated.sum(axis=1, keepdims=True)
    if (library <= 0).any():
        raise RuntimeError("Pseudobulk anchor has zero shared-gene library size.")
    return np.log1p(aggregated / library * 1.0e6).astype(np.float32)


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.T @ source_centered / max(1, len(source) - 1)
    target_cov = target_centered.T @ target_centered / max(1, len(target) - 1)
    return (
        F.mse_loss(source.mean(dim=0), target.mean(dim=0))
        + F.mse_loss(source_cov, target_cov)
    )


def mmd_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    distances_xx = torch.cdist(source, source).square()
    distances_yy = torch.cdist(target, target).square()
    distances_xy = torch.cdist(source, target).square()
    result = source.new_tensor(0.0)
    for bandwidth in (0.5, 1.0, 2.0, 4.0):
        gamma = 1.0 / (2.0 * bandwidth * bandwidth)
        result = result + (
            torch.exp(-gamma * distances_xx).mean()
            + torch.exp(-gamma * distances_yy).mean()
            - 2.0 * torch.exp(-gamma * distances_xy).mean()
        )
    return result / 4.0


def alignment_state_cpu(
    model: PseudobulkAlignmentModel,
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def train_alignment(
    fold: str,
    seed: int,
    pseudobulk_expression: np.ndarray,
    pseudobulk_latent: np.ndarray,
    bulk_training_expression: np.ndarray,
    training_gsms: list[str],
    heldout_gsms: list[str],
    steps: int,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_everything(seed)
    sc_model = load_sc_model(seed)
    model = PseudobulkAlignmentModel(
        n_genes=pseudobulk_expression.shape[1],
        hidden_dim=sc_model.config.hidden_dim,
        latent_dim=sc_model.config.latent_dim,
        dropout=sc_model.config.dropout,
    )
    model.initialize_from_single_cell(sc_model)
    del sc_model
    model = model.to(DEVICE)

    core_parameters = [
        *model.input_projection.parameters(),
        *model.input_norm.parameters(),
        *model.shared_linear.parameters(),
        *model.shared_norm.parameters(),
        *model.latent_projection.parameters(),
    ]
    for parameter in core_parameters:
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            model.sc_gene_scale,
            model.sc_gene_shift,
            model.bulk_gene_scale,
            model.bulk_gene_shift,
        ],
        lr=2.0e-3,
        weight_decay=1.0e-4,
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    best_loss = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    for step in range(1, steps + 1):
        pseudo_indices = rng.choice(
            len(pseudobulk_expression),
            size=min(batch_size, len(pseudobulk_expression)),
            replace=False,
        )
        bulk_indices = rng.choice(
            len(bulk_training_expression),
            size=min(batch_size, len(bulk_training_expression)),
            replace=False,
        )
        pseudo_x = torch.as_tensor(
            pseudobulk_expression[pseudo_indices], device=DEVICE
        )
        pseudo_target = torch.as_tensor(
            pseudobulk_latent[pseudo_indices], device=DEVICE
        )
        bulk_x = torch.as_tensor(
            bulk_training_expression[bulk_indices], device=DEVICE
        )
        optimizer.zero_grad(set_to_none=True)
        pseudo_predicted = model(pseudo_x, "single_cell_pseudobulk")
        bulk_latent = model(bulk_x, "bulk")
        mse = F.mse_loss(pseudo_predicted, pseudo_target)
        coral = coral_loss(bulk_latent, pseudo_target)
        mmd = mmd_loss(bulk_latent, pseudo_target)
        adapter_regularization = (
            (model.sc_gene_scale - 1.0).square().mean()
            + model.sc_gene_shift.square().mean()
            + (model.bulk_gene_scale - 1.0).square().mean()
            + model.bulk_gene_shift.square().mean()
        )
        loss = mse + 0.2 * coral + 0.1 * mmd + 0.01 * adapter_regularization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        observed = float(loss.detach().cpu())
        if observed < best_loss:
            best_loss = observed
            best_state = alignment_state_cpu(model)
        if step == 1 or step % 25 == 0 or step == steps:
            row = {
                "outer_fold": fold,
                "seed": seed,
                "step": step,
                "total_loss": observed,
                "pseudobulk_mse": float(mse.detach().cpu()),
                "coral_loss": float(coral.detach().cpu()),
                "mmd_loss": float(mmd.detach().cpu()),
                "adapter_regularization": float(
                    adapter_regularization.detach().cpu()
                ),
                "n_training_bulk_samples": len(training_gsms),
                "n_heldout_bulk_samples_not_used": len(heldout_gsms),
                "train_heldout_overlap": len(set(training_gsms) & set(heldout_gsms)),
                "outer_heldout_expression_used": False,
                "outer_heldout_outcomes_used": False,
            }
            rows.append(row)
            print(
                f"[{now_utc()}] alignment {fold} seed={seed} "
                f"step={step}/{steps} loss={observed:.6f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("No alignment state retained.")
    model.load_state_dict(best_state, strict=True)
    payload = {
        "format": "TBSTATE_V12_PSEUDOBULK_ALIGNMENT_V1",
        "outer_fold": fold,
        "seed": seed,
        "n_genes": pseudobulk_expression.shape[1],
        "hidden_dim": model.hidden_dim,
        "latent_dim": model.latent_dim,
        "dropout": model.dropout,
        "model_state": alignment_state_cpu(model),
        "training_gsms": training_gsms,
        "training_gsms_sha256": hash_values(training_gsms),
        "heldout_gsms": heldout_gsms,
        "outer_heldout_expression_used": False,
        "outer_heldout_outcomes_used": False,
        "loss_definition": "MSE + 0.2*CORAL + 0.1*MMD + 0.01*ADAPTER_PRIOR",
        "shared_core_frozen": True,
        "training_complete": True,
        "best_training_loss": best_loss,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload, rows


def save_torch_payload(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".building")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}; observed {args.seed}.")
    if args.preflight_only:
        return preflight(args)
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA is required for fold-specific alignment.")
    if (
        args.resume
        and not args.force_step
        and STATUS_PATH.is_file()
        and json.loads(STATUS_PATH.read_text(encoding="utf-8")).get("status")
        == "PASS"
    ):
        print("PASS: step 04 already complete; use --force-step to rebuild.")
        return 0
    if (
        not SC_STATUS.is_file()
        or json.loads(SC_STATUS.read_text(encoding="utf-8")).get("status")
        != "PASS"
    ):
        raise RuntimeError("Step 03 single-cell encoder PASS is required.")

    lookup = json.loads(GENE_LOOKUP.read_text(encoding="utf-8"))
    gene_order = [str(gene) for gene in lookup["shared_gene_order"]]
    adata = ad.read_h5ad(SC_H5AD, backed="r")
    obs = adata.obs[
        ["sample_id_std", "group", "celltype_major", "celltype_fine"]
    ].copy()
    latent_table = pd.read_csv(SC_LATENT, sep="\t", low_memory=False)
    if not np.array_equal(
        latent_table["cell_id"].astype(str).to_numpy(),
        obs.index.astype(str).to_numpy(),
    ):
        raise RuntimeError("SC latent rows do not match H5AD cell order.")
    latent_columns = [
        column for column in latent_table.columns if column.startswith("latent_")
    ]
    cell_latent = latent_table[latent_columns].to_numpy(dtype=np.float32)
    del latent_table

    families = [
        (
            ["sample_id_std", "celltype_major"],
            "SAMPLE_CELLTYPE",
        ),
        (
            ["sample_id_std", "group"],
            "SAMPLE_STAGE",
        ),
        (
            ["sample_id_std", "celltype_major", "group"],
            "SAMPLE_CELLTYPE_STAGE",
        ),
    ]
    metadata_parts: list[pd.DataFrame] = []
    count_parts: list[np.ndarray] = []
    latent_parts: list[np.ndarray] = []
    composition_parts: list[pd.DataFrame] = []
    for columns, anchor_type in families:
        metadata, counts, mean_latent, composition = aggregate_anchor_family(
            obs,
            adata.layers["counts"],
            cell_latent,
            gene_order,
            lookup["single_cell_column_indices"],
            columns,
            anchor_type,
            args.min_cells,
        )
        metadata_parts.append(metadata)
        count_parts.append(counts)
        latent_parts.append(mean_latent)
        composition_parts.append(composition)
    metadata, counts, mean_latent, composition = bootstrap_anchors(
        obs,
        adata.layers["counts"],
        cell_latent,
        gene_order,
        lookup["single_cell_column_indices"],
        args.min_cells,
        args.bootstrap_fraction,
    )
    metadata_parts.append(metadata)
    count_parts.append(counts)
    latent_parts.append(mean_latent)
    composition_parts.append(composition)
    adata.file.close()

    anchor_metadata = pd.concat(metadata_parts, ignore_index=True)
    aggregated_counts = np.vstack(count_parts)
    anchor_latent = np.vstack(latent_parts)
    composition_table = pd.concat(composition_parts, ignore_index=True)
    if len(anchor_metadata) != len(aggregated_counts):
        raise RuntimeError("Pseudobulk metadata/expression row mismatch.")
    pseudobulk_log1p_cpm = counts_to_log1p_cpm(aggregated_counts)
    expression_frame = pd.DataFrame(
        pseudobulk_log1p_cpm,
        index=anchor_metadata["anchor_id"],
        columns=gene_order,
    )
    expression_frame.index.name = "anchor_id"
    latent_frame = pd.DataFrame(
        anchor_latent,
        columns=latent_columns,
    )
    latent_frame.insert(0, "anchor_id", anchor_metadata["anchor_id"].to_numpy())
    write_tsv(anchor_metadata, METADATA_PATH)
    temporary_expression = EXPRESSION_PATH.with_suffix(".joblib.building")
    joblib.dump(expression_frame, temporary_expression, compress=3)
    os.replace(temporary_expression, EXPRESSION_PATH)
    write_tsv(latent_frame, LATENT_PATH, float_format="%.7g")
    write_tsv(composition_table, COMPOSITION_PATH, float_format="%.7g")

    bulk = joblib.load(BULK_SHARED)
    if bulk.columns.astype(str).tolist() != gene_order:
        raise RuntimeError("Bulk shared gene order differs from pseudobulk.")
    # Locked V11 values are log2(CPM + pseudocount); multiplication by ln(2)
    # places them on the natural-log scale used by the SC encoder input.
    bulk_encoder_input = bulk.to_numpy(dtype=np.float32) * np.float32(np.log(2.0))
    split = pd.read_csv(SPLIT_LOCK, sep="\t", low_memory=False)
    audit_rows: list[dict[str, Any]] = []
    checkpoint_paths: list[str] = []
    for fold in FOLDS:
        fold_split = split[split["outer_fold"].eq(fold)]
        training_gsms = sorted(
            set(
                fold_split.loc[
                    fold_split["split_role"].eq("OUTER_TRAINING"), "GSM"
                ].astype(str)
            )
            & set(bulk.index.astype(str))
        )
        heldout_gsms = sorted(
            set(
                fold_split.loc[
                    fold_split["split_role"].eq("OUTER_HELDOUT"), "GSM"
                ].astype(str)
            )
            & set(bulk.index.astype(str))
        )
        if set(training_gsms) & set(heldout_gsms):
            raise RuntimeError(f"Training/heldout overlap in {fold}.")
        training_positions = bulk.index.astype(str).get_indexer(training_gsms)
        if (training_positions < 0).any():
            raise RuntimeError(f"Missing training bulk samples in {fold}.")
        training_expression = bulk_encoder_input[training_positions]
        for seed in SEEDS:
            path = alignment_checkpoint(fold, seed)
            if args.resume and not args.force_step and path.is_file():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload.get("training_complete") is True:
                    checkpoint_paths.append(str(path))
                    continue
            payload, rows = train_alignment(
                fold=fold,
                seed=seed,
                pseudobulk_expression=pseudobulk_log1p_cpm,
                pseudobulk_latent=anchor_latent,
                bulk_training_expression=training_expression,
                training_gsms=training_gsms,
                heldout_gsms=heldout_gsms,
                steps=args.alignment_steps,
                batch_size=args.batch_size,
            )
            save_torch_payload(payload, path)
            checkpoint_paths.append(str(path))
            audit_rows.extend(rows)
    if audit_rows:
        write_tsv(pd.DataFrame(audit_rows), ALIGNMENT_AUDIT)
    elif not ALIGNMENT_AUDIT.is_file():
        raise RuntimeError("Resume found no alignment audit.")

    config = {
        "step": "04_build_pseudobulk_alignment",
        "generated_utc": now_utc(),
        "n_anchors": len(anchor_metadata),
        "n_shared_genes": len(gene_order),
        "latent_dim": len(latent_columns),
        "anchor_types": anchor_metadata["anchor_type"].value_counts().to_dict(),
        "minimum_cells": args.min_cells,
        "bootstrap_fraction": args.bootstrap_fraction,
        "bootstrap_seeds": list(SEEDS),
        "alignment_seeds": list(SEEDS),
        "folds": list(FOLDS),
        "alignment_steps": args.alignment_steps,
        "loss": "MSE + 0.2*CORAL + 0.1*MMD + 0.01*ADAPTER_PRIOR",
        "shared_core_initialized_from_sc_encoder": True,
        "shared_core_frozen_during_alignment": True,
        "outer_heldout_expression_used": False,
        "outer_heldout_outcomes_used": False,
    }
    write_json(config, ALIGNMENT_CONFIG)
    status = {
        **config,
        "status": "PASS",
        "outputs": {
            "metadata": str(METADATA_PATH),
            "expression": str(EXPRESSION_PATH),
            "mean_latent": str(LATENT_PATH),
            "composition": str(COMPOSITION_PATH),
            "alignment_audit": str(ALIGNMENT_AUDIT),
            "checkpoints": checkpoint_paths,
        },
    }
    write_json(status, STATUS_PATH)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
