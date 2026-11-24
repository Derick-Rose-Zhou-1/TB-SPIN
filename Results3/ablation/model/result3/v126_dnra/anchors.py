"""Participant-traceable pseudobulk anchor aggregation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from v126_dnra.splits import validate_anchor_split


@dataclass(frozen=True)
class AnchorSet:
    expression: np.ndarray
    latent: np.ndarray
    metadata: pd.DataFrame


def build_anchor_table(expression: np.ndarray, latent: np.ndarray, metadata: pd.DataFrame, minimum_cells: int = 20) -> AnchorSet:
    required = {"participant_id", "source_sample", "celltype_major", "stage"}
    if required.difference(metadata.columns):
        raise ValueError("ANCHOR_METADATA_COLUMNS_MISSING")
    if len(expression) != len(metadata) or len(latent) != len(metadata):
        raise ValueError("ANCHOR_INPUT_ROW_MISMATCH")
    rows, expressions, latents = [], [], []
    groups = metadata.groupby(["participant_id", "source_sample", "celltype_major"], sort=True, dropna=False)
    for (participant_id, source_sample, celltype_major), index in groups.indices.items():
        position = np.asarray(index, dtype=np.int64)
        if len(position) < minimum_cells:
            continue
        rows.append({"anchor_id": f"SAMPLE_CELLTYPE::{source_sample}::{celltype_major}", "anchor_family": "SAMPLE_CELLTYPE", "participant_id": str(participant_id), "source_sample": str(source_sample), "celltype_major": str(celltype_major), "n_cells": int(len(position))})
        expressions.append(np.asarray(expression[position], dtype=np.float32).mean(axis=0))
        latents.append(np.asarray(latent[position], dtype=np.float32).mean(axis=0))
    if not rows:
        raise ValueError("NO_ANCHORS_MEET_MINIMUM_CELLS")
    return AnchorSet(np.vstack(expressions), np.vstack(latents), pd.DataFrame(rows))


def build_sparse_anchor_table(
    counts: csr_matrix,
    latent: np.ndarray,
    metadata: pd.DataFrame,
    minimum_cells: int = 20,
    bootstrap_seeds: tuple[int, ...] = (20260727, 20260728, 20260729),
    bootstrap_fraction: float = 0.8,
) -> AnchorSet:
    """Build the historical four-family anchor design from count-space cells.

    The primary families are sample/cell-type, sample/stage, and
    sample/cell-type/stage.  The fourth family is three deterministic 80%
    bootstrap replicas of sample/cell-type/stage groups.  CPM normalization is
    deliberately applied only after count aggregation.
    """
    required = {"participant_id", "source_sample", "celltype_major", "stage"}
    if required.difference(metadata.columns):
        raise ValueError("ANCHOR_METADATA_COLUMNS_MISSING")
    if counts.shape[0] != len(metadata) or len(latent) != len(metadata):
        raise ValueError("ANCHOR_INPUT_ROW_MISMATCH")
    if not 0 < bootstrap_fraction <= 1:
        raise ValueError("INVALID_BOOTSTRAP_FRACTION")
    rows: list[dict] = []
    expressions: list[np.ndarray] = []
    latents: list[np.ndarray] = []

    def append_anchor(
        position: np.ndarray,
        family: str,
        anchor_number: int,
        bootstrap_seed: int | str = "",
        bootstrap_value: float = 1.0,
    ) -> None:
        first = metadata.iloc[int(position[0])]
        aggregate = np.asarray(counts[position].sum(axis=0), dtype=np.float32)
        library = float(aggregate.sum())
        if library <= 0:
            raise ValueError("ZERO_LIBRARY_ANCHOR")
        expressions.append(np.log1p(aggregate.reshape(-1) / library * np.float32(1.0e6)))
        latents.append(np.asarray(latent[position], dtype=np.float32).mean(axis=0))
        rows.append(
            {
                "anchor_id": f"{family}::{anchor_number:05d}",
                "anchor_family": family,
                "participant_id": str(first["participant_id"]),
                "source_sample": str(first["source_sample"]),
                "celltype_major": str(first["celltype_major"]),
                "stage": str(first["stage"]),
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_fraction": float(bootstrap_value),
                "min_cell_threshold": int(minimum_cells),
                "n_cells": int(len(position)),
            }
        )

    for family, columns in (
        ("SAMPLE_CELLTYPE", ["participant_id", "source_sample", "celltype_major"]),
        ("SAMPLE_STAGE", ["participant_id", "source_sample", "stage"]),
        ("SAMPLE_CELLTYPE_STAGE", ["participant_id", "source_sample", "celltype_major", "stage"]),
    ):
        groups = metadata.groupby(columns, sort=True, dropna=False).indices
        anchor_number = 0
        for index in groups.values():
            position = np.asarray(index, dtype=np.int64)
            if len(position) >= minimum_cells:
                append_anchor(position, family, anchor_number)
                anchor_number += 1

    bootstrap_groups = metadata.groupby(
        ["participant_id", "source_sample", "celltype_major", "stage"], sort=True, dropna=False
    ).indices
    eligible = [np.asarray(index, dtype=np.int64) for index in bootstrap_groups.values() if len(index) >= minimum_cells]
    anchor_number = 0
    for seed in bootstrap_seeds:
        rng = np.random.default_rng(seed)
        for position in eligible:
            take = min(len(position), max(minimum_cells, int(round(len(position) * bootstrap_fraction))))
            selected = np.sort(rng.choice(position, size=take, replace=False))
            append_anchor(selected, "BOOTSTRAP_SAMPLE_CELLTYPE", anchor_number, seed, bootstrap_fraction)
            anchor_number += 1
    if not rows:
        raise ValueError("NO_ANCHORS_MEET_MINIMUM_CELLS")
    return AnchorSet(np.vstack(expressions).astype(np.float32), np.vstack(latents).astype(np.float32), pd.DataFrame(rows))


def make_participant_grouped_anchor_split(metadata: pd.DataFrame, seed: int, validation_fraction: float = 0.25) -> pd.DataFrame:
    if not 0 < validation_fraction < 1:
        raise ValueError("INVALID_VALIDATION_FRACTION")
    required = {"anchor_id", "participant_id"}
    if required.difference(metadata.columns):
        raise ValueError("ANCHOR_SPLIT_METADATA_COLUMNS_MISSING")
    participants = np.array(sorted(metadata["participant_id"].astype(str).unique()))
    if len(participants) < 2:
        raise ValueError("INSUFFICIENT_PARTICIPANTS_FOR_ANCHOR_SPLIT")
    n_validation = min(len(participants) - 1, max(1, int(round(len(participants) * validation_fraction))))
    validation = set(np.random.default_rng(seed).permutation(participants)[:n_validation].tolist())
    result = metadata[["anchor_id", "participant_id"]].copy()
    result["participant_id"] = result["participant_id"].astype(str)
    result["partition"] = np.where(result["participant_id"].isin(validation), "validation", "train")
    validate_anchor_split(
        result.loc[result["partition"].eq("train"), "participant_id"].unique().tolist(),
        result.loc[result["partition"].eq("validation"), "participant_id"].unique().tolist(),
    )
    return result
