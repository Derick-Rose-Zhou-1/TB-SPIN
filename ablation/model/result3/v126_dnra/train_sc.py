"""Train one de novo 16,930-gene single-cell VAE checkpoint on CUDA."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from v126_dnra.data import reorder_to_contract
from v126_dnra.sc_representation import build_sc_model, loss_weight_for_variant, train_minibatch


def _read_rows(group, rows: np.ndarray, n_columns: int) -> csr_matrix:
    indptr = np.asarray(group["indptr"], dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    lengths = indptr[rows + 1] - indptr[rows]
    output_ptr = np.zeros(len(rows) + 1, dtype=np.int64)
    output_ptr[1:] = np.cumsum(lengths)
    output_indices = np.empty(int(output_ptr[-1]), dtype=np.int32)
    output_data = np.empty(int(output_ptr[-1]), dtype=np.float32)
    offset = 0
    for row, length in zip(rows, lengths):
        if length:
            start, stop = int(indptr[row]), int(indptr[row + 1])
            output_indices[offset : offset + length] = group["indices"][start:stop]
            output_data[offset : offset + length] = group["data"][start:stop]
            offset += int(length)
    return csr_matrix((output_data, output_indices, output_ptr), shape=(len(rows), n_columns), dtype=np.float32)


def load_counts_csr(group, n_columns: int) -> csr_matrix:
    """Materialize the immutable H5AD counts layer once for deterministic in-memory batching."""
    indptr = np.asarray(group["indptr"], dtype=np.int64)
    indices = np.asarray(group["indices"], dtype=np.int32)
    data = np.asarray(group["data"], dtype=np.float32)
    return csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, n_columns), dtype=np.float32)


def load_formal_rows(manifest: Path, expected_total: int, expected_stage_counts: dict[str, int]) -> np.ndarray:
    frame = pd.read_csv(manifest, sep="\t", compression="infer")
    required = {"h5ad_obs_index", "stage"}
    if missing := required.difference(frame.columns):
        raise RuntimeError(f"FORMAL_MANIFEST_COLUMNS_MISSING:{sorted(missing)}")
    rows = pd.to_numeric(frame["h5ad_obs_index"], errors="raise").to_numpy(dtype=np.int64)
    if len(rows) != expected_total or len(np.unique(rows)) != expected_total or (rows < 0).any():
        raise RuntimeError("FORMAL_MANIFEST_ROW_CONTRACT_FAILED")
    observed = {str(stage): int(count) for stage, count in frame.groupby("stage").size().items()}
    if observed != expected_stage_counts:
        raise RuntimeError(f"FORMAL_MANIFEST_STAGE_CONTRACT_FAILED:{observed}")
    return np.sort(rows)


def _stage_codes(handle: h5py.File) -> np.ndarray:
    cohort = handle["obs/cohort_group"]
    categories = cohort["categories"].asstr()[:].tolist()
    lookup = {name: index for index, name in enumerate(categories)}
    expected = {"Tian_PBMC | HC": 0, "Tian_PBMC | LTBI": 1, "Tian_PBMC | ATB": 2}
    if set(expected).difference(lookup):
        raise RuntimeError(f"MISSING_ORDINAL_STAGES:{categories}")
    raw_codes = np.asarray(cohort["codes"], dtype=np.int64)
    remap = np.full(len(categories), -1, dtype=np.int64)
    for label, stage in expected.items():
        remap[lookup[label]] = stage
    stages = remap[raw_codes]
    if (stages < 0).any():
        raise RuntimeError("UNMAPPED_ORDINAL_STAGE")
    return stages


def io_efficient_batches(n_rows: int, batch_size: int, block_size: int, rng: np.random.Generator):
    if n_rows <= 0 or batch_size <= 0 or block_size < batch_size:
        raise ValueError("INVALID_IO_BATCH_CONFIGURATION")
    block_starts = np.arange(0, n_rows, block_size, dtype=np.int64)
    for block_start in rng.permutation(block_starts):
        block_stop = min(int(block_start + block_size), n_rows)
        rows = np.arange(block_start, block_stop, dtype=np.int64)
        for start in range(0, len(rows), batch_size):
            yield rows[start : start + batch_size]


def summarize_epoch(epoch: int, metrics: list[dict[str, float]], variant: str, seed: int) -> dict[str, float | int | str]:
    metric_names = ("total_loss", "reconstruction_loss", "kl_loss", "stage_loss", "gradient_norm")
    if not metrics:
        raise RuntimeError("EMPTY_EPOCH_METRICS")
    return {
        "epoch": epoch,
        **{key: float(np.mean([item[key] for item in metrics])) for key in metric_names},
        "variant": variant,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--gene-contract", required=True, type=Path)
    parser.add_argument("--variant", choices=("FULL", "NO_ORDINAL_SINGLE_CELL_SUPERVISION"), required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sc = config["single_cell"]
    formal = config["formal_training"]
    contract = pd.read_csv(args.gene_contract, sep="\t")
    if len(contract) != config["gene_contract"]["n_genes"]:
        raise RuntimeError("GENE_CONTRACT_SIZE_MISMATCH")
    columns = contract["h5ad_column_index"].to_numpy(dtype=np.int64)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    stage_weight = loss_weight_for_variant(args.variant, sc["stage_loss_weight"])
    model = build_sc_model(
        n_genes=config["gene_contract"]["n_genes"],
        latent_dim=sc["latent_dim"],
        hidden_dim=sc["hidden_dim"],
        dropout=sc["dropout"],
        stage_loss_weight=stage_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=sc["learning_rate"], weight_decay=sc["weight_decay"])
    rng = np.random.default_rng(args.seed)
    rows = []
    h5ad = Path(config["h5ad"])
    formal_rows = load_formal_rows(
        Path(config["formal_cell_manifest"]),
        expected_total=formal["n_cells"],
        expected_stage_counts=formal["stage_counts"],
    )
    with h5py.File(h5ad, "r") as handle:
        stages = _stage_codes(handle)
        if formal_rows[-1] >= len(stages):
            raise RuntimeError("FORMAL_MANIFEST_INDEX_OUT_OF_BOUNDS")
        source_counts = load_counts_csr(handle["layers/counts"], handle["layers/counts"].attrs["shape"][1])
    contract_counts = reorder_to_contract(source_counts, columns)
    del source_counts
    formal_counts = contract_counts[formal_rows]
    formal_stages = stages[formal_rows]
    del contract_counts
    print(json.dumps({"event": "COUNTS_CACHE_READY", "n_cells": int(formal_counts.shape[0]), "n_genes": int(formal_counts.shape[1]), "stage_counts": formal["stage_counts"]}, sort_keys=True), flush=True)
    
    all_rows = np.arange(len(formal_stages), dtype=np.int64)
    for epoch in range(1, sc["epochs"] + 1):
        epoch_metrics = []
        for batch_rows in io_efficient_batches(len(all_rows), sc["batch_size"], 2048, rng):
            sparse = formal_counts[batch_rows]
            counts = torch.as_tensor(sparse.toarray(), dtype=torch.float32, device=device)
            batch_stages = torch.as_tensor(formal_stages[batch_rows], dtype=torch.long, device=device)
            epoch_metrics.append(train_minibatch(model, counts, batch_stages, optimizer, min(1.0, epoch / sc["kl_warmup_epochs"])))
        epoch_row = summarize_epoch(epoch, epoch_metrics, args.variant, args.seed)
        rows.append(epoch_row)
        print(json.dumps(epoch_row, sort_keys=True), flush=True)
    checkpoint = {"format": "V126_DE_NOVO_SC_VAE", "variant": args.variant, "seed": args.seed, "config": sc, "formal_training": formal, "gene_contract": str(args.gene_contract), "model_state": model.cpu().state_dict(), "n_cells": int(len(formal_stages)), "n_genes": int(len(columns)), "stage_loss_weight": stage_weight}
    torch.save(checkpoint, args.output / "best_sc_vae.pt")
    pd.DataFrame(rows).to_csv(args.output / "training_log.tsv", sep="\t", index=False)
    (args.output / "STATUS.json").write_text(json.dumps({"status": "PASS", "variant": args.variant, "seed": args.seed, "n_cells": int(len(formal_stages)), "n_genes": int(len(columns)), "device": str(device), "stage_loss_weight": stage_weight}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
