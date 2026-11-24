"""Project the immutable formal cell universe through one de novo VAE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from v126_dnra.data import reorder_to_contract
from v126_dnra.projection import project_latent
from v126_dnra.sc_representation import build_sc_model
from v126_dnra.train_sc import load_counts_csr, load_formal_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--gene-contract", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    formal = config["formal_training"]
    manifest = pd.read_csv(config["formal_cell_manifest"], sep="\t", compression="infer")
    rows = load_formal_rows(
        Path(config["formal_cell_manifest"]),
        expected_total=formal["n_cells"],
        expected_stage_counts=formal["stage_counts"],
    )
    manifest["h5ad_obs_index"] = pd.to_numeric(manifest["h5ad_obs_index"], errors="raise").astype(np.int64)
    manifest = manifest.sort_values("h5ad_obs_index", kind="stable").reset_index(drop=True)
    if not np.array_equal(manifest["h5ad_obs_index"].to_numpy(), rows):
        raise RuntimeError("FORMAL_MANIFEST_ORDER_CONTRACT_FAILED")
    contract = pd.read_csv(args.gene_contract, sep="\t")
    columns = contract["h5ad_column_index"].to_numpy(dtype=np.int64)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sc = checkpoint["config"]
    model = build_sc_model(
        n_genes=len(columns),
        latent_dim=sc["latent_dim"],
        hidden_dim=sc["hidden_dim"],
        dropout=sc["dropout"],
        stage_loss_weight=checkpoint["stage_loss_weight"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with h5py.File(config["h5ad"], "r") as handle:
        source = load_counts_csr(handle["layers/counts"], handle["layers/counts"].attrs["shape"][1])
    counts = reorder_to_contract(source, columns)[rows]
    del source
    latent = project_latent(model, counts, args.batch_size, device)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "FORMAL_LATENT.npy", latent)
    manifest.to_csv(args.output / "FORMAL_CELL_METADATA.tsv.gz", sep="\t", index=False, compression="gzip")
    (args.output / "STATUS.json").write_text(
        json.dumps({"status": "PASS", "n_cells": int(len(manifest)), "n_genes": int(counts.shape[1]), "latent_dim": int(latent.shape[1]), "checkpoint": str(args.checkpoint)}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
