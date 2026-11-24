"""Build participant-traceable formal pseudobulk anchors for one VAE seed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from v126_dnra.anchors import build_sparse_anchor_table, make_participant_grouped_anchor_split
from v126_dnra.data import reorder_to_contract
from v126_dnra.train_sc import load_counts_csr, load_formal_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--gene-contract", required=True, type=Path)
    parser.add_argument("--latent", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    formal = config["formal_training"]
    rows = load_formal_rows(Path(config["formal_cell_manifest"]), formal["n_cells"], formal["stage_counts"])
    metadata = pd.read_csv(args.metadata, sep="\t", compression="infer")
    metadata["h5ad_obs_index"] = pd.to_numeric(metadata["h5ad_obs_index"], errors="raise").astype(np.int64)
    metadata = metadata.sort_values("h5ad_obs_index", kind="stable").reset_index(drop=True)
    if len(metadata) != formal["n_cells"] or not np.array_equal(metadata["h5ad_obs_index"].to_numpy(), rows):
        raise RuntimeError("FORMAL_ANCHOR_METADATA_CONTRACT_FAILED")
    latent = np.load(args.latent)
    if latent.shape != (formal["n_cells"], config["single_cell"]["latent_dim"]):
        raise RuntimeError("FORMAL_LATENT_SHAPE_CONTRACT_FAILED")
    contract = pd.read_csv(args.gene_contract, sep="\t")
    columns = contract["h5ad_column_index"].to_numpy(dtype=np.int64)
    with h5py.File(config["h5ad"], "r") as handle:
        source = load_counts_csr(handle["layers/counts"], handle["layers/counts"].attrs["shape"][1])
    counts = reorder_to_contract(source, columns)[rows]
    del source
    anchor_config = config["anchors"]
    anchors = build_sparse_anchor_table(
        counts,
        latent,
        metadata,
        minimum_cells=int(anchor_config["minimum_cells"]),
        bootstrap_seeds=tuple(int(seed) for seed in anchor_config["bootstrap_seeds"]),
        bootstrap_fraction=float(anchor_config["bootstrap_fraction"]),
    )
    split = make_participant_grouped_anchor_split(anchors.metadata, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "ANCHOR_LOG1P_CPM.npy", anchors.expression)
    np.save(args.output / "ANCHOR_LATENT.npy", anchors.latent)
    anchors.metadata.to_csv(args.output / "ANCHOR_METADATA.tsv", sep="\t", index=False)
    split.to_csv(args.output / "ANCHOR_TRAIN_VALIDATION_SPLIT.tsv", sep="\t", index=False)
    (args.output / "STATUS.json").write_text(
        json.dumps({"status": "PASS", "n_anchors": int(len(anchors.metadata)), "n_genes": int(anchors.expression.shape[1]), "latent_dim": int(anchors.latent.shape[1]), "anchor_families": list(anchor_config["families"]), "participant_overlap": 0}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
