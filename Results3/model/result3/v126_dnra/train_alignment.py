"""Fit one leakage-safe LODO alignment adapter from formal anchors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from v126_dnra.alignment import bulk_log2cpm_to_ln, fit_alignment, validate_bulk_gene_order
from v126_dnra.splits import validate_outer_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--sc-checkpoint", required=True, type=Path)
    parser.add_argument("--heldout-cohort", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("bulk_alignment_transform") != "log2cpm_times_ln2":
        raise RuntimeError("BULK_ALIGNMENT_TRANSFORM_CONTRACT_FAILED")
    endpoints = pd.read_csv(config["endpoint_table"], sep="\t")
    cohorts = set(endpoints["cohort"].astype(str).unique())
    validate_outer_split(args.heldout_cohort, cohorts.difference({args.heldout_cohort}))
    heldout_gsm = set(endpoints.loc[endpoints["cohort"].astype(str).eq(args.heldout_cohort), "GSM"].astype(str))
    bulk = joblib.load(config["bulk_expression"])
    bulk.index = bulk.index.astype(str)
    frozen_genes = pd.read_csv(config["frozen_gene_list"], sep="\t")["gene_symbol"].astype(str).tolist()
    validate_bulk_gene_order(bulk.columns.to_numpy(), frozen_genes)
    train_bulk = bulk.loc[~bulk.index.isin(heldout_gsm)]
    if set(train_bulk.index).intersection(heldout_gsm):
        raise RuntimeError("HELDOUT_BULK_EXPRESSION_LEAKAGE")
    anchors = args.anchors
    expression = np.load(anchors / "ANCHOR_LOG1P_CPM.npy")
    latent = np.load(anchors / "ANCHOR_LATENT.npy")
    metadata = pd.read_csv(anchors / "ANCHOR_METADATA.tsv", sep="\t")
    split = pd.read_csv(anchors / "ANCHOR_TRAIN_VALIDATION_SPLIT.tsv", sep="\t")
    partition = metadata[["anchor_id"]].merge(split[["anchor_id", "partition"]], on="anchor_id", how="left", validate="one_to_one")["partition"].to_numpy()
    if pd.isna(partition).any():
        raise RuntimeError("ANCHOR_PARTITION_MISSING")
    checkpoint = torch.load(args.sc_checkpoint, map_location="cpu", weights_only=False)
    payload, rows = fit_alignment(
        expression,
        latent,
        partition,
        bulk_log2cpm_to_ln(train_bulk.to_numpy(dtype=np.float32)),
        checkpoint,
        config["alignment"],
        args.variant,
        args.seed,
        "cuda:0" if torch.cuda.is_available() else "cpu",
    )
    payload.update({"heldout_cohort": args.heldout_cohort, "heldout_expression_used": False, "heldout_outcomes_used": False, "n_development_bulk": int(len(train_bulk)), "bulk_alignment_transform": config["bulk_alignment_transform"]})
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = args.output / "best_alignment.pt.building"
    torch.save(payload, temporary)
    temporary.replace(args.output / "best_alignment.pt")
    pd.DataFrame(rows).to_csv(args.output / "training_log.tsv", sep="\t", index=False)
    (args.output / "STATUS.json").write_text(json.dumps({"status": "PASS", "heldout_cohort": args.heldout_cohort, "n_development_bulk": int(len(train_bulk)), "n_anchors": int(len(expression)), "best_validation_mse": payload["best_validation_mse"], "bulk_alignment_transform": config["bulk_alignment_transform"], "heldout_expression_used": False, "heldout_outcomes_used": False}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
