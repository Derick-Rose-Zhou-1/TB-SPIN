#!/usr/bin/env python3
"""Project ncumap through frozen final TB-STATE V3 models without fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from fit_v3_reference_job import DISTROS
from tb_state_sc_v3_training import PreparedData, encode_cell_states
from tbstate_sc_v3_model import TBStateSCV3


ROOT = Path(os.environ.get("TBSPIN_R2_RUN", "results2_run"))
EXTERNAL = Path(os.environ.get("TBSPIN_R2_EXTERNAL", "data/ncumap.h5ad"))
SEEDS = (20260828, 20260829, 20260830)
TASKS = ("T1", "T2")


def stage_map(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    if upper == "HC":
        return "HC"
    if upper in {"LTBI", "TBINF"}:
        return "TBINF"
    if upper == "ATB":
        return "ATB"
    return f"NON_PRIMARY::{text}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True, choices=range(3))
    args = parser.parse_args()
    seed = SEEDS[args.index]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for external encoding")
    output = ROOT / f"18_EXTERNAL_VALIDATION/seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ROOT / f"17_FINAL_FULL_MODEL/seed_{seed}/model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    risk_layer = json.loads((ROOT / "17_FINAL_FULL_MODEL/FINAL_RISK_LAYER.json").read_text(encoding="utf-8"))
    seed_layer = risk_layer["layers"][str(seed)]
    gene_ids = [str(x) for x in checkpoint["gene_ids"]]
    training_celltypes = [str(x) for x in checkpoint["celltype_levels"]]
    config = checkpoint["train_config"]

    adata = ad.read_h5ad(EXTERNAL)
    if "counts" not in adata.layers:
        raise RuntimeError("external counts layer is absent")
    external_genes = pd.Index(adata.var_names.astype(str))
    if external_genes.has_duplicates:
        raise RuntimeError("external var_names are duplicated; identifier alignment is ambiguous")
    lookup = pd.Series(np.arange(len(external_genes), dtype=np.int64), index=external_genes)
    matched = [gene for gene in gene_ids if gene in lookup.index]
    missing = [gene for gene in gene_ids if gene not in lookup.index]
    positions = lookup.loc[matched].to_numpy(np.int64)
    source = adata.layers["counts"]
    source = sparse.csr_matrix(source) if not sparse.issparse(source) else source.tocsr()
    matched_matrix = source[:, positions].tocsr()
    if missing:
        matched_order = {gene: i for i, gene in enumerate(matched)}
        pieces = []
        for gene in gene_ids:
            if gene in matched_order:
                pieces.append(matched_matrix[:, matched_order[gene]])
            else:
                pieces.append(sparse.csr_matrix((adata.n_obs, 1), dtype=matched_matrix.dtype))
        counts = sparse.hstack(pieces, format="csr")
    else:
        counts = matched_matrix[:, [matched.index(gene) for gene in gene_ids]].tocsr()
    if counts.data.size and not np.allclose(counts.data, np.rint(counts.data), atol=1e-6):
        raise RuntimeError("non-integer external values reached the count target")

    obs = adata.obs.copy()
    celltype_source = "celltype_raw" if "celltype_raw" in obs else "celltype"
    if celltype_source not in obs:
        raise RuntimeError("external celltype/celltype_raw column absent")
    participant_source = "participant_id" if "participant_id" in obs else "sample_id"
    if participant_source not in obs:
        raise RuntimeError("external participant_id/sample_id absent")
    stage_source = next((x for x in ("stage", "group", "condition", "disease") if x in obs), None)
    if stage_source is None:
        raise RuntimeError("external stage-like column absent")
    raw_celltype = obs[celltype_source].astype(str)
    mapping: dict[str, str] = {}
    for value in sorted(raw_celltype.unique()):
        if value in training_celltypes:
            mapping[value] = value
        elif value.strip().upper() == "NK" and "NK cell" in training_celltypes:
            mapping[value] = "NK cell"
        else:
            mapping[value] = f"UNMATCHED::{value}"
    mapped_celltype = raw_celltype.map(mapping)
    metadata = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "sample_id_std": obs[participant_source].astype(str).to_numpy(),
        "analysis_stage": obs[stage_source].map(stage_map).to_numpy(),
        "celltype_raw": mapped_celltype.to_numpy(),
    })
    participant_to_cells = {
        str(participant): np.asarray(indices, dtype=np.int64)
        for participant, indices in metadata.groupby("sample_id_std", observed=True).indices.items()
    }
    data = PreparedData(
        counts=counts, metadata=metadata, gene_ids=gene_ids, participant_col="sample_id_std",
        participant_to_cells=participant_to_cells,
        participant_to_stage=metadata.groupby("sample_id_std", observed=True).analysis_stage.first().to_dict(),
        celltype_codes=np.zeros(len(metadata), dtype=np.int64), celltype_levels=training_celltypes,
        identity_weights=np.ones(len(training_celltypes), dtype=np.float32),
    )
    model = TBStateSCV3(
        n_genes=len(gene_ids), n_major_celltypes=len(training_celltypes), n_technical_batches=1,
        transition_names=TASKS, state_dim=int(config["state_dim"]), tau=float(config["tau"]),
        identity_dim=int(config["identity_dim"]), hidden_dim=int(config["hidden_dim"]),
        likelihood=str(config["likelihood"]),
    ).to("cuda")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    latent = encode_cell_states(model, data, sorted(participant_to_cells), torch.device("cuda"), chunk_size=4096)
    latent.insert(0, "seed", seed)
    latent.to_csv(output / "external_latent.tsv.gz", sep="\t", index=False, compression="gzip")
    zcols = [column for column in latent if column.startswith("z_state_")]
    risk_frames: list[pd.DataFrame] = []
    for task in TASKS:
        for celltype, group in latent.groupby("celltype_raw", observed=True):
            key = f"{task}|{celltype}"
            piece = group[["cell_id", "sample_id_std", "analysis_stage", "celltype_raw"]].copy()
            piece.insert(0, "seed", seed)
            piece["task"] = task
            layer = seed_layer.get(key, {"status": "UNMATCHED_CELLTYPE"})
            if layer.get("status") != "PASS":
                piece["D"] = np.nan
                piece["Y"] = np.nan
                piece["R"] = np.nan
                piece["projection_status"] = layer.get("status", "UNMATCHED_CELLTYPE")
            else:
                direction = np.asarray(layer["w"], dtype=float)
                d = group[zcols].to_numpy(float) @ direction + float(layer["intercept_b"])
                y = np.exp(np.clip(d - float(layer["reference_median_c"]), -80, 80))
                r = DISTROS[str(layer["distribution"])].cdf(y, *layer["parameters"])
                piece["D"] = d
                piece["Y"] = y
                piece["R"] = r
                piece["projection_status"] = "PASS"
            for label, cutoff in (("P90", 0.90), ("P95", 0.95), ("P975", 0.975)):
                piece[f"risk_{label}_seed"] = np.where(piece.R.notna(), piece.R >= cutoff, False)
            risk_frames.append(piece)
    pd.concat(risk_frames, ignore_index=True).to_csv(
        output / "external_risk.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    pd.DataFrame({"external_label": list(mapping), "mapped_celltype_raw": list(mapping.values())}).to_csv(
        output / "celltype_raw_mapping.tsv", sep="\t", index=False
    )
    audit = {
        "status": "PASS", "seed": seed, "external_cell_n": int(adata.n_obs),
        "external_gene_n": int(adata.n_vars), "training_gene_n": len(gene_ids),
        "matched_gene_n": len(matched), "missing_gene_n": len(missing),
        "extra_gene_n": int(adata.n_vars - len(matched)), "match_fraction": len(matched) / len(gene_ids),
        "zero_filled_genes": missing, "celltype_source": celltype_source,
        "participant_source": participant_source, "stage_source": stage_source,
        "fine_tuning": False,
    }
    (output / "AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "DONE.json").write_text(json.dumps({"status": "PASS", "seed": seed}, indent=2), encoding="utf-8")
    print(f"V3_EXTERNAL_PROJECTION=PASS seed={seed} match_fraction={audit['match_fraction']:.4f}", flush=True)


if __name__ == "__main__":
    main()
