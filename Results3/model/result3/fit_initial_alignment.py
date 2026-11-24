from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import torch


RELEASE = Path(os.environ["TBSPIN_R3_RELEASE"])
OUT = Path(os.environ.get("TBSPIN_R3_OUTPUT", "result3_alignment"))
SOURCE = Path(__file__).with_name("initial_alignment.py")
CANONICAL = SOURCE.parent
H5AD = RELEASE / "PROVENANCE/SHARED_INPUT_CACHE/00_INPUTS_LOCKED/single_cell/Tian_PBMC_16986genes_HVG3000_res2_QCclean_fullgenes.h5ad"
LATENT = RELEASE / "CELL_PROJECTION/02_SC_CELL_LATENT.tsv"
SC_CHECKPOINT = RELEASE / "CHECKPOINTS/02_SC_ENCODER_CHECKPOINT.pt"
GENES = RELEASE / "RESULTS/CORE/01_SHARED_GENE_LIST.tsv"
BULK = RELEASE / "RESULTS/CORE/01_SHARED_BULK_LOG2CPM.joblib"
ENDPOINTS = RELEASE / "RESULTS/V126_RUN/02_ENDPOINTS/PARTICIPANT_ENDPOINTS.tsv"
COHORTS = ("GSE79362", "GSE94438", "GSE107993_GSE107994_combined")
SEED = 20260727


def main() -> None:
    target_dir = OUT / "01_INITIAL_ALIGNMENT"
    target_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(CANONICAL))
    spec = importlib.util.spec_from_file_location("original_alignment_all_outer", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module.SEEDS = (20260727, 20260728, 20260729)
    module.sc_checkpoint = lambda seed: SC_CHECKPOINT

    missing = [c for c in COHORTS if not (target_dir / f"INITIAL_ALIGNMENT_{c}_seed{SEED}.pt").is_file()]
    if not missing:
        print("ALL_INITIAL_ALIGNMENTS_ALREADY_COMPLETE", flush=True)
        return

    gene_table = pd.read_csv(GENES, sep="\t")
    gene_order = gene_table["gene_symbol"].astype(str).tolist()
    source_indices = [[int(v) for v in str(x).split(";")] for x in gene_table["single_cell_source_columns"]]
    lookup = {g: idx for g, idx in zip(gene_order, source_indices)}
    adata = ad.read_h5ad(H5AD, backed="r")
    obs = adata.obs[["sample_id_std", "group", "celltype_major", "celltype_fine"]].copy()
    latent_table = pd.read_csv(LATENT, sep="\t", low_memory=False)
    if not np.array_equal(latent_table["cell_id"].astype(str).to_numpy(), obs.index.astype(str).to_numpy()):
        raise RuntimeError("SC_LATENT_CELL_ORDER_MISMATCH")
    latent_cols = [c for c in latent_table if c.startswith("latent_")]
    cell_latent = latent_table[latent_cols].to_numpy(dtype=np.float32)
    del latent_table

    count_parts, latent_parts = [], []
    for columns, name in (
        (["sample_id_std", "celltype_major"], "SAMPLE_CELLTYPE"),
        (["sample_id_std", "group"], "SAMPLE_STAGE"),
        (["sample_id_std", "celltype_major", "group"], "SAMPLE_CELLTYPE_STAGE"),
    ):
        _, counts, means, _ = module.aggregate_anchor_family(
            obs, adata.layers["counts"], cell_latent, gene_order, lookup, columns, name, 20
        )
        count_parts.append(counts)
        latent_parts.append(means)
    _, counts, means, _ = module.bootstrap_anchors(
        obs, adata.layers["counts"], cell_latent, gene_order, lookup, 20, 0.8
    )
    count_parts.append(counts)
    latent_parts.append(means)
    adata.file.close()
    pseudo_expression = module.counts_to_log1p_cpm(np.vstack(count_parts))
    pseudo_latent = np.vstack(latent_parts).astype(np.float32)

    bulk = joblib.load(BULK)
    bulk.index = bulk.index.astype(str)
    endpoints = pd.read_csv(ENDPOINTS, sep="\t", low_memory=False)
    endpoints["GSM"] = endpoints["GSM"].astype(str)
    audit = []
    for heldout in missing:
        target = target_dir / f"INITIAL_ALIGNMENT_{heldout}_seed{SEED}.pt"
        training_gsms = sorted(set(endpoints.loc[~endpoints["cohort"].eq(heldout), "GSM"]) & set(bulk.index))
        heldout_gsms = sorted(set(endpoints.loc[endpoints["cohort"].eq(heldout), "GSM"]) & set(bulk.index))
        positions = bulk.index.get_indexer(training_gsms)
        training_expression = bulk.to_numpy(dtype=np.float32)[positions] * np.float32(np.log(2.0))
        payload, rows = module.train_alignment(
            fold=f"heldout_{heldout}", seed=SEED,
            pseudobulk_expression=pseudo_expression,
            pseudobulk_latent=pseudo_latent,
            bulk_training_expression=training_expression,
            training_gsms=training_gsms, heldout_gsms=heldout_gsms,
            steps=300, batch_size=128,
        )
        module.save_torch_payload(payload, target)
        audit.extend(rows)
        print(f"INITIAL_ALIGNMENT_COMPLETE={heldout}", flush=True)
    pd.DataFrame(audit).to_csv(target_dir / "INITIAL_ALIGNMENT_TRAINING_AUDIT.tsv", sep="\t", index=False)
    (target_dir / "INITIAL_ALIGNMENT_STATUS.json").write_text(json.dumps({
        "status": "PASS",
        "cohorts": list(COHORTS),
        "seed": SEED,
        "endpoint_outcomes_used": False,
        "heldout_expression_used": False,
        "heldout_outcomes_used": False,
        "source_implementation": str(SOURCE),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
