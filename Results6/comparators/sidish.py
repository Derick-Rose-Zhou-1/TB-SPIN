from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from SIDISH.SIDISH import SIDISH

STAGE = Path(os.environ["SIDISH_STAGE_ROOT"])
EXEC = Path(os.environ["SIDISH_EXEC_ROOT"])
INPUT = STAGE / "input"
OUT = EXEC / "03_COX" / "FULL_COX"
QA = EXEC / "10_QA"
SEED = 20260905
OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

labels = pd.read_parquet(INPUT / "formal_labels.parquet")
x = load_npz(INPUT / "formal_log1p_cp10k.npz").tocsr()
genes = np.load(INPUT / "gene_order.npy", allow_pickle=True).astype(str)
bulk = joblib.load(INPUT / "bulk_log2cpm.joblib")
cox = pd.read_csv(INPUT / "cox_baseline_participants.tsv", sep="\t")

gene_key = pd.Index(pd.Series(genes).str.upper())
bulk_key = pd.Index(pd.Series(bulk.columns.astype(str)).str.upper())
if gene_key.has_duplicates or bulk_key.has_duplicates:
    raise RuntimeError("case-normalized gene keys must remain unique")
common = gene_key.intersection(bulk_key, sort=False)
if len(labels) != 214402 or x.shape[0] != 214402 or len(common) != 16930:
    raise RuntimeError(
        f"formal input gate failed cells={len(labels)} x={x.shape} shared={len(common)}"
    )
if len(bulk) != 795 or len(cox) != 426:
    raise RuntimeError(f"bulk/cox gate failed bulk={len(bulk)} cox={len(cox)}")

gene_pos = gene_key.get_indexer(common)
obs_cols = [
    c
    for c in ["cell_id", "participant_id", "dataset_id", "disease_group", "lineage", "subtype"]
    if c in labels
]
obs = labels[obs_cols].copy()
obs.index = obs["cell_id"].astype(str)

# This is the same float32 dense input contract used by the already-passed official canary.
adata = ad.AnnData(
    X=x[:, gene_pos].astype(np.float32).toarray(),
    obs=obs,
    var=pd.DataFrame(index=common.astype(str)),
)

bulk_lookup = bulk.copy()
bulk_lookup.index = bulk_lookup.index.astype(str)
bulk_lookup.columns = bulk_key
cox = cox.loc[cox["GSM"].astype(str).isin(bulk_lookup.index)].copy()
patients = bulk_lookup.loc[cox["GSM"].astype(str), common].copy()
patients.insert(0, "event", cox["y_true"].astype(int).to_numpy())
patients.insert(0, "duration", cox["time_to_event_or_censor_days"].astype(float).to_numpy())

config = {
    "official_class": "SIDISH.SIDISH.SIDISH",
    "official_package": "sidish==1.0.0",
    "native_distribution": "Weibull (SIDISH.Utils.weibull_min)",
    "formal_cells": len(adata),
    "formal_genes": adata.n_vars,
    "bulk_expression_available": len(bulk),
    "cox_training_participants": len(patients),
    "phase1": {
        "epochs": 225,
        "iteration_epochs": 20,
        "latent_size": 32,
        "layer_dims": [512, 128],
        "batch_size": 256,
        "optimizer": "Adam",
        "lr": 1e-4,
        "lr_3": 1e-4,
        "dropout": 0,
    },
    "phase2": {
        "epochs": 500,
        "hidden": 128,
        "lr": 1e-4,
        "dropout": 0,
        "test_size": 0.2,
        "batch_size": 256,
    },
    "train": {
        "iterations": 5,
        "percentile": 0.95,
        "steepness": 30,
        "num_workers": 0,
        "show": False,
    },
    "seed": SEED,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "input_dtype": str(adata.X.dtype),
    "float32_equivalence": "NOT_NEEDED_ALREADY_PRODUCTION_INPUT_DTYPE",
}
(QA / "SIDISH_FULL_CONFIG.json").write_text(json.dumps(config, indent=2) + "\n")

model = SIDISH(adata, patients, device=config["device"], seed=SEED)
model.init_Phase1(225, 20, 32, [512, 128], 256, "Adam", 1e-4, 1e-4, 0)
model.init_Phase2(500, 128, 1e-4, 0, 0.2, 256)
result = model.train(5, 0.95, 30, str(OUT) + "/", num_workers=0, show=False)

required = {"risk_value", "SIDISH_value", "SIDISH"}
if not required.issubset(result.obs.columns):
    raise RuntimeError(f"official output missing columns {sorted(required - set(result.obs.columns))}")
scores = result.obs.reset_index(drop=True)[obs_cols + ["risk_value", "SIDISH_value", "SIDISH"]].copy()
scores = scores.rename(columns={"SIDISH_value": "native_high_risk"})
if len(scores) != 214402 or not np.isfinite(scores["risk_value"]).all():
    raise RuntimeError("official full score QA failed")
scores.to_parquet(EXEC / "03_COX" / "SIDISH_COX_CELL_SCORES.parquet", index=False)
scores.loc[scores["native_high_risk"].astype(bool)].to_parquet(
    EXEC / "03_COX" / "SIDISH_COX_NATIVE_HIGH_RISK.parquet", index=False
)

qa = {
    "status": "PASS",
    "formal_cells": len(scores),
    "shared_genes": len(common),
    "bulk_expression_available": len(bulk),
    "cox_training_participants": len(patients),
    "native_high_risk_count": int(scores["native_high_risk"].astype(bool).sum()),
    "risk_finite": True,
    "device": config["device"],
}
(QA / "SIDISH_FULL_COX.json").write_text(json.dumps(qa, indent=2) + "\n")
print(json.dumps(qa, indent=2), flush=True)

