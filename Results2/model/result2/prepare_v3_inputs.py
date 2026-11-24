#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.model_selection import StratifiedKFold


ROOT = Path(os.environ.get("TBSPIN_R2_RUN", "results2_run"))
H5AD = Path(os.environ.get("TBSPIN_R2_TRAIN", "data/train_new.h5ad"))
SEED = 20260828


def decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_column(group: h5py.Group, name: str) -> np.ndarray:
    obj = group[name]
    if isinstance(obj, h5py.Group) and {"categories", "codes"}.issubset(obj):
        categories = np.asarray([decode(x) for x in obj["categories"][()]], dtype=object)
        codes = obj["codes"][()]
        values = np.empty(len(codes), dtype=object)
        values[:] = None
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values
    return np.asarray([decode(x) for x in obj[()]], dtype=object)


def dataframe_index(group: h5py.Group) -> np.ndarray:
    name = decode(group.attrs["_index"])
    return read_column(group, name)


def load_csr(group: h5py.Group) -> sparse.csr_matrix:
    shape = tuple(int(value) for value in group.attrs["shape"])
    return sparse.csr_matrix(
        (group["data"][()], group["indices"][()], group["indptr"][()]), shape=shape
    )


def write_hvgs(counts: sparse.csr_matrix, genes: np.ndarray, mask: np.ndarray, label: str) -> None:
    print(f"[HVG] {label} cells={int(mask.sum())}", flush=True)
    frame = ad.AnnData(X=counts[mask])
    frame.var_names = genes
    sc.pp.highly_variable_genes(
        frame,
        flavor="seurat_v3",
        n_top_genes=8000,
        span=0.3,
        subset=False,
        check_values=True,
    )
    table = frame.var.reset_index(names="gene_id")
    table.to_csv(ROOT / "05_FEATURES" / f"{label}_HVG_STATISTICS.tsv.gz", sep="\t", index=False)
    ranked = table.loc[table["highly_variable_rank"].notna()].sort_values(
        ["highly_variable_rank", "gene_id"]
    )
    if len(ranked) != 8000:
        raise RuntimeError(f"{label}: expected 8000 HVGs, found {len(ranked)}")
    for n in (4000, 5000, 8000):
        ranked.head(n).assign(feature_rank=np.arange(1, n + 1)).to_csv(
            ROOT / "05_FEATURES" / f"{label}_HVG{n}.tsv", sep="\t", index=False
        )
    del frame, table, ranked
    gc.collect()


def main() -> None:
    for name in (
        "00_INPUT", "01_ENVIRONMENT", "02_INPUT_AUDIT", "03_QC", "04_SPLITS",
        "05_FEATURES", "06_MODEL_CODE", "07_MODEL_PREFLIGHT", "08_OOF_MODELS",
        "09_OOF_LATENT", "10_OOF_PARTICIPANT", "11_FISHER_COORDINATE",
        "12_REFERENCE_DISTRIBUTION", "13_RISK_CELL_CALL", "14_PARTICIPANT_BURDEN",
        "15_CELLTYPE_RAW_ANALYSIS", "16_RISK_CELL_BIOLOGY", "17_FINAL_FULL_MODEL",
        "18_EXTERNAL_VALIDATION", "19_BASELINES", "20_ABLATIONS", "21_NEGATIVE_CONTROLS",
        "22_ROBUSTNESS", "23_FIGURES", "24_TABLES", "25_METHODS", "26_REPORT",
        "27_FINAL_LOCK", "28_REVIEW_PACKAGE", "scripts", "logs",
    ):
        (ROOT / name).mkdir(parents=True, exist_ok=True)

    with h5py.File(H5AD, "r") as handle:
        obs = handle["obs"]
        var = handle["var"]
        cell_ids = dataframe_index(obs)
        genes = dataframe_index(var)
        metadata = pd.DataFrame(
            {
                "cell_id": cell_ids,
                "sample_id_std": read_column(obs, "sample_id_std"),
                "original_stage": read_column(obs, "group"),
                "dataset": read_column(obs, "dataset"),
                "celltype_raw": read_column(obs, "celltype_raw"),
            }
        )
        qc = pd.DataFrame(
            {
                "qc_clean_keep": read_column(obs, "qc_clean_keep"),
                "qc_remove_final": read_column(obs, "qc_remove_final"),
                "doublet_status": read_column(obs, "doublet_status"),
                "doublet_label": read_column(obs, "doublet_label"),
            }
        )
        counts = load_csr(handle["layers"]["counts"])
        structure = {
            "n_obs": len(cell_ids),
            "n_vars": len(genes),
            "obs_columns": list(obs.keys()),
            "var_columns": list(var.keys()),
            "layers": list(handle["layers"].keys()),
            "obsm": list(handle["obsm"].keys()),
            "uns": list(handle["uns"].keys()),
        }

    if counts.shape != (356_949, 16_986):
        raise RuntimeError(f"Unexpected count shape: {counts.shape}")
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise RuntimeError("Counts are non-finite or negative")
    if not np.allclose(counts.data, np.rint(counts.data), atol=1e-6):
        raise RuntimeError("layers[counts] is not integer-like")
    if metadata["cell_id"].duplicated().any() or metadata["sample_id_std"].isna().any():
        raise RuntimeError("Cell or participant identifiers are invalid")

    clean = (
        qc["qc_clean_keep"].eq("True")
        & qc["qc_remove_final"].eq("False")
        & qc["doublet_status"].eq("ok")
        & qc["doublet_label"].eq("singlet")
    )
    excluded = metadata.loc[~clean, ["cell_id"]].copy()
    excluded["reason"] = "explicit_preexisting_qc_flag"
    excluded.to_csv(ROOT / "03_QC/FINAL_EXCLUDE_CELL_IDS.tsv", sep="\t", index=False)
    excluded.to_csv(ROOT / "03_QC/EXCLUSION_REASON.tsv", sep="\t", index=False)
    pd.DataFrame(
        [{"original_cell_n": len(metadata), "excluded_cell_n": int((~clean).sum()), "clean_cell_n": int(clean.sum())}]
    ).to_csv(ROOT / "03_QC/QC_SUMMARY.tsv", sep="\t", index=False)
    metadata = metadata.loc[clean].reset_index(drop=True)
    counts = counts[clean.to_numpy()].tocsr()

    stage_map = {"HC": "HC", "LTBI": "TBINF", "ATB": "ATB"}
    metadata["analysis_stage"] = metadata["original_stage"].map(stage_map)
    if metadata["analysis_stage"].isna().any():
        raise RuntimeError(f"Unmapped training stages: {sorted(metadata.loc[metadata['analysis_stage'].isna(), 'original_stage'].unique())}")
    if metadata.groupby("sample_id_std")["analysis_stage"].nunique().max() != 1:
        raise RuntimeError("A training participant maps to more than one stage")

    model_metadata = metadata[
        ["cell_id", "sample_id_std", "analysis_stage", "original_stage", "dataset", "celltype_raw"]
    ].copy()
    model_metadata.to_csv(
        ROOT / "02_INPUT_AUDIT/MODEL_READY_METADATA.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    pd.DataFrame({"gene_id": genes}).to_csv(
        ROOT / "02_INPUT_AUDIT/GENE_MANIFEST.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    pd.DataFrame(
        [
            {"field": "cell_id", "source": "obs index"},
            {"field": "participant_id", "source": "sample_id_std"},
            {"field": "stage", "source": "group; LTBI mapped to TBINF"},
            {"field": "dataset", "source": "dataset"},
            {"field": "cell_annotation", "source": "celltype_raw"},
        ]
    ).to_csv(ROOT / "02_INPUT_AUDIT/METADATA_MAP.tsv", sep="\t", index=False)
    pd.DataFrame([structure]).to_csv(ROOT / "02_INPUT_AUDIT/H5AD_STRUCTURE.tsv", sep="\t", index=False)
    pd.DataFrame(
        [{"source": "layers[counts]", "shape": str(counts.shape), "finite": True, "nonnegative": True, "integer_like": True, "status": "PASS"}]
    ).to_csv(ROOT / "02_INPUT_AUDIT/RAW_COUNT_AUDIT.tsv", sep="\t", index=False)
    for filename, columns in (
        ("STAGE_COUNTS.tsv", ["analysis_stage"]),
        ("DATASET_COUNTS.tsv", ["dataset"]),
        ("CELLTYPE_RAW_COUNTS.tsv", ["celltype_raw"]),
    ):
        model_metadata.groupby(columns, observed=True).size().rename("cell_n").reset_index().to_csv(
            ROOT / "02_INPUT_AUDIT" / filename, sep="\t", index=False
        )
    participant = (
        model_metadata.groupby("sample_id_std", observed=True)
        .agg(analysis_stage=("analysis_stage", "first"), dataset=("dataset", "first"), n_cells=("cell_id", "size"))
        .reset_index()
        .sort_values("sample_id_std")
        .reset_index(drop=True)
    )
    participant.to_csv(ROOT / "02_INPUT_AUDIT/PARTICIPANT_COUNTS.tsv", sep="\t", index=False)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    participant["outer_fold"] = -1
    for fold, (_, test) in enumerate(splitter.split(participant, participant["analysis_stage"])):
        participant.loc[test, "outer_fold"] = fold
    participant.to_csv(ROOT / "04_SPLITS/PARTICIPANT_OUTER_FOLDS.tsv", sep="\t", index=False)
    inner_rows = []
    for fold in range(5):
        current = participant.loc[participant["outer_fold"].ne(fold)].copy()
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + fold + 1)
        current["inner_fold"] = -1
        for inner_fold, (_, validation) in enumerate(inner.split(current, current["analysis_stage"])):
            current.iloc[validation, current.columns.get_loc("inner_fold")] = inner_fold
        current.insert(0, "heldout_outer_fold", fold)
        inner_rows.append(current)
    pd.concat(inner_rows, ignore_index=True).to_csv(
        ROOT / "04_SPLITS/PARTICIPANT_INNER_FOLDS.tsv", sep="\t", index=False
    )
    balance = participant.groupby(["outer_fold", "analysis_stage"], observed=True).agg(
        participant_n=("sample_id_std", "size"), cell_n=("n_cells", "sum")
    ).reset_index()
    balance.to_csv(ROOT / "04_SPLITS/SPLIT_AUDIT.tsv", sep="\t", index=False)

    fold_by_participant = participant.set_index("sample_id_std")["outer_fold"]
    cell_fold = model_metadata["sample_id_std"].map(fold_by_participant).to_numpy(int)
    for fold in range(5):
        write_hvgs(counts, genes, cell_fold != fold, f"FOLD_{fold}")
    write_hvgs(counts, genes, np.ones(len(metadata), dtype=bool), "FINAL_FULL")

    (ROOT / "02_INPUT_AUDIT/INPUT_AUDIT_STATUS.json").write_text(
        json.dumps(
            {
                "status": "PASS", "raw_counts": "PASS", "annotation_column": "celltype_raw",
                "original_cell_n": 356949, "excluded_cell_n": int((~clean).sum()),
                "clean_cell_n": int(clean.sum()), "participant_n": len(participant),
            }, indent=2
        ), encoding="utf-8"
    )
    print("V3_INPUT_PREPARATION=PASS", flush=True)


if __name__ == "__main__":
    main()
