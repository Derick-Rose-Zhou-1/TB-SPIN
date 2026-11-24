"""Prepare normalised counts, summarise risk cells, and run the three Wilcoxon contrasts."""
import argparse
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from expression_inputs import (AXES, OUT, RAW_SOURCE, load_formal, load_normalized,
                               setup_dirs, sha256, write_json, write_tsv)

def stage01_prepare_expression() -> None:
    formal = load_formal()
    with h5py.File(RAW_SOURCE, "r") as h:
        source_ids = np.asarray(h["obs/_index"].asstr()[:], dtype=str)
        source_genes = np.asarray(h["var/_index"].asstr()[:], dtype=str)
        lookup = {cell: i for i, cell in enumerate(source_ids)}
        missing = [cell for cell in formal.cell_id.astype(str) if cell not in lookup]
        if missing:
            raise RuntimeError(f"RAW_COUNT_CELL_MISSING:{len(missing)}")
        row_idx = np.fromiter((lookup[cell] for cell in formal.cell_id.astype(str)),
                              dtype=np.int64, count=len(formal))
        layer = h["layers/counts"]
        data, indices, indptr = layer["data"], layer["indices"], layer["indptr"]
        blocks = []
        for lo in range(0, len(row_idx), 4096):
            matrices = []
            for row in row_idx[lo:lo + 4096]:
                a, b = int(indptr[row]), int(indptr[row + 1])
                matrices.append(sparse.csr_matrix((np.asarray(data[a:b], dtype=np.float64),
                                                   np.asarray(indices[a:b], dtype=np.int32),
                                                   np.array([0, b-a], dtype=np.int64)),
                                                  shape=(1, len(source_genes))))
            blocks.append(sparse.vstack(matrices, format="csr"))
        counts = sparse.vstack(blocks, format="csr")

    # Collapse duplicate official symbols by SUM if present.
    gene_codes, unique_genes = pd.factorize(source_genes, sort=False)
    if len(unique_genes) != len(source_genes):
        mapper = sparse.csr_matrix((np.ones(len(source_genes)),
                                    (np.arange(len(source_genes)), gene_codes)),
                                   shape=(len(source_genes), len(unique_genes)))
        counts = (counts @ mapper).tocsr()
        genes = np.asarray(unique_genes, dtype=str)
    else:
        genes = source_genes
    if counts.shape != (214402, len(genes)):
        raise RuntimeError(f"COUNT_SHAPE_MISMATCH:{counts.shape}")
    valid = (not counts.data.size or (np.isfinite(counts.data).all() and
             counts.data.min() >= 0 and np.allclose(counts.data, np.rint(counts.data))))
    if not valid:
        raise RuntimeError("RAW_COUNTS_INVALID")
    sparse.save_npz(OUT / "03_EXPRESSION/FORMAL_COUNTS_CSR_INTERNAL_CACHE.npz", counts, compressed=True)
    np.save(OUT / "03_EXPRESSION/GENE_ORDER.npy", genes.astype(object), allow_pickle=True)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    factors = np.divide(10000.0, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    norm = sparse.diags(factors).dot(counts).tocsr().astype(np.float32)
    norm.data = np.log1p(norm.data)
    sparse.save_npz(OUT / "03_EXPRESSION/FORMAL_LOG1P_CP10K_CSR_INTERNAL_CACHE.npz", norm, compressed=True)

    write_tsv(pd.DataFrame({"cell_id": formal.cell_id.astype(str),
                            "source_row_index": row_idx, "matched": True}),
              "01_INPUT_AUDIT/CELL_ALIGNMENT.tsv")
    map_df = pd.DataFrame({"source_gene": source_genes, "official_gene": genes[gene_codes],
                           "source_index": np.arange(len(source_genes)),
                           "duplicate_symbol_collapsed_by_sum": pd.Series(source_genes).duplicated(False).to_numpy()})
    write_tsv(map_df, "01_INPUT_AUDIT/GENE_ALIGNMENT.tsv")
    write_tsv(pd.DataFrame([{"path": str(RAW_SOURCE), "layer": "counts",
                             "sha256": sha256(RAW_SOURCE), "source_cell_n": len(source_ids),
                             "matched_formal_cell_n": len(formal), "source_gene_n": len(source_genes),
                             "final_gene_n": len(genes)}]), "01_INPUT_AUDIT/RAW_COUNT_SOURCE.tsv")
    write_json({"matched_formal_cells": len(formal), "duplicate_cell_ids": 0,
                "missing_formal_cells": 0, "counts_finite": True, "counts_nonnegative": True,
                "counts_integer_like": True, "final_gene_n": len(genes), "nnz": int(counts.nnz),
                "duplicate_gene_symbols_collapsed_by_sum": int(len(source_genes)-len(genes))},
               "01_INPUT_AUDIT/RAW_COUNT_QC.json")
    write_json({"normalization": "scanpy-equivalent normalize_total", "target_sum": 10000,
                "log_transform": "log1p", "cell_n": len(formal), "gene_n": len(genes),
                "raw_source_sha256": sha256(RAW_SOURCE)},
               "03_EXPRESSION/NORMALIZATION_CONFIG.json")
    print(f"EXPRESSION_READY cells={counts.shape[0]} genes={counts.shape[1]} nnz={counts.nnz}", flush=True)


def stage02_risk_architecture() -> None:
    d = load_formal()
    count_rows, stage_rows, lineage_rows = [], [], []
    for axis, cfg in AXES.items():
        high = d[cfg["label"]].astype(bool)
        count_rows.append({"risk_axis": axis, "total_n": len(d), "high_n": int(high.sum()),
                           "background_n": int((~high).sum()), "high_fraction": float(high.mean())})
        for group, idx in d.groupby("disease_group", dropna=False, sort=True).groups.items():
            h = high.loc[idx]
            stage_rows.append({"risk_axis": axis, "disease_group": group,
                               "all_n": len(idx), "high_n": int(h.sum()),
                               "background_n": int((~h).sum())})
        for lineage, idx in d.groupby("major_lineage", dropna=False, sort=True).groups.items():
            h = high.loc[idx]
            lineage_rows.append({"risk_axis": axis, "lineage": lineage, "all_n": len(idx),
                                 "high_n": int(h.sum()), "background_n": int((~h).sum()),
                                 "fraction_of_high": float(h.sum()/high.sum()),
                                 "within_lineage_high_fraction": float(h.mean())})
    write_tsv(pd.DataFrame(count_rows), "02_RISK_ARCHITECTURE/RISK_AXIS_COUNTS.tsv")
    write_tsv(pd.DataFrame(stage_rows), "02_RISK_ARCHITECTURE/RISK_AXIS_STAGE_COUNTS_DESCRIPTIVE.tsv")
    write_tsv(pd.DataFrame(lineage_rows), "02_RISK_ARCHITECTURE/RISK_AXIS_LINEAGE_COUNTS_DESCRIPTIVE.tsv")

    overlap = []
    for a, b in [("H180", "H365"), ("H180", "COX"), ("H365", "COX")]:
        A = d[AXES[a]["label"]].to_numpy(bool); B = d[AXES[b]["label"]].to_numpy(bool)
        inter, union = int((A & B).sum()), int((A | B).sum())
        overlap.append({"axis_A": a, "axis_B": b, "n_A": int(A.sum()), "n_B": int(B.sum()),
                        "intersection": inter, "union": union, "Jaccard": inter/union,
                        "Overlap_coefficient": inter/min(A.sum(), B.sum())})
    write_tsv(pd.DataFrame(overlap), "02_RISK_ARCHITECTURE/THREE_AXIS_CELL_OVERLAP.tsv")
    a = d[AXES["H180"]["label"]].to_numpy(bool)
    b = d[AXES["H365"]["label"]].to_numpy(bool)
    c = d[AXES["COX"]["label"]].to_numpy(bool)
    labels = np.full(len(d), "NONE", dtype=object)
    labels[a & b & c] = "ALL3_HIGH"; labels[a & b & ~c] = "H180_H365_ONLY"
    labels[a & ~b & c] = "H180_COX_ONLY"; labels[~a & b & c] = "H365_COX_ONLY"
    labels[a & ~b & ~c] = "H180_ONLY"; labels[~a & b & ~c] = "H365_ONLY"
    labels[~a & ~b & c] = "COX_ONLY"
    classes = pd.DataFrame({"cell_id": d.cell_id.astype(str), "three_axis_class": labels})
    write_tsv(classes, "02_RISK_ARCHITECTURE/THREE_AXIS_CELL_CLASS.tsv.gz")
    class_counts = (classes.three_axis_class.value_counts().rename_axis("three_axis_class")
                    .reset_index(name="n_cells").sort_values("three_axis_class", kind="mergesort"))
    write_tsv(class_counts, "02_RISK_ARCHITECTURE/THREE_AXIS_CELL_CLASS_COUNTS.tsv")
    print("RISK_ARCHITECTURE_COMPLETE", flush=True)


def _scanpy_deg(norm: sparse.csr_matrix, genes: np.ndarray, high: np.ndarray,
                axis: str) -> pd.DataFrame:
    import anndata as ad
    import scanpy as sc
    obs = pd.DataFrame({"risk_status": pd.Categorical(np.where(high, "HIGH", "BACKGROUND"),
                                                        categories=["BACKGROUND", "HIGH"])})
    adata = ad.AnnData(X=norm, obs=obs, var=pd.DataFrame(index=genes.astype(str)))
    sc.tl.rank_genes_groups(adata, groupby="risk_status", groups=["HIGH"],
                            reference="BACKGROUND", method="wilcoxon",
                            corr_method="benjamini-hochberg", tie_correct=True,
                            pts=True, use_raw=False, n_genes=adata.n_vars, key_added="deg")
    z = sc.get.rank_genes_groups_df(adata, group="HIGH", key="deg").rename(columns={
        "names": "gene", "scores": "wilcoxon_score", "logfoldchanges": "log2FC",
        "pvals": "raw_pvalue", "pvals_adj": "FDR"})
    z["gene"] = z.gene.astype(str)
    position = {gene: i for i, gene in enumerate(genes.astype(str))}
    order = np.asarray([position[gene] for gene in z.gene], dtype=int)
    z["pct_high"] = (norm[high].getnnz(axis=0) / int(high.sum()))[order]
    z["pct_background"] = (norm[~high].getnnz(axis=0) / int((~high).sum()))[order]
    z["mean_high"] = np.asarray(norm[high].mean(axis=0)).ravel()[order]
    z["mean_background"] = np.asarray(norm[~high].mean(axis=0)).ravel()[order]
    z["delta_pct"] = z.pct_high - z.pct_background
    z["high_cell_n"] = int(high.sum()); z["background_cell_n"] = int((~high).sum())
    z["tested_gene_n"] = len(genes)
    z["LOW_DETECTION_FLAG"] = (z.pct_high < .01) & (z.pct_background < .01)
    z.insert(0, "risk_axis", axis)
    cols = ["risk_axis", "gene", "wilcoxon_score", "raw_pvalue", "FDR", "log2FC",
            "pct_high", "pct_background", "mean_high", "mean_background", "delta_pct",
            "high_cell_n", "background_cell_n", "tested_gene_n", "LOW_DETECTION_FLAG"]
    return z[cols].sort_values(["FDR", "log2FC", "gene"], ascending=[True, False, True],
                               kind="mergesort")


def stage03_run_deg() -> None:
    formal = load_formal(); norm, genes = load_normalized()
    for axis, cfg in AXES.items():
        high = formal[cfg["label"]].to_numpy(bool)
        z = _scanpy_deg(norm, genes, high, axis)
        up = z[(z.FDR < .05) & (z.log2FC > 1.5)].copy()
        down = z[(z.FDR < .05) & (z.log2FC < -1.5)].copy()
        fdr = z[z.FDR < .05].copy()
        base = cfg["dir"]
        write_tsv(z, f"{base}/{axis}_HIGH_VS_BACKGROUND_ALL_GENES.tsv.gz")
        write_tsv(up, f"{base}/{axis}_RISK_UP_FDR05_LOG2FC15.tsv")
        write_tsv(down, f"{base}/{axis}_RISK_DOWN_FDR05_LOG2FC15.tsv")
        write_tsv(fdr, f"{base}/{axis}_FDR05_ALL.tsv")
        write_tsv(z, f"{base}/{axis}_VOLCANO_INPUT.tsv.gz")
        top = up.sort_values(["FDR", "log2FC", "gene"],
                             ascending=[True, False, True], kind="mergesort").head(100)
        write_tsv(top, f"{base}/{axis}_TOP100_RISK_UP.tsv")
        print(f"DEG={axis}:tested={len(z)}:fdr05={len(fdr)}:up={len(up)}:down={len(down)}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("counts","composition","deg"))
    args = parser.parse_args()
    setup_dirs()
    {"counts":stage01_prepare_expression,"composition":stage02_risk_architecture,
     "deg":stage03_run_deg}[args.stage]()
