from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import wilcoxon

from r4_common import CONTRASTS, MAJOR_CELLTYPES, bh_adjust, load_config, read_scores, safe_to_csv, score_path, setup_logging


DEG_COLUMNS = [
    "contrast", "comparison", "analysis_level", "lineage", "gene",
    "paired_participant_n", "mean_log2FC", "median_log2FC", "statistic",
    "P", "FDR", "mean_expression_risk", "mean_expression_background",
    "direction",
]


def empty_deg_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=DEG_COLUMNS)


def read_deg_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return empty_deg_frame()


def aggregate_counts(counts: sp.csr_matrix, meta: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    records: list[dict] = []
    row_ids: list[int] = []
    col_ids: list[int] = []
    for level, lineage in [("GLOBAL", "ALL"), *[("LINEAGE", x) for x in MAJOR_CELLTYPES]]:
        eligible = np.ones(len(meta), dtype=bool) if level == "GLOBAL" else meta["celltype"].eq(lineage).to_numpy()
        for participant in sorted(meta.loc[eligible, "sample_id_std"].astype(str).unique()):
            pmask = meta["sample_id_std"].astype(str).eq(participant).to_numpy() & eligible
            for association in ("scPAS+", "scPAS-", "0"):
                mask = pmask & labels.eq(association).to_numpy()
                n = int(mask.sum())
                if n == 0:
                    continue
                group_index = len(records)
                cells = np.flatnonzero(mask)
                row_ids.extend([group_index] * len(cells))
                col_ids.extend(cells.tolist())
                records.append({
                    "analysis_level": level, "lineage": lineage, "participant_id": participant,
                    "association": association, "cell_n": n,
                })
    indicator = sp.csr_matrix(
        (np.ones(len(row_ids), dtype=np.float32), (row_ids, col_ids)),
        shape=(len(records), counts.shape[0]),
    )
    aggregated = indicator @ counts
    return pd.DataFrame(records), np.asarray(aggregated.toarray(), dtype=np.float64)


def paired_deg(group_meta: pd.DataFrame, matrix: np.ndarray, genes: np.ndarray,
               risk_label: str, background_label: str, min_cells: int = 20,
               gene_chunk: int = 512) -> tuple[pd.DataFrame, int]:
    g = group_meta.copy()
    g["row"] = np.arange(len(g))
    left = g[(g["association"] == risk_label) & (g["cell_n"] >= min_cells)]
    right = g[(g["association"] == background_label) & (g["cell_n"] >= min_cells)]
    participants = sorted(set(left["participant_id"]) & set(right["participant_id"]))
    if len(participants) < 5:
        return pd.DataFrame(), len(participants)
    left_rows = left.set_index("participant_id").loc[participants, "row"].to_numpy(int)
    right_rows = right.set_index("participant_id").loc[participants, "row"].to_numpy(int)
    left_counts = matrix[left_rows]
    right_counts = matrix[right_rows]
    left_lib = left_counts.sum(axis=1, keepdims=True)
    right_lib = right_counts.sum(axis=1, keepdims=True)
    if (left_lib == 0).any() or (right_lib == 0).any():
        raise ValueError("zero pseudobulk library size")
    left_log = np.log2(left_counts / left_lib * 1e6 + 1.0)
    right_log = np.log2(right_counts / right_lib * 1e6 + 1.0)
    delta = left_log - right_log
    nonzero = np.any(delta != 0, axis=0)
    p = np.ones(delta.shape[1], dtype=float)
    statistic = np.zeros(delta.shape[1], dtype=float)
    if nonzero.any():
        nz = np.flatnonzero(nonzero)
        for start in range(0, len(nz), gene_chunk):
            idx = nz[start:start + gene_chunk]
            test = wilcoxon(delta[:, idx], axis=0, alternative="two-sided", zero_method="wilcox", method="asymptotic")
            statistic[idx] = np.asarray(test.statistic)
            p[idx] = np.asarray(test.pvalue)
    fdr = bh_adjust(p)
    result = pd.DataFrame({
        "gene": genes.astype(str), "paired_participant_n": len(participants),
        "mean_log2FC": delta.mean(axis=0), "median_log2FC": np.median(delta, axis=0),
        "statistic": statistic, "P": p, "FDR": fdr,
        "mean_expression_risk": left_log.mean(axis=0),
        "mean_expression_background": right_log.mean(axis=0),
    })
    return result, len(participants)


def run_contrast(cfg: dict, contrast: str) -> None:
    out = Path(cfg["out"])
    log = setup_logging(out / f"logs/03_pseudobulk_deg_{contrast}.log", f"deg_{contrast}")
    started = time.monotonic()
    atlas = ad.read_h5ad(cfg["atlas"], backed="r")
    meta = atlas.obs[["celltype", "sample_id_std"]].copy()
    meta.index = atlas.obs_names.astype(str)
    genes = np.asarray(atlas.var["gene_symbol"].astype(str) if "gene_symbol" in atlas.var else atlas.var_names.astype(str))
    counts = atlas.layers["counts"][:]
    if not sp.issparse(counts):
        raise TypeError("counts layer is not sparse")
    counts = counts.tocsr()
    atlas.file.close()
    labels = read_scores(score_path(cfg, contrast)).set_index("cell_id").loc[meta.index, "scPAS_label"].astype(str)
    group_meta, aggregate = aggregate_counts(counts, meta, labels)
    del counts

    comparisons = [("TARGET_ASSOCIATED", "scPAS+", "0")]
    if contrast == "C05":
        comparisons = [
            ("SYMPTOMATIC_ASSOCIATED", "scPAS+", "0"),
            ("ASYMPTOMATIC_ASSOCIATED", "scPAS-", "0"),
            ("DIRECT_SYMPTOM_AXIS", "scPAS+", "scPAS-"),
        ]
    global_rows: list[pd.DataFrame] = []
    lineage_rows: list[pd.DataFrame] = []
    status_rows: list[dict] = []
    for (level, lineage), indices in group_meta.groupby(["analysis_level", "lineage"]).groups.items():
        gm = group_meta.loc[list(indices)].reset_index(drop=True)
        mx = aggregate[np.asarray(list(indices), dtype=int)]
        for comparison, risk, background in comparisons:
            result, n_pair = paired_deg(gm, mx, genes, risk, background)
            status = "PASS" if n_pair >= 5 else "NOT_ESTIMABLE_LT5_PAIRED_PARTICIPANTS"
            status_rows.append({
                "contrast": contrast, "comparison": comparison, "analysis_level": level,
                "lineage": lineage, "risk_label": risk, "background_label": background,
                "paired_participant_n": n_pair, "min_cells_per_participant_group": 20,
                "method": "PARTICIPANT_PAIRED_PSEUDOBULK_LOGCPM_WILCOXON_ASYMPTOTIC", "status": status,
            })
            if result.empty:
                continue
            result.insert(0, "lineage", lineage)
            result.insert(0, "analysis_level", level)
            result.insert(0, "comparison", comparison)
            result.insert(0, "contrast", contrast)
            result["direction"] = np.where(result["mean_log2FC"] >= 0, "RISK_HIGHER", "BACKGROUND_HIGHER")
            (global_rows if level == "GLOBAL" else lineage_rows).append(result)
    global_df = pd.concat(global_rows, ignore_index=True) if global_rows else empty_deg_frame()
    lineage_df = pd.concat(lineage_rows, ignore_index=True) if lineage_rows else empty_deg_frame()
    global_df = global_df.reindex(columns=DEG_COLUMNS)
    lineage_df = lineage_df.reindex(columns=DEG_COLUMNS)
    safe_to_csv(global_df, out / f"02_DEG_GLOBAL/{contrast}_DEG_ALL.tsv.gz", compression="gzip")
    safe_to_csv(lineage_df, out / f"03_DEG_LINEAGE/{contrast}_LINEAGE_DEG_ALL.tsv.gz", compression="gzip")
    safe_to_csv(pd.DataFrame(status_rows), out / f"00_PROVENANCE/{contrast}_DEG_ELIGIBILITY.tsv")
    progress = {
        "contrast": contrast, "status": "PASS", "elapsed_seconds": time.monotonic() - started,
        "global_rows": len(global_df), "lineage_rows": len(lineage_df),
    }
    (out / f"logs/{contrast}_DEG_DONE.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    log.info("completed %s", progress)


def merge(cfg: dict) -> None:
    out = Path(cfg["out"])
    global_frames = [read_deg_table(out / f"02_DEG_GLOBAL/{c}_DEG_ALL.tsv.gz") for c in CONTRASTS]
    lineage_frames = [read_deg_table(out / f"03_DEG_LINEAGE/{c}_LINEAGE_DEG_ALL.tsv.gz") for c in CONTRASTS]
    g = pd.concat(global_frames, ignore_index=True) if global_frames else empty_deg_frame()
    l = pd.concat(lineage_frames, ignore_index=True) if lineage_frames else empty_deg_frame()
    g = g.reindex(columns=DEG_COLUMNS)
    l = l.reindex(columns=DEG_COLUMNS)
    safe_to_csv(g, out / "02_DEG_GLOBAL/R4_GLOBAL_DEG_ALL.tsv.gz", compression="gzip")
    safe_to_csv(l, out / "03_DEG_LINEAGE/R4_LINEAGE_MATCHED_DEG_ALL.tsv.gz", compression="gzip")
    gup = g[(g["FDR"] < 0.05) & (g["mean_log2FC"] >= 1.5)] if not g.empty else g
    lup = l[(l["FDR"] < 0.05) & (l["mean_log2FC"] >= 1.5)] if not l.empty else l
    safe_to_csv(gup, out / "02_DEG_GLOBAL/R4_GLOBAL_DEG_UP_FDR005_LFC15.tsv.gz", compression="gzip")
    safe_to_csv(lup, out / "03_DEG_LINEAGE/R4_LINEAGE_MATCHED_DEG_UP_FDR005_LFC15.tsv.gz", compression="gzip")


def main() -> int:
    parser = argparse.ArgumentParser(description="Participant-paired pseudobulk logCPM Wilcoxon DEG.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--contrast", choices=list(CONTRASTS))
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.merge:
        merge(cfg)
    elif args.contrast:
        run_contrast(cfg, args.contrast)
    else:
        raise SystemExit("specify --contrast or --merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
