#!/usr/bin/env python3
"""Reproducible post-processing of frozen Result 4 DEG statistics.

This script DOES NOT recompute scPAS labels, pseudobulk expression, or DEG tests.
It applies the prespecified threshold FDR < 0.05 and |mean_log2FC| >= 1.0 to
the frozen participant-paired pseudobulk DEG tables, then performs offline
GO Biological Process and KEGG over-representation analysis (ORA).

ORA uses every gene tested in the corresponding DEG group as the background.
Benjamini-Hochberg correction is performed separately for each combination of
contrast, comparison, analysis level, lineage, query direction, and database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import hypergeom


FDR_CUTOFF = 0.05
ABS_LOG2FC_CUTOFF = 1.0
EXPECTED_TESTED_GENES = 16986
DATABASES = {
    "GO_BP": "GO_Biological_Process_2023.gmt",
    "KEGG": "KEGG_2021_Human.gmt",
}
GROUP_COLS = ["contrast", "comparison", "analysis_level", "lineage"]
DEG_REQUIRED = {
    "contrast", "comparison", "analysis_level", "lineage", "gene",
    "paired_participant_n", "mean_log2FC", "median_log2FC", "statistic",
    "P", "FDR", "mean_expression_risk", "mean_expression_background",
    "direction",
}


def bh_adjust(values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted P values with monotonicity enforcement."""
    p = np.asarray(values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    good = np.isfinite(p)
    if not good.any():
        return out
    pv = p[good]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    out[good] = restored
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    """Write deterministic TSV or TSV.GZ output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".gz"):
        frame.to_csv(
            path, sep="\t", index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
            lineterminator="\n",
        )
    else:
        frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def load_gmt(path: Path) -> dict[str, set[str]]:
    gene_sets: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                genes = {x.strip().upper() for x in fields[2:] if x.strip()}
                if genes:
                    gene_sets[fields[0]] = genes
    if not gene_sets:
        raise ValueError(f"No gene sets read from {path}")
    return gene_sets


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def load_deg_inputs(base: Path) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    audits: list[dict] = []
    patterns = [
        ("GLOBAL", base / "00_INPUTS/DEG_GLOBAL", "C*_DEG_ALL.tsv.gz"),
        ("LINEAGE", base / "00_INPUTS/DEG_LINEAGE", "C*_LINEAGE_DEG_ALL.tsv.gz"),
    ]
    for expected_level, folder, pattern in patterns:
        for path in sorted(folder.glob(pattern)):
            frame = pd.read_csv(path, sep="\t")
            missing = DEG_REQUIRED - set(frame.columns)
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            if not frame.empty and set(frame["analysis_level"].astype(str)) != {expected_level}:
                raise ValueError(f"{path}: analysis_level mismatch")
            for col in ["mean_log2FC", "FDR", "P"]:
                frame[col] = pd.to_numeric(frame[col], errors="raise")
            if not frame["FDR"].between(0, 1).all():
                raise ValueError(f"{path}: FDR outside [0,1]")
            duplicate_n = int(frame.duplicated(GROUP_COLS + ["gene"]).sum())
            if duplicate_n:
                raise ValueError(f"{path}: {duplicate_n} duplicated group/gene rows")
            frames.append(frame)
            audits.append({
                "file": str(path.relative_to(base)), "bytes": path.stat().st_size,
                "sha256": sha256(path), "rows": len(frame),
                "groups": int(frame.groupby(GROUP_COLS, dropna=False).ngroups) if len(frame) else 0,
                "duplicate_group_gene_rows": duplicate_n, "status": "PASS",
            })
    if not frames:
        raise FileNotFoundError("No frozen DEG input tables found")
    merged = pd.concat(frames, ignore_index=True)
    return merged, audits


def classify_deg(deg: pd.DataFrame) -> pd.DataFrame:
    selected = deg[(deg["FDR"] < FDR_CUTOFF) & (deg["mean_log2FC"].abs() >= ABS_LOG2FC_CUTOFF)].copy()
    selected["DEG_direction"] = np.where(selected["mean_log2FC"] >= ABS_LOG2FC_CUTOFF, "UP", "DOWN")
    selected["DEG_definition"] = "FDR < 0.05 and |mean_log2FC| >= 1.0"
    return selected.sort_values(GROUP_COLS + ["DEG_direction", "FDR", "mean_log2FC", "gene"],
                                ascending=[True, True, True, True, True, True, False, True],
                                kind="mergesort").reset_index(drop=True)


def ora_group(deg: pd.DataFrame, gene_sets: dict[str, set[str]], database: str) -> pd.DataFrame:
    universe = set(deg["gene"].astype(str).str.upper())
    if len(universe) != len(deg):
        raise ValueError(f"Duplicate or case-colliding genes in {tuple(deg.iloc[0][GROUP_COLS])}")
    query_masks = {
        "UP": (deg["FDR"] < FDR_CUTOFF) & (deg["mean_log2FC"] >= ABS_LOG2FC_CUTOFF),
        "DOWN": (deg["FDR"] < FDR_CUTOFF) & (deg["mean_log2FC"] <= -ABS_LOG2FC_CUTOFF),
        "ALL_DEG": (deg["FDR"] < FDR_CUTOFF) & (deg["mean_log2FC"].abs() >= ABS_LOG2FC_CUTOFF),
    }
    rows: list[dict] = []
    identity = {col: deg.iloc[0][col] for col in GROUP_COLS}
    for query_direction, mask in query_masks.items():
        query = set(deg.loc[mask, "gene"].astype(str).str.upper())
        if not query:
            continue
        for term, raw_members in gene_sets.items():
            members = raw_members & universe
            overlap = query & members
            M, K, N, k = len(universe), len(members), len(query), len(overlap)
            if K == 0 or k == 0:
                continue
            p = float(hypergeom.sf(k - 1, M, K, N))
            odds = ((k + 0.5) * (M - K - N + k + 0.5)) / ((K - k + 0.5) * (N - k + 0.5))
            rows.append({
                **identity, "query_direction": query_direction, "database": database,
                "term": term, "query_n": N, "universe_n": M, "term_n": K,
                "overlap_n": k, "odds_ratio": odds, "P": p,
                "overlap_genes": ";".join(sorted(overlap)),
                "background_definition": "all genes tested in the corresponding frozen DEG group",
                "power_flag": "LOW_QUERY_N_LT10" if N < 10 else "OK",
            })
    result = pd.DataFrame(rows)
    if len(result):
        result["FDR"] = bh_adjust(result["P"].to_numpy())
    return result


def build_annotations(selected: pd.DataFrame, significant_ora: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict] = []
    for row in significant_ora.itertuples(index=False):
        for gene in str(row.overlap_genes).split(";"):
            long_rows.append({
                **{col: getattr(row, col) for col in GROUP_COLS},
                "gene": gene, "query_direction": row.query_direction,
                "database": row.database, "term": row.term,
                "term_FDR": row.FDR, "term_P": row.P,
                "odds_ratio": row.odds_ratio, "overlap_n": row.overlap_n,
                "query_n": row.query_n,
            })
    long = pd.DataFrame(long_rows, columns=GROUP_COLS + [
        "gene", "query_direction", "database", "term", "term_FDR", "term_P",
        "odds_ratio", "overlap_n", "query_n",
    ])
    summary = selected.copy()
    summary["gene_upper"] = summary["gene"].astype(str).str.upper()
    if len(long):
        agg = (long.sort_values(GROUP_COLS + ["gene", "database", "query_direction", "term_FDR", "term"])
               .groupby(GROUP_COLS + ["gene", "database", "query_direction"], dropna=False)["term"]
               .agg(lambda x: "; ".join(dict.fromkeys(map(str, x)))).reset_index())
        for database in DATABASES:
            for direction in ["UP", "DOWN", "ALL_DEG"]:
                col = f"{database}_{direction}_terms_FDR005"
                part = agg[(agg["database"] == database) & (agg["query_direction"] == direction)].copy()
                part = part.rename(columns={"gene": "gene_upper", "term": col})[GROUP_COLS + ["gene_upper", col]]
                summary = summary.merge(part, on=GROUP_COLS + ["gene_upper"], how="left")
                summary[col] = summary[col].fillna("")
    else:
        for database in DATABASES:
            for direction in ["UP", "DOWN", "ALL_DEG"]:
                summary[f"{database}_{direction}_terms_FDR005"] = ""
    summary["significant_functional_annotation"] = np.where(
        summary.filter(regex="_terms_FDR005$").ne("").any(axis=1), "YES", "NO"
    )
    return summary.drop(columns="gene_upper"), long


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    base = args.base.resolve()
    deg, input_audit = load_deg_inputs(base)

    group_qc: list[dict] = []
    for keys, frame in deg.groupby(GROUP_COLS, dropna=False, sort=True):
        universe_n = frame["gene"].astype(str).str.upper().nunique()
        group_qc.append({
            **dict(zip(GROUP_COLS, keys)), "tested_gene_rows": len(frame),
            "unique_tested_genes": universe_n,
            "expected_global_gene_n": EXPECTED_TESTED_GENES if keys[2] == "GLOBAL" else np.nan,
            "finite_log2FC": int(np.isfinite(frame["mean_log2FC"]).sum()),
            "finite_FDR": int(np.isfinite(frame["FDR"]).sum()),
            "status": "PASS" if universe_n > 0 else "FAIL_EMPTY_UNIVERSE",
        })
    group_qc_df = pd.DataFrame(group_qc)
    if not (group_qc_df.loc[group_qc_df.analysis_level == "GLOBAL", "unique_tested_genes"] == EXPECTED_TESTED_GENES).all():
        raise ValueError("At least one global DEG group does not contain 16,986 tested genes")

    selected = classify_deg(deg)
    write_tsv(selected, base / "01_DEG/DEG_ALL_FDR005_ABSLOG2FC1.tsv.gz")
    write_tsv(selected[selected.DEG_direction == "UP"], base / "01_DEG/DEG_UP_FDR005_LOG2FC_GE1.tsv.gz")
    write_tsv(selected[selected.DEG_direction == "DOWN"], base / "01_DEG/DEG_DOWN_FDR005_LOG2FC_LE_MINUS1.tsv.gz")
    counts = (selected.groupby(GROUP_COLS + ["DEG_direction"], dropna=False).size().unstack(fill_value=0)
              .reindex(columns=["UP", "DOWN"], fill_value=0).reset_index())
    counts["ALL_DEG"] = counts["UP"] + counts["DOWN"]
    write_tsv(counts, base / "01_DEG/DEG_COUNTS_BY_COMPARISON.tsv")
    for keys, frame in selected.groupby(GROUP_COLS, dropna=False, sort=True):
        suffix = "__".join(safe_name(x) for x in keys)
        write_tsv(frame, base / f"01_DEG/BY_GROUP/{suffix}__DEG_FDR005_ABSLOG2FC1.tsv.gz")

    libraries = {db: load_gmt(base / "00_INPUTS/gene_sets" / filename) for db, filename in DATABASES.items()}
    ora_frames: list[pd.DataFrame] = []
    for _, frame in deg.groupby(GROUP_COLS, dropna=False, sort=True):
        for database, gene_sets in libraries.items():
            result = ora_group(frame, gene_sets, database)
            if len(result):
                ora_frames.append(result)
    ora = pd.concat(ora_frames, ignore_index=True) if ora_frames else pd.DataFrame()
    if len(ora):
        ora = ora.sort_values(GROUP_COLS + ["query_direction", "database", "FDR", "P", "term"], kind="mergesort")
    go = ora[ora.database == "GO_BP"].copy()
    kegg = ora[ora.database == "KEGG"].copy()
    write_tsv(go, base / "02_GO_KEGG/GO_BP_ORA_ALL.tsv.gz")
    write_tsv(go[go.FDR < FDR_CUTOFF], base / "02_GO_KEGG/GO_BP_ORA_FDR005.tsv.gz")
    write_tsv(kegg, base / "02_GO_KEGG/KEGG_ORA_ALL.tsv.gz")
    write_tsv(kegg[kegg.FDR < FDR_CUTOFF], base / "02_GO_KEGG/KEGG_ORA_FDR005.tsv.gz")
    for contrast, frame in ora.groupby("contrast", sort=True):
        write_tsv(frame, base / f"02_GO_KEGG/BY_CONTRAST/{contrast}_GO_BP_KEGG_ORA_ALL.tsv.gz")

    significant = ora[ora.FDR < FDR_CUTOFF].copy()
    annotated, long_annotation = build_annotations(selected, significant)
    write_tsv(annotated, base / "03_ANNOTATION/DEG_WITH_SIGNIFICANT_GO_KEGG.tsv.gz")
    write_tsv(long_annotation, base / "03_ANNOTATION/DEG_TO_SIGNIFICANT_GO_KEGG_LONG.tsv.gz")

    ora_summary = (ora.assign(significant=ora.FDR < FDR_CUTOFF)
                   .groupby(GROUP_COLS + ["query_direction", "database"], dropna=False)
                   .agg(query_n=("query_n", "first"), universe_n=("universe_n", "first"),
                        terms_tested=("term", "size"), terms_FDR005=("significant", "sum"),
                        power_flag=("power_flag", "first")).reset_index())
    write_tsv(ora_summary, base / "04_AUDIT/ORA_SUMMARY.tsv")
    write_tsv(group_qc_df, base / "04_AUDIT/GROUP_QC.tsv")

    for path in sorted((base / "00_INPUTS/gene_sets").glob("*.gmt")):
        input_audit.append({
            "file": str(path.relative_to(base)), "bytes": path.stat().st_size,
            "sha256": sha256(path), "rows": sum(1 for _ in path.open(encoding="utf-8")),
            "groups": np.nan, "duplicate_group_gene_rows": np.nan, "status": "PASS",
        })
    write_tsv(pd.DataFrame(input_audit), base / "04_AUDIT/INPUT_FILE_AUDIT.tsv")
    parameters = pd.DataFrame([
        ["DEG_source", "frozen participant-paired pseudobulk DEG statistics; no DEG recomputation"],
        ["DEG_filter", "FDR < 0.05 and |mean_log2FC| >= 1.0"],
        ["UP_filter", "FDR < 0.05 and mean_log2FC >= 1.0"],
        ["DOWN_filter", "FDR < 0.05 and mean_log2FC <= -1.0"],
        ["ORA_test", "one-sided hypergeometric over-representation test"],
        ["ORA_background", "all genes tested in each contrast/comparison/analysis-level/lineage group"],
        ["multiple_testing", "Benjamini-Hochberg within each group, direction, and database"],
        ["GO_library", "GO_Biological_Process_2023 (frozen local GMT)"],
        ["KEGG_library", "KEGG_2021_Human (frozen local GMT)"],
        ["organism", "Homo sapiens"],
        ["software", f"Python {platform.python_version()}; pandas {pd.__version__}; numpy {np.__version__}; scipy {scipy.__version__}"],
    ], columns=["parameter", "value"])
    write_tsv(parameters, base / "04_AUDIT/ANALYSIS_PARAMETERS.tsv")

    validation = {
        "status": "PASS",
        "frozen_deg_rows": int(len(deg)),
        "selected_deg_rows": int(len(selected)),
        "global_groups": int((group_qc_df.analysis_level == "GLOBAL").sum()),
        "lineage_groups": int((group_qc_df.analysis_level == "LINEAGE").sum()),
        "go_bp_terms_fdr005": int((go.FDR < FDR_CUTOFF).sum()),
        "kegg_terms_fdr005": int((kegg.FDR < FDR_CUTOFF).sum()),
        "all_global_universes_16986": bool((group_qc_df.loc[group_qc_df.analysis_level == "GLOBAL", "unique_tested_genes"] == EXPECTED_TESTED_GENES).all()),
        "models_recomputed": False,
        "deg_tests_recomputed": False,
    }
    (base / "04_AUDIT/VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    output_rows = []
    for path in sorted(p for p in base.rglob("*") if p.is_file() and "00_INPUTS" not in p.parts and path.name != "SHA256SUMS.txt"):
        output_rows.append({"file": str(path.relative_to(base)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_tsv(pd.DataFrame(output_rows), base / "04_AUDIT/OUTPUT_FILE_MANIFEST.tsv")
    print(json.dumps(validation, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
