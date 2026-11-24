"""Select displayed GO and KEGG terms from directional enrichment tables."""
import math
import hashlib
from pathlib import Path
import pandas as pd

V5 = Path.cwd().resolve()
OUT = V5 / "26_FINAL_BALANCED_UP_DOWN_GO_KEGG"
REPLAY = OUT / "12_REPRODUCIBILITY/replay_ora"
AXES = ["H180", "H365", "COX"]
DIRECTIONS = ["UP", "DOWN"]
DATABASES = ["GO_BP", "KEGG"]
STANDARD_COLUMNS = [
    "axis",
    "direction",
    "database",
    "term_id",
    "term",
    "gene_set_size",
    "query_gene_n",
    "universe_gene_n",
    "overlap_n",
    "overlap_genes",
    "GeneRatio",
    "BgRatio",
    "odds_ratio_if_available",
    "raw_pvalue",
    "FDR",
    "minus_log10_FDR",
    "MSIGDB_release",
    "GMT_sha256",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False, na_rep="NA", lineterminator="\n")


def result_path(root: Path, axis: str, direction: str, database: str, kind: str) -> Path:
    folder = "02_GO_BP_ORA" if database == "GO_BP" else "03_KEGG_ORA"
    return root / folder / f"{axis}_{direction}_{database}_{kind}.tsv"


def read_result(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"MISSING_ORA_RESULT={path}")
    frame = pd.read_csv(path, sep="\t")
    if list(frame.columns) != STANDARD_COLUMNS:
        raise RuntimeError(f"ORA_SCHEMA_MISMATCH={path}")
    return frame


def overlap_set(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {item for item in str(value).split(";") if item and item != "NA"}


def redundancy_audit(sig: pd.DataFrame, database: str, threshold: float) -> pd.DataFrame:
    output_rows: list[dict[str, object]] = []
    for axis in AXES:
        for direction in DIRECTIONS:
            group = sig[
                (sig["axis"] == axis)
                & (sig["direction"] == direction)
                & (sig["database"] == database)
            ].copy()
            group = group.sort_values(
                ["FDR", "minus_log10_FDR", "overlap_n", "term"],
                ascending=[True, False, False, True],
                kind="mergesort",
            )
            retained: list[tuple[str, set[str]]] = []
            for rank, row in enumerate(group.itertuples(index=False), start=1):
                genes = overlap_set(row.overlap_genes)
                best_term = "NA"
                best_jaccard = 0.0
                for prior_term, prior_genes in retained:
                    union = genes | prior_genes
                    jaccard = len(genes & prior_genes) / len(union) if union else 0.0
                    if jaccard > best_jaccard:
                        best_jaccard = jaccard
                        best_term = prior_term
                redundant = best_jaccard >= threshold
                if not redundant:
                    retained.append((row.term, genes))
                record = row._asdict()
                record.update(
                    {
                        "process_rank": rank,
                        "jaccard_threshold": threshold,
                        "max_jaccard_to_retained": best_jaccard,
                        "redundant_with": best_term,
                        "REDUNDANT_DISPLAY_TERM": redundant,
                    }
                )
                output_rows.append(record)
    columns = STANDARD_COLUMNS + [
        "process_rank",
        "jaccard_threshold",
        "max_jaccard_to_retained",
        "redundant_with",
        "REDUNDANT_DISPLAY_TERM",
    ]
    return pd.DataFrame(output_rows, columns=columns)


def select_balanced(eligible: pd.DataFrame, n_common: int, status: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for axis in AXES:
        for direction in DIRECTIONS:
            group = eligible[
                (eligible["axis"] == axis) & (eligible["direction"] == direction)
            ].copy()
            ranked_parts = []
            for database in DATABASES:
                part = group[group["database"] == database].copy()
                part = part.sort_values(
                    ["FDR", "overlap_n", "term"],
                    ascending=[True, False, True],
                    kind="mergesort",
                )
                part["rank_within_database"] = range(1, len(part) + 1)
                ranked_parts.append(part)
            ranked = pd.concat(ranked_parts, ignore_index=True) if ranked_parts else group
            ranked["selected"] = False
            ranked["selection_reason"] = "BALANCE_GATE_NOT_ESTIMABLE"
            if status == "PASS":
                go = ranked[ranked["database"] == "GO_BP"]
                kegg = ranked[ranked["database"] == "KEGG"]
                target_kegg = min(math.floor(n_common / 3), len(kegg))
                target_go = n_common - target_kegg
                if len(go) < target_go:
                    target_go = len(go)
                    target_kegg = n_common - target_go
                if len(kegg) < target_kegg:
                    target_kegg = len(kegg)
                    target_go = n_common - target_kegg
                if target_go > len(go) or target_kegg > len(kegg):
                    raise RuntimeError("BALANCED_SELECTION_INFEASIBLE")
                selected_indices = list(go.head(target_go).index) + list(kegg.head(target_kegg).index)
                ranked["selection_reason"] = "NOT_SELECTED_BEYOND_FIXED_QUOTA"
                ranked.loc[selected_indices, "selected"] = True
                ranked.loc[selected_indices, "selection_reason"] = "SELECTED_FIXED_GO_KEGG_QUOTA"
                if int(ranked["selected"].sum()) != n_common:
                    raise RuntimeError("BALANCED_SELECTION_COUNT_MISMATCH")
            rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def main():
    all_frames = []
    sig_frames = []
    count_rows = []
    for axis in AXES:
        for direction in DIRECTIONS:
            for database in DATABASES:
                all_df = read_result(result_path(OUT, axis, direction, database, "ALL"))
                sig_df = read_result(result_path(OUT, axis, direction, database, "FDR05"))
                all_frames.append(all_df)
                sig_frames.append(sig_df)
                count_rows.append(
                    {
                        "axis": axis,
                        "direction": direction,
                        "database": database,
                        "query_gene_n": int(all_df["query_gene_n"].iloc[0]) if len(all_df) else int(
                            pd.read_csv(OUT / f"01_GENESETS/{axis}_{direction}_FDR05_LFC10.tsv", sep="\t").shape[0]
                        ),
                        "tested_term_n": len(all_df),
                        "rawP05_term_n": int((pd.to_numeric(all_df["raw_pvalue"], errors="coerce") < 0.05).sum()),
                        "FDR05_term_n": len(sig_df),
                    }
                )
    all_ora = pd.concat(all_frames, ignore_index=True)
    sig_ora = pd.concat(sig_frames, ignore_index=True)
    all_ora.to_csv(
        OUT / "04_ALL_ORA/ALL_UP_DOWN_GO_KEGG_ORA.tsv.gz",
        sep="\t",
        index=False,
        na_rep="NA",
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        lineterminator="\n",
    )
    write_tsv(sig_ora, OUT / "04_ALL_ORA/ALL_UP_DOWN_GO_KEGG_ORA_FDR05.tsv")
    ora_counts = pd.DataFrame(count_rows)
    write_tsv(ora_counts, OUT / "04_ALL_ORA/ORA_COUNTS.tsv")

    go_audit = redundancy_audit(sig_ora, "GO_BP", 0.60)
    kegg_audit = redundancy_audit(sig_ora, "KEGG", 0.70)
    write_tsv(go_audit, OUT / "05_REDUNDANCY/GO_BP_REDUNDANCY_AUDIT.tsv")
    write_tsv(kegg_audit, OUT / "05_REDUNDANCY/KEGG_REDUNDANCY_AUDIT.tsv")
    eligible = pd.concat([go_audit, kegg_audit], ignore_index=True)
    eligible = eligible[eligible["REDUNDANT_DISPLAY_TERM"] == False].copy()
    eligible = eligible.sort_values(["axis", "direction", "database", "FDR", "term"], kind="mergesort")
    write_tsv(eligible, OUT / "06_BALANCED_SELECTION/ELIGIBLE_DISPLAY_TERMS.tsv")

    eligible_counts = {
        (axis, direction): int(
            ((eligible["axis"] == axis) & (eligible["direction"] == direction)).sum()
        )
        for axis in AXES
        for direction in DIRECTIONS
    }
    min_available = min(eligible_counts.values())
    n_common = min(8, min_available)
    status = "PASS" if n_common >= 5 else "NOT_ESTIMABLE"
    balance_row = {
        "BALANCED_FIGURE_STATUS": status,
        "MIN_AVAILABLE": min_available,
        "N_COMMON": n_common,
    }
    for (axis, direction), count in eligible_counts.items():
        balance_row[f"{axis}_{direction}_eligible_n"] = count
    write_tsv(pd.DataFrame([balance_row]), OUT / "06_BALANCED_SELECTION/BALANCE_GATE.tsv")

    selection = select_balanced(eligible, n_common, status)
    selection_columns = [
        "axis",
        "direction",
        "database",
        "term",
        "FDR",
        "minus_log10_FDR",
        "overlap_n",
        "overlap_genes",
        "rank_within_database",
        "selected",
        "selection_reason",
    ]
    write_tsv(selection[selection_columns], OUT / "06_BALANCED_SELECTION/FINAL_DISPLAY_SELECTION.tsv")

    selected = selection[selection["selected"] == True].copy()
    figure_source_path = OUT / "07_FIGURE_SOURCE/FIG3D_BALANCED_SOURCE.tsv"
    if status == "PASS":
        selected["panel"] = selected["axis"] + "_" + selected["direction"]
        selected["source_file"] = [
            str(result_path(OUT, row.axis, row.direction, row.database, "FDR05").relative_to(OUT))
            for row in selected.itertuples(index=False)
        ]
        selected["source_sha256"] = [
            sha256(result_path(OUT, row.axis, row.direction, row.database, "FDR05"))
            for row in selected.itertuples(index=False)
        ]
        figure_columns = [
            "panel",
            "axis",
            "direction",
            "database",
            "term",
            "FDR",
            "minus_log10_FDR",
            "overlap_n",
            "overlap_genes",
            "source_file",
            "source_sha256",
        ]
        write_tsv(selected[figure_columns], figure_source_path)
    else:
        raise ValueError("Insufficient eligible terms for the balanced panel")


if __name__ == "__main__":
    main()
