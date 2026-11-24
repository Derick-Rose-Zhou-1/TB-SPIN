import numpy as np
import pandas as pd
from gsea_common import MSIGDB_RELEASE

def numeric(frame: pd.DataFrame, column: str):
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan

def tag_counts(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    split = values.astype(str).str.split("/", n=1, expand=True)
    if split.shape[1] != 2:
        missing = pd.Series(pd.array([pd.NA] * len(values), dtype="Int64"), index=values.index)
        return missing.copy(), missing.copy()
    return (
        pd.to_numeric(split[0], errors="coerce").astype("Int64"),
        pd.to_numeric(split[1], errors="coerce").astype("Int64"),
    )

def normalize_result(raw: pd.DataFrame, branch: str, database: str) -> pd.DataFrame:
    raw = raw.reset_index(drop=True)
    tags = raw.get("Tag %", pd.Series("", index=raw.index)).astype(str)
    genes = raw.get("Gene %", pd.Series("", index=raw.index)).astype(str)
    matched, size = tag_counts(tags)
    table = pd.DataFrame({
        "branch": branch, "database": database, "term": raw["Term"].astype(str),
        "ES": numeric(raw, "ES"), "NES": numeric(raw, "NES"),
        "nominal_P": numeric(raw, "NOM p-val"), "FDR": numeric(raw, "FDR q-val"),
        "FWER": numeric(raw, "FWER p-val"), "tag_pct": tags, "gene_pct": genes,
        "leading_edge_genes": raw.get("Lead_genes", pd.Series("", index=raw.index)).fillna("").astype(str),
        "gene_set_size": size, "matched_gene_n": matched,
        "rank_metric": "SIGNED_NEG_LOG10_RAW_P", "MSIGDB_release": MSIGDB_RELEASE,
    })
    if table["term"].duplicated().any():
        raise RuntimeError(f"{branch} {database}: duplicate GSEA terms")
    if not np.isfinite(table[["ES", "NES", "nominal_P", "FDR", "FWER"]].to_numpy(float)).all():
        raise RuntimeError(f"{branch} {database}: non-finite normalized GSEA output")
    table["abs_NES"] = table["NES"].abs()
    table = table.sort_values(["FDR", "abs_NES", "term"], ascending=[True, False, True])
    return table.drop(columns="abs_NES")
