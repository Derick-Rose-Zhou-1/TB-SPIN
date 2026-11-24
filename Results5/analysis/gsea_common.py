from pathlib import Path

import numpy as np
import pandas as pd

RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "inputs/resources"
SEED = 20260830
PERMUTATIONS = 10000
MIN_SIZE = 10
MAX_SIZE = 500
WEIGHT = 1.0
MSIGDB_RELEASE = "2025.1.Hs"
GMT_PATHS = {
    "GO_BP": RESOURCE_ROOT / "c5.go.bp.v2025.1.Hs.symbols.gmt",
    "GO_CC": RESOURCE_ROOT / "c5.go.cc.v2025.1.Hs.symbols.gmt",
    "GO_MF": RESOURCE_ROOT / "c5.go.mf.v2025.1.Hs.symbols.gmt",
    "KEGG": RESOURCE_ROOT / "c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt",
    "REACTOME": RESOURCE_ROOT / "c2.cp.reactome.v2025.1.Hs.symbols.gmt",
    "HALLMARK": RESOURCE_ROOT / "h.all.v2025.1.Hs.symbols.gmt",
}

def make_rank(frame: pd.DataFrame) -> pd.DataFrame:
    rank = frame.copy()
    pvalue = np.maximum(rank["raw_P"].to_numpy(float), 1e-300)
    rank["P_UNDERFLOW"] = rank["raw_P"].eq(0)
    rank["rank_score"] = np.sign(rank["log2FC"]) * -np.log10(pvalue)
    rank["abs_log2FC"] = rank["log2FC"].abs()
    rank = rank.sort_values(
        ["rank_score", "abs_log2FC", "raw_P", "gene"],
        ascending=[False, False, True, True], kind="mergesort",
    ).reset_index(drop=True)
    if not np.isfinite(rank["rank_score"]).all():
        raise RuntimeError("Primary GSEA rank contains non-finite values")
    supplied = rank["rank_score"].to_numpy(float).copy()
    for index in range(1, len(supplied)):
        if supplied[index] >= supplied[index - 1]:
            supplied[index] = np.nextafter(supplied[index - 1], -np.inf)
    if not np.isfinite(supplied).all() or not np.all(np.diff(supplied) < 0):
        raise RuntimeError("Technical GSEA rank is not finite and strictly descending")
    rank["gsea_input_rank"] = supplied
    return rank
