from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


def read_gmt(path: Path) -> dict[str, set[str]]:
    terms = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 3:
            terms[fields[0]] = {gene.upper() for gene in fields[2:] if gene}
    return terms

def bh(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    qvalues = np.empty_like(pvalues, dtype=float)
    running = 1.0
    for rank in range(len(pvalues), 0, -1):
        index = order[rank - 1]
        running = min(running, pvalues[index] * len(pvalues) / rank)
        qvalues[index] = running
    return qvalues

def calculate(query: set[str], universe: set[str], terms: dict[str, set[str]], mode: str) -> pd.DataFrame:
    if mode == "annotated":
        annotated = set().union(*terms.values())
        universe = universe & annotated
        query = query & universe
    else:
        query = query & universe
    rows = []
    for term, members in terms.items():
        background_members = members & universe
        hits = background_members & query
        # Keep the historical clusterProfiler correction scope: overlap terms.
        if not hits:
            continue
        pvalue = hypergeom.sf(len(hits) - 1, len(universe), len(background_members), len(query))
        rows.append((term, len(query), len(universe), len(background_members), len(hits), pvalue))
    frame = pd.DataFrame(rows, columns=["term", "query_n", "universe_n", "term_n", "overlap_n", "raw_P"])
    frame["BH_FDR"] = bh(frame["raw_P"].to_numpy()) if len(frame) else np.array([], dtype=float)
    return frame.sort_values(["BH_FDR", "raw_P", "term"]).reset_index(drop=True)
