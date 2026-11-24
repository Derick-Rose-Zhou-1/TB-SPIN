from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from r4_common import bh_adjust, load_config, load_gmt, safe_to_csv, save_gmt, setup_logging, sha256


LIBRARIES = {
    "GO_BP": "GO_Biological_Process_2023",
    "KEGG": "KEGG_2021_Human",
}


def ensure_gene_sets(out: Path, log) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    rows: list[dict] = []
    for database, library in LIBRARIES.items():
        path = out / f"00_PROVENANCE/gene_sets/{library}.gmt"
        if not path.exists():
            log.info("retrieving the same named Enrichr library used previously: %s", library)
            gene_sets = gp.get_library(name=library, organism="Human")
            save_gmt(gene_sets, path)
        paths[database] = path
        rows.append({
            "database": database, "library": library, "version_or_date": library.rsplit("_", 1)[-1],
            "retrieval_date": dt.date.today().isoformat(), "source": "Enrichr via gseapy.get_library",
            "source_file": str(path), "sha256": sha256(path),
        })
    safe_to_csv(pd.DataFrame(rows), out / "00_PROVENANCE/GENE_SET_MANIFEST.tsv")
    return paths


def ora_one(deg: pd.DataFrame, database: str, sets: dict[str, set[str]]) -> pd.DataFrame:
    universe = set(deg["gene"].astype(str).str.upper())
    comparison = str(deg["comparison"].iloc[0])
    directions = [("TARGET_HIGH", deg["mean_log2FC"] >= 1.5)]
    if comparison == "DIRECT_SYMPTOM_AXIS":
        directions = [
            ("SYMPTOMATIC_HIGH", deg["mean_log2FC"] >= 1.5),
            ("ASYMPTOMATIC_HIGH", deg["mean_log2FC"] <= -1.5),
        ]
    output: list[dict] = []
    for direction, effect_mask in directions:
        query = set(deg.loc[(deg["FDR"] < 0.05) & effect_mask, "gene"].astype(str).str.upper())
        if not query:
            continue
        for term, term_genes in sets.items():
            members = term_genes & universe
            overlap = query & members
            M, K, N, k = len(universe), len(members), len(query), len(overlap)
            if K == 0 or k == 0:
                continue
            p = float(hypergeom.sf(k - 1, M, K, N))
            non_query_term = K - k
            query_non_term = N - k
            neither = M - K - N + k
            odds = ((k + 0.5) * (neither + 0.5)) / ((non_query_term + 0.5) * (query_non_term + 0.5))
            output.append({
                "direction": direction, "database": database, "term": term,
                "query_n": N, "universe_n": M, "term_n": K, "overlap_n": k,
                "odds_ratio": odds, "P": p, "genes": ";".join(sorted(overlap)),
            })
    result = pd.DataFrame(output)
    if not result.empty:
        result["FDR"] = bh_adjust(result["P"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline GO BP and KEGG ORA using tested genes as universe.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["out"])
    log = setup_logging(out / "logs/04_go_kegg_ora.log", "go_kegg")
    paths = ensure_gene_sets(out, log)
    global_deg = pd.read_csv(out / "02_DEG_GLOBAL/R4_GLOBAL_DEG_ALL.tsv.gz", sep="\t")
    lineage_deg = pd.read_csv(out / "03_DEG_LINEAGE/R4_LINEAGE_MATCHED_DEG_ALL.tsv.gz", sep="\t")
    frames: list[pd.DataFrame] = []
    for level, table in [("GLOBAL", global_deg), ("LINEAGE", lineage_deg)]:
        if table.empty:
            continue
        for keys, deg in table.groupby(["contrast", "comparison", "lineage"], dropna=False):
            contrast, comparison, lineage = keys
            for database, path in paths.items():
                result = ora_one(deg, database, load_gmt(path))
                if result.empty:
                    continue
                result.insert(0, "lineage", lineage)
                result.insert(0, "analysis_level", level)
                result.insert(0, "comparison", comparison)
                result.insert(0, "contrast", contrast)
                frames.append(result)
    all_ora = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[
        "contrast", "comparison", "analysis_level", "lineage", "direction", "database", "term",
        "query_n", "universe_n", "term_n", "overlap_n", "odds_ratio", "P", "genes", "FDR"
    ])
    go = all_ora[all_ora["database"] == "GO_BP"].copy()
    kegg = all_ora[all_ora["database"] == "KEGG"].copy()
    safe_to_csv(go, out / "04_GO_KEGG/R4_GO_BP_ORA.tsv")
    safe_to_csv(kegg, out / "04_GO_KEGG/R4_KEGG_ORA.tsv")
    log.info("GO rows=%d KEGG rows=%d", len(go), len(kegg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
