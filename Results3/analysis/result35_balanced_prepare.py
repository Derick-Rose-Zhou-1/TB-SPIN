from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


V5 = Path.cwd().resolve()
OUT = V5 / "26_FINAL_BALANCED_UP_DOWN_GO_KEGG"

DEG_SOURCES = {
    "H180": Path("04_H180_DEG/H180_HIGH_VS_BACKGROUND_ALL_GENES.tsv.gz"),
    "H365": Path("05_H365_DEG/H365_HIGH_VS_BACKGROUND_ALL_GENES.tsv.gz"),
    "COX": Path("06_COX_DEG/COX_HIGH_VS_BACKGROUND_ALL_GENES.tsv.gz"),
}
GMT_SOURCES = {
    "GO_BP": Path("08_MSIGDB/c5.go.bp.v2025.1.Hs.symbols.gmt"),
    "KEGG": Path("08_MSIGDB/c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt"),
}
STRICT_ANCHORS = {
    "H180": (69, 407),
    "H365": (140, 359),
    "COX": (313, 170),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False, na_rep="NA", lineterminator="\n")


def main() -> None:
    if OUT.exists():
        raise SystemExit("OUTPUT_EXISTS")
    for path in list(DEG_SOURCES.values()) + list(GMT_SOURCES.values()):
        if not (V5 / path).is_file():
            raise SystemExit(f"MISSING_INPUT={path}")

    subdirs = [
        "00_PROVENANCE",
        "01_GENESETS",
        "02_GO_BP_ORA",
        "03_KEGG_ORA",
        "04_ALL_ORA",
        "05_REDUNDANCY",
        "06_BALANCED_SELECTION",
        "07_FIGURE_SOURCE",
        "08_FIGURES",
        "09_TABLES",
        "10_METHODS",
        "11_MANUSCRIPT",
        "12_REPRODUCIBILITY",
        "13_REVIEW_PACKAGE",
        "scripts",
        "logs",
    ]
    for subdir in subdirs:
        (OUT / subdir).mkdir(parents=True, exist_ok=False)

    manifest_rows = []
    strict_rows = []
    count_rows = []
    input_hash_rows = []

    for axis, relative in DEG_SOURCES.items():
        source = V5 / relative
        frame = pd.read_csv(source, sep="\t")
        required = {"gene", "log2FC", "raw_pvalue", "FDR"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"DEG_SCHEMA_MISMATCH={relative}")
        frame = frame.copy()
        frame["gene"] = frame["gene"].astype(str).str.strip()
        frame = frame[frame["gene"].ne("") & frame["gene"].ne("nan")]
        frame["FDR"] = pd.to_numeric(frame["FDR"], errors="raise")
        frame["log2FC"] = pd.to_numeric(frame["log2FC"], errors="raise")
        frame["raw_pvalue"] = pd.to_numeric(frame["raw_pvalue"], errors="raise")
        frame = frame.sort_values(
            ["FDR", "log2FC", "gene"], ascending=[True, False, True], kind="mergesort"
        ).drop_duplicates("gene", keep="first")
        tested_n = int(frame["gene"].nunique())
        digest = sha256(source)
        manifest_rows.append(
            {
                "axis": axis,
                "source": str(relative),
                "sha256": digest,
                "row_n": len(frame),
                "tested_gene_n": tested_n,
            }
        )
        input_hash_rows.append(
            {"input_type": "DEG", "axis_or_database": axis, "file": str(relative), "sha256": digest}
        )

        strict_up = int(((frame["FDR"] < 0.05) & (frame["log2FC"] > 1.5)).sum())
        strict_down = int(((frame["FDR"] < 0.05) & (frame["log2FC"] < -1.5)).sum())
        anchor_up, anchor_down = STRICT_ANCHORS[axis]
        strict_rows.append(
            {
                "axis": axis,
                "strict_rule": "BH-FDR<0.05 and abs(log2FC)>1.5",
                "UP_actual": strict_up,
                "DOWN_actual": strict_down,
                "UP_expected_anchor": anchor_up,
                "DOWN_expected_anchor": anchor_down,
                "UP_anchor_match": strict_up == anchor_up,
                "DOWN_anchor_match": strict_down == anchor_down,
                "existing_strict_results_modified": False,
            }
        )

        universe = pd.DataFrame({"gene": sorted(frame["gene"].unique())})
        write_tsv(universe, OUT / "01_GENESETS" / f"{axis}_UNIVERSE.tsv")

        for direction in ["UP", "DOWN"]:
            positive = direction == "UP"
            direction_mask = frame["log2FC"].gt(0) if positive else frame["log2FC"].lt(0)
            lfc10_mask = frame["log2FC"].gt(1.0) if positive else frame["log2FC"].lt(-1.0)
            lfc15_mask = frame["log2FC"].gt(1.5) if positive else frame["log2FC"].lt(-1.5)
            fdr_mask = frame["FDR"].lt(0.05)
            genes = frame[fdr_mask & lfc10_mask].copy()
            genes["abs_log2FC"] = genes["log2FC"].abs()
            genes = genes.sort_values(
                ["FDR", "abs_log2FC", "gene"], ascending=[True, False, True], kind="mergesort"
            )
            columns = ["gene", "log2FC", "raw_pvalue", "FDR"]
            write_tsv(genes[columns], OUT / "01_GENESETS" / f"{axis}_{direction}_FDR05_LFC10.tsv")
            count_rows.append(
                {
                    "axis": axis,
                    "direction": direction,
                    "tested_gene_n": tested_n,
                    "FDR05_gene_n": int((fdr_mask & direction_mask).sum()),
                    "LFC10_gene_n": int((fdr_mask & lfc10_mask).sum()),
                    "LFC15_gene_n": int((fdr_mask & lfc15_mask).sum()),
                }
            )

    write_tsv(pd.DataFrame(manifest_rows), OUT / "00_PROVENANCE/DEG_SOURCE_MANIFEST.tsv")
    write_tsv(pd.DataFrame(strict_rows), OUT / "00_PROVENANCE/STRICT_LFC15_AUDIT.tsv")
    write_tsv(pd.DataFrame(count_rows), OUT / "01_GENESETS/GENESET_COUNTS.tsv")

    resource_rows = []
    for database, relative in GMT_SOURCES.items():
        source = V5 / relative
        digest = sha256(source)
        resource_rows.append(
            {
                "database": database,
                "MSIGDB_release": "2025.1.Hs",
                "source": str(relative),
                "sha256": digest,
                "size_bytes": source.stat().st_size,
            }
        )
        input_hash_rows.append(
            {
                "input_type": "MSIGDB_GMT",
                "axis_or_database": database,
                "file": str(relative),
                "sha256": digest,
            }
        )
    write_tsv(pd.DataFrame(resource_rows), OUT / "00_PROVENANCE/MSIGDB_RESOURCE_LOCK.tsv")
    write_tsv(pd.DataFrame(input_hash_rows), OUT / "12_REPRODUCIBILITY/INPUT_SHA256.tsv")



    print("PREPARE_COMPLETE")


if __name__ == "__main__":
    main()
