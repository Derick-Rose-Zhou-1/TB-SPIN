from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


RELEASE = Path(os.environ["TBSPIN_R3_RELEASE"])
OUT = Path(os.environ.get("RESULT35_TARGET_ROOT", "result3_expression"))
FORMAL_SOURCE = RELEASE / "CELL_PROJECTION/18_FINAL_RISK_CELL_LOCK_V4/05_FINAL_LABELS/TB_STATE_FORMAL_RISK_CELLS_V126.parquet"
RAW_SOURCE = RELEASE / "PROVENANCE/SHARED_INPUT_CACHE/00_INPUTS_LOCKED/single_cell/Tian_PBMC_16986genes_HVG3000_res2_QCclean_fullgenes.h5ad"
AXES = {
    "H180": {"contrast": "ALL_H180", "score": "U6", "label": "TB_STATE_HIGH_6M", "high": 48677,
             "dir": "04_H180_DEG"},
    "H365": {"contrast": "ALL_H365", "score": "U12", "label": "TB_STATE_HIGH_12M", "high": 65287,
             "dir": "05_H365_DEG"},
    "COX": {"contrast": "ALL_COX", "score": "UCOX", "label": "TB_STATE_HIGH_Cox", "high": 52718,
            "dir": "06_COX_DEG"},
}

DATABASES = {
    "GO_BP": "c5.go.bp.v2025.1.Hs.symbols.gmt",
    "GO_CC": "c5.go.cc.v2025.1.Hs.symbols.gmt",
    "GO_MF": "c5.go.mf.v2025.1.Hs.symbols.gmt",
    "KEGG": "c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt",
    "REACTOME": "c2.cp.reactome.v2025.1.Hs.symbols.gmt",
    "HALLMARK": "h.all.v2025.1.Hs.symbols.gmt",
}

DIRS = [
    "00_PROVENANCE", "01_INPUT_AUDIT", "02_RISK_ARCHITECTURE", "03_EXPRESSION",
    "04_H180_DEG", "05_H365_DEG", "06_COX_DEG", "07_THREE_AXIS_GENE_COMPARISON",
    "08_MSIGDB", "09_H180_GSEA", "10_H365_GSEA", "11_COX_GSEA",
    "12_GSEA_COMPARISON", "13_H180_ORA", "14_H365_ORA", "15_COX_ORA",
    "16_CORE_PROGRAMMES", "17_TABLES", "18_FIGURE_SOURCE", "19_FIGURES",
    "20_METHODS", "21_REPORT", "22_PROGRAMME_FREEZE", "23_REPRODUCIBILITY",
    "24_REVIEW_PACKAGE", "scripts", "logs",
]


def setup_dirs() -> None:
    for name in DIRS:
        (OUT / name).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(obj, rel: str) -> None:
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(df: pd.DataFrame, rel: str, *, index: bool = False) -> None:
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(sep="\t", index=index, na_rep="NA", float_format="%.10g", lineterminator="\n")
    if p.suffix == ".gz":
        kwargs["compression"] = {"method": "gzip", "mtime": 0}
    df.to_csv(p, **kwargs)


def load_formal() -> pd.DataFrame:
    d = pd.read_parquet(FORMAL_SOURCE).reset_index(drop=True)
    if "lineage" in d.columns and "major_lineage" not in d.columns:
        d = d.rename(columns={"lineage": "major_lineage"})
    for c in ["cell_id", "participant_id", "disease_group", "major_lineage"]:
        if c not in d.columns:
            raise RuntimeError(f"FORMAL_COLUMN_MISSING:{c}")
    if len(d) != 214402 or d.cell_id.duplicated().any():
        raise RuntimeError("FORMAL_CELL_LOCK_MISMATCH")
    for axis, cfg in AXES.items():
        high = d[cfg["label"]].astype(bool).to_numpy()
        if int(high.sum()) != cfg["high"]:
            raise RuntimeError(f"FORMAL_RISK_LABEL_MISMATCH:{axis}")
        if not np.array_equal(high, d[cfg["score"]].to_numpy(float) >= .95):
            raise RuntimeError(f"BOOLEAN_PERCENTILE_MISMATCH:{axis}")
    return d


def load_normalized() -> tuple[sparse.csr_matrix, np.ndarray]:
    x = sparse.load_npz(OUT / "03_EXPRESSION/FORMAL_LOG1P_CP10K_CSR_INTERNAL_CACHE.npz").tocsr()
    genes = np.load(OUT / "03_EXPRESSION/GENE_ORDER.npy", allow_pickle=True).astype(str)
    return x, genes
