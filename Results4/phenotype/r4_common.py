from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONTRASTS = {
    "C01": ("TB_NEGATIVE", "MTB_INFECTION", 1581, 435, 354933),
    "C02": ("MTB_INFECTION", "INCIPIENT_TB", 2034, 0, 354915),
    "C03": ("MTB_INFECTION", "ASYMPTOMATIC_TB", 5010, 8, 351931),
    "C04": ("MTB_INFECTION", "SYMPTOMATIC_TB", 3470, 9287, 344192),
    "C05": ("ASYMPTOMATIC_TB", "SYMPTOMATIC_TB", 437, 2250, 354262),
}

FULL_TASK_NAMES = {
    "C01": "C01_TBNEG_vs_G2_INFECTION",
    "C02": "C02_G2_INFECTION_vs_G3_INCIPIENT",
    "C03": "C03_G2_INFECTION_vs_ASYMPTOMATIC_TB",
    "C04": "C04_G2_INFECTION_vs_G6_SYMPTOMATIC",
    "C05": "C05_ASYMPTOMATIC_TB_vs_G6_SYMPTOMATIC",
}

MAJOR_CELLTYPES = ["B cell", "T cell", "NK cell", "Monocyte", "Neutrophil", "pDC"]


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_logging(log_file: Path, name: str) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bh_adjust(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    restored = np.empty_like(q)
    restored[order] = q
    out[ok] = restored
    return out


def read_scores(path: Path) -> pd.DataFrame:
    score = pd.read_csv(path, sep="\t", dtype={"cell_id": str})
    required = {
        "cell_id", "scPAS_RS", "scPAS_NRS", "scPAS_Pvalue", "scPAS_FDR", "scPAS_label"
    }
    missing = required - set(score.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if score["cell_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate cell IDs")
    return score


def score_path(config: dict, contrast: str) -> Path:
    return Path(config["score_paths"][contrast])


def manifest_path(config: dict, contrast: str) -> Path:
    return Path(config["manifest_paths"][contrast])


def safe_to_csv(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression=compression)


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def odds_ratio_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    vals = np.array([a, b, c, d], dtype=float)
    if (vals == 0).any():
        vals += 0.5
    aa, bb, cc, dd = vals
    odds = (a * d / (b * c)) if b > 0 and c > 0 else (aa * dd / (bb * cc))
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    lo = math.exp(math.log(aa * dd / (bb * cc)) - 1.96 * se)
    hi = math.exp(math.log(aa * dd / (bb * cc)) + 1.96 * se)
    return float(odds), float(lo), float(hi)


def save_gmt(gene_sets: dict[str, list[str]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for term in sorted(gene_sets):
            genes = sorted({str(g).upper() for g in gene_sets[term] if str(g).strip()})
            handle.write("\t".join([term, "NA", *genes]) + "\n")


def load_gmt(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                out[fields[0]] = {x.upper() for x in fields[2:] if x}
    return out


def celltype_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_field": ["celltype"] * len(MAJOR_CELLTYPES),
            "source_label": MAJOR_CELLTYPES,
            "canonical_major_celltype": MAJOR_CELLTYPES,
            "status": ["EXACT"] * len(MAJOR_CELLTYPES),
        }
    )
