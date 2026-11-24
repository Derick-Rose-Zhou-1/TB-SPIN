from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.optimize import minimize_scalar
from scipy.stats import norm


SEED = 20260920
ANALYSIS_SCOPE = 'DESCRIPTIVE_GLOBAL_LABEL_TRANSFER'
THRESHOLDS = {'R1':1.0, 'R2':1.5}
X_GLOBAL = None
LABEL_TASKS_GLOBAL = None

def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    good = np.isfinite(p)
    if not good.any():
        return out
    q = p[good]
    order = np.argsort(q, kind="stable")
    ranked = q[order] * len(q) / np.arange(1, len(q) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(ranked)
    restored[order] = np.minimum(ranked, 1.0)
    out[good] = restored
    return out

def build_deg_table(genes, p, log2fc, mean_high, mean_background) -> pd.DataFrame:
    fdr = bh_fdr(np.asarray(p, dtype=float))
    lfc = np.asarray(log2fc, dtype=float)
    frame = pd.DataFrame({
        "gene": np.asarray(genes, dtype=str),
        "p_value": np.asarray(p, dtype=float),
        "fdr": fdr,
        "log2fc": lfc,
        "mean_risk_cp10k": np.asarray(mean_high, dtype=float),
        "mean_background_cp10k": np.asarray(mean_background, dtype=float),
    })
    frame["R1_selected"] = np.isfinite(fdr) & (fdr < 0.05) & (lfc >= THRESHOLDS["R1"])
    frame["R2_selected"] = np.isfinite(fdr) & (fdr < 0.05) & (lfc >= THRESHOLDS["R2"])
    return frame.sort_values(["fdr", "log2fc", "gene"], ascending=[True, False, True], kind="stable")

def positive_mask(method: str, endpoint: str, values: pd.Series) -> np.ndarray:
    s = pd.Series(values)
    if method == "TBSTATE":
        return s.astype(str).str.lower().isin({"true", "1", "1.0"}).to_numpy()
    if method == "SCAB":
        return s.astype(str).eq("scAB+ cells").to_numpy()
    if method == "SCISSOR":
        return s.astype(str).isin({"Scissor+", "1", "1.0"}).to_numpy()
    if method == "SCPAS":
        return s.astype(str).eq("scPAS+").to_numpy()
    if method == "DEGAS":
        return s.astype(str).eq("DEGAS+").to_numpy()
    if method == "SIDISH":
        return s.astype(str).isin({"1", "1.0", "True", "true"}).to_numpy()
    raise KeyError((method, endpoint))

def loco_mean_z_scores(expr: pd.DataFrame, cohorts, genes) -> tuple[np.ndarray, int]:
    cohorts = np.asarray(cohorts).astype(str)
    use = [g for g in genes if g in expr.columns]
    scores = np.full(len(expr), np.nan, dtype=float)
    if not use:
        return scores, 0
    x = expr.loc[:, use].astype(float)
    usable_any = set()
    for cohort in sorted(np.unique(cohorts)):
        test = cohorts == cohort
        train = ~test
        if train.sum() == 0:
            continue
        mu = x.loc[train].mean(axis=0)
        sd = x.loc[train].std(axis=0, ddof=0)
        valid = sd.gt(0) & np.isfinite(sd) & np.isfinite(mu)
        if not valid.any():
            continue
        usable_any.update(np.asarray(use)[valid.to_numpy()].tolist())
        z = (x.loc[test, valid.index[valid]] - mu[valid]) / sd[valid]
        scores[test] = z.mean(axis=1, skipna=True).to_numpy()
    return scores, len(usable_any)

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t")

def align_values(cells: np.ndarray, path: Path, value_col: str) -> np.ndarray:
    frame = read_table(path)[["cell_id", value_col]]
    if len(frame) != len(cells) or not frame.cell_id.is_unique:
        raise ValueError(f"invalid label cardinality/uniqueness: {path}")
    ids = frame.cell_id.astype(str).to_numpy()
    if not np.array_equal(ids, cells):
        aligned = pd.DataFrame({"cell_id": cells}).merge(frame, on="cell_id", how="left", validate="one_to_one")
        if aligned[value_col].isna().any():
            raise ValueError(f"missing labels after exact cell join: {path}")
        frame = aligned
    return frame[value_col].to_numpy()

def input_paths(formal_root: Path, scpas_root: Path, degas_root: Path, sidish_path: Path):
    return {
        "formal_counts": formal_root / "00_INPUT/FORMAL_RAW_COUNTS.npz",
        "genes": formal_root / "00_INPUT/GENES.txt",
        "universe": formal_root / "00_INPUT/FORMAL_CELL_UNIVERSE.tsv",
        "bulk": formal_root / "00_INPUT/BULK_LOG2CPM.tsv",
        "baseline": formal_root / "00_INPUT/BASELINE_ENDPOINTS.tsv",
        "cox_baseline": formal_root / "00_INPUT/COX_BASELINE_ENDPOINTS.tsv",
        "scab_h1": formal_root / "03_SCAB/SCAB_COX_214402.tsv",
        "scab_h2": formal_root / "03_SCAB/SCAB_H2_214402.tsv",
        "scab_h3": formal_root / "03_SCAB/SCAB_H3_214402.tsv",
        "scpas_h1": scpas_root / "05_SCPAS/SCPAS_COX_214402.tsv",
        "scpas_h2": scpas_root / "05_SCPAS/SCPAS_H2_214402.tsv",
        "scpas_h3": scpas_root / "05_SCPAS/SCPAS_H3_214402.tsv",
        "degas_h2": degas_root / "04_DEGAS/DEGAS_H2_214402.tsv",
        "degas_h3": degas_root / "04_DEGAS/DEGAS_H3_214402.tsv",
        "sidish_h1": sidish_path,
    }

def load_label_tasks(paths, cells):
    specs = [
        ("SCAB", "H1", paths["scab_h1"], "native_state_if_available"),
        ("SCAB", "H2", paths["scab_h2"], "native_state_if_available"),
        ("SCAB", "H3", paths["scab_h3"], "native_state_if_available"),
        ("SCPAS", "H1", paths["scpas_h1"], "scPAS"),
        ("SCPAS", "H2", paths["scpas_h2"], "scPAS"),
        ("SCPAS", "H3", paths["scpas_h3"], "scPAS"),
        ("DEGAS", "H2", paths["degas_h2"], "native_label"),
        ("DEGAS", "H3", paths["degas_h3"], "native_label"),
        ("SIDISH", "H1", paths["sidish_h1"], "native_high_risk"),
    ]
    tasks = []
    for method, endpoint, path, col in specs:
        values = align_values(cells, path, col)
        mask = positive_mask(method, endpoint, pd.Series(values))
        if mask.sum() == 0 or mask.sum() == len(mask):
            raise ValueError(f"single-class native label: {method} {endpoint}")
        tasks.append({"method": method, "endpoint": endpoint, "path": path, "column": col, "positive": mask})
    return tasks

def _marker_block(bounds):
    lo, hi = bounds
    block = X_GLOBAL[:, lo:hi].toarray()
    linear = np.expm1(block)
    results = []
    for task in LABEL_TASKS_GLOBAL:
        high = task["positive"]
        background = ~high
        test = stats.mannwhitneyu(
            block[high], block[background], axis=0,
            alternative="two-sided", method="asymptotic", use_continuity=True,
        )
        ma = linear[high].mean(axis=0)
        mb = linear[background].mean(axis=0)
        lfc = np.log2((ma + 1e-9) / (mb + 1e-9))
        results.append((np.asarray(test.pvalue), lfc, ma, mb))
    return lo, hi, results

def compute_all_degs(count_path, genes, tasks, workers=8, block_size=64):
    global X_GLOBAL, LABEL_TASKS_GLOBAL
    counts = sparse.load_npz(count_path).tocsr()
    if counts.shape != (214402, len(genes)):
        raise ValueError(f"raw count shape mismatch: {counts.shape}, genes={len(genes)}")
    lib = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(lib <= 0):
        raise ValueError("non-positive single-cell library size")
    X_GLOBAL = (sparse.diags(10000.0 / lib) @ counts).tocsr()
    X_GLOBAL.data = np.log1p(X_GLOBAL.data)
    LABEL_TASKS_GLOBAL = tasks
    n_tasks, n_genes = len(tasks), len(genes)
    arrays = [np.empty((n_tasks, n_genes), dtype=np.float64) for _ in range(4)]
    blocks = [(lo, min(lo + block_size, n_genes)) for lo in range(0, n_genes, block_size)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for bi, (lo, hi, result) in enumerate(pool.map(_marker_block, blocks), 1):
            for ti, values in enumerate(result):
                for ai, value in enumerate(values):
                    arrays[ai][ti, lo:hi] = value
            if bi % 25 == 0 or bi == len(blocks):
                print(f"DEG_BLOCK {bi}/{len(blocks)} genes={hi}", flush=True)
    X_GLOBAL = None
    LABEL_TASKS_GLOBAL = None
    return arrays

def load_baseline(paths):
    bulk = pd.read_csv(paths["bulk"], sep="\t").rename(columns={"sample_id": "GSM"}).set_index("GSM")
    bulk.index = bulk.index.astype(str)
    binary = pd.read_csv(paths["baseline"], sep="\t")
    cox = pd.read_csv(paths["cox_baseline"], sep="\t")
    for frame in (binary, cox):
        frame["GSM"] = frame.GSM.astype(str)
        frame["participant_id"] = frame.participant_id.astype(str)
        if frame.participant_id.duplicated().any():
            # Duplicate participants are allowed across endpoint rows, but not within an endpoint.
            for endpoint, d in frame.groupby("endpoint"):
                if d.participant_id.duplicated().any():
                    raise ValueError(f"more than one baseline sample per participant: {endpoint}")
        if not frame.GSM.isin(bulk.index).all():
            raise ValueError("baseline sample missing from bulk expression")
    expected = {"E_6M": 412, "E_12M": 377}
    if binary.groupby("endpoint").size().to_dict() != expected or len(cox) != 426:
        raise ValueError("baseline endpoint cardinality mismatch")
    return bulk, binary, cox

def endpoint_meta(endpoint, binary, cox):
    if endpoint == "H1":
        d = cox.copy()
    else:
        key = "E_6M" if endpoint == "H2" else "E_12M"
        d = binary.loc[binary.endpoint.eq(key)].copy()
    if "cohort" not in d:
        raise ValueError("baseline mapping lacks cohort")
    d = d.reset_index(drop=True)
    d["endpoint_short"] = endpoint
    d["time"] = pd.to_numeric(d.get("time_to_event_or_censor_days"), errors="coerce")
    d["y_true"] = pd.to_numeric(d.y_true, errors="raise").astype(int)
    return d

def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "DEG_TABLES").mkdir(exist_ok=False)
    paths = input_paths(args.formal_root, args.scpas_root, args.degas_root, args.sidish)
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing inputs: " + "; ".join(missing))
    universe = pd.read_csv(paths["universe"], sep="\t")
    cells = universe.cell_id.astype(str).to_numpy()
    if len(cells) != 214402 or universe.cell_id.duplicated().any():
        raise ValueError("formal universe is not 214402 unique cells")
    genes = pd.read_csv(paths["genes"], header=None)[0].astype(str).to_numpy()
    if pd.Index(genes).duplicated().any():
        raise ValueError("duplicate genes in formal raw-count matrix")
    tasks = load_label_tasks(paths, cells)

    pvals, lfc, means_high, means_background = compute_all_degs(
        paths["formal_counts"], genes, tasks, workers=args.workers, block_size=args.block_size)
    task_tables = {}
    for i, task in enumerate(tasks):
        method, endpoint = task["method"], task["endpoint"]
        table = build_deg_table(genes, pvals[i], lfc[i], means_high[i], means_background[i])
        table.insert(0, "endpoint", endpoint)
        table.insert(0, "method", method)
        table.to_csv(out / "DEG_TABLES" / f"{method}_{endpoint}_ALL_GENES.tsv.gz",
                     sep="\t", index=False, compression="gzip")
        task_tables[(method, endpoint)] = table
    bulk, binary_meta, cox_meta = load_baseline(paths)
    patient_rows = []
    for task in tasks:
        method, endpoint = task["method"], task["endpoint"]
        meta = endpoint_meta(endpoint, binary_meta, cox_meta)
        expr = bulk.reindex(meta.GSM.astype(str)).reset_index(drop=True)
        for threshold in ('R1',):
            table = task_tables[(method, endpoint)]
            selected = table.loc[table[f"{threshold}_selected"], "gene"].astype(str).tolist()
            score, usable = loco_mean_z_scores(expr, meta.cohort.astype(str).to_numpy(), selected)
            scored = meta[["participant_id", "GSM", "cohort", "y_true", "time"]].copy()
            scored["method"] = method
            scored["endpoint"] = endpoint
            scored["threshold"] = threshold
            scored["score"] = score
            scored["n_degs_selected"] = len(selected)
            scored["n_degs_bulk_usable"] = usable
            scored["analysis_scope"] = ANALYSIS_SCOPE
            patient_rows.append(scored)
    pd.concat(patient_rows, ignore_index=True).to_csv(out / "PATIENT_SCORES.tsv.gz", sep="\t", index=False)

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--formal-root", type=Path, required=True)
    p.add_argument("--scpas-root", type=Path, required=True)
    p.add_argument("--degas-root", type=Path, required=True)
    p.add_argument("--sidish", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--block-size", type=int, default=64)
    return p.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
