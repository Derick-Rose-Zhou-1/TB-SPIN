#!/usr/bin/env python3
"""Score all Tian PBMC cells for one Result4 V5 scPAS final model."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
from scipy.special import ndtr


def ts() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def emit(task: str, stage: str, status: str, message: str) -> None:
    print(f"[{ts()}] [INFO] task={task} method=scPAS_v5_allcell stage={stage} status={status} {message}", flush=True)


def bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(ranked, 1.0)
    return out


def write_status(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("timestamp", ts())
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--v5-root", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--permutations", type=int, default=100)
    ap.add_argument("--chunk-size", type=int, default=10000)
    args = ap.parse_args()

    root = Path(args.root)
    v5 = Path(args.v5_root)
    task = args.task
    model_dir = v5 / "08_SCPAS" / task
    outdir = v5 / "10_ALL_CELL_SCORING" / task
    outdir.mkdir(parents=True, exist_ok=True)
    status_path = outdir / "ALL_CELL_SCORE_STATUS.json"
    started_wall = time.time()
    print(f"[{ts()}] [STEP START] V5_ALL_CELL_SCORE | {task}", flush=True)
    write_status(status_path, {"task": task, "status": "RUNNING", "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET")})

    coef_path = model_dir / "FINAL_MODEL_COEFFICIENTS.tsv"
    scaling_path = root / "07_SCPAS/NETWORK/SC_GENE_SCALING.tsv"
    h5ad_path = root / "01_INPUT/single_cell/Tian_PBMC_16986genes_HVG3000_res2_QCclean_fullgenes_RDS356949_6group.h5ad"
    if not coef_path.exists():
        raise FileNotFoundError(f"missing final coefficients: {coef_path}")
    if not scaling_path.exists():
        raise FileNotFoundError(f"missing single-cell scaling: {scaling_path}")
    if not h5ad_path.exists():
        raise FileNotFoundError(f"missing h5ad: {h5ad_path}")

    coef = pd.read_csv(coef_path, sep="\t")
    coef = coef[coef["gene"] != "(Intercept)"].copy()
    coef = coef[coef["coefficient"] != 0].copy()
    scaling = pd.read_csv(scaling_path, sep="\t").set_index("gene")
    genes = [g for g in coef["gene"].astype(str) if g in scaling.index]

    if not genes:
        emit(task, "ALL_CELL_SCORE", "WARNING", "NULL_GENE_MODEL_NO_CELL_LOCALIZATION; exporting auditable all-background output")
        a = ad.read_h5ad(h5ad_path, backed="r")
        try:
            cell_id = a.obs_names.astype(str).to_numpy()
            six_group = a.obs["celltype_six_group"].astype(str).to_numpy()
            sample_id = a.obs["sample_id"].astype(str).to_numpy()
        finally:
            a.file.close()
        n = len(cell_id)
        result = pd.DataFrame({
            "cell_id": cell_id, "six_group": six_group, "donor": "NOT_AVAILABLE",
            "sample": sample_id, "task": task, "continuous_score": np.zeros(n),
            "normalized_score": np.zeros(n), "permutation_P": np.ones(n),
            "FDR": np.ones(n), "scPAS_class": "0",
            "model_fold_or_final": "FINAL_DISCOVERY_V5_NULL_GENE_MODEL",
        })
        result.to_csv(outdir / "ALL_CELL_SCORES.tsv.gz", sep="\t", index=False, compression="gzip")
        pd.DataFrame([{
            "task": task, "permutations": args.permutations, "seed": 12345,
            "official_rule": "NOT_APPLICABLE_NULL_GENE_MODEL",
            "positive_cells": 0, "negative_cells": 0, "background_cells": n,
            "nonzero_model_genes": 0, "gpu_physical": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
            "status": "NULL_GENE_MODEL_NO_CELL_LOCALIZATION", "runtime_seconds": time.time() - started_wall,
        }]).to_csv(outdir / "PERMUTATION_SUMMARY.tsv", sep="\t", index=False)
        write_status(status_path, {"task": task, "status": "COMPLETE", "rows": n, "nonzero_model_genes": 0})
        print(f"[{ts()}] [STEP DONE] V5_ALL_CELL_SCORE | {task} NULL_MODEL_EXPORT_COMPLETE", flush=True)
        return 0

    beta_map = coef.set_index("gene")["coefficient"]
    beta = beta_map.loc[genes].to_numpy(dtype=np.float64)
    means = scaling.loc[genes, "mean"].to_numpy(dtype=np.float64)
    sds = scaling.loc[genes, "sd"].to_numpy(dtype=np.float64)
    valid = np.isfinite(sds) & (sds > 0)
    genes = list(np.asarray(genes)[valid])
    beta = beta[valid]
    means = means[valid]
    sds = sds[valid]
    if not genes:
        raise RuntimeError("no genes remain after valid single-cell scaling filter")

    emit(task, "ALL_CELL_SCORE", "STARTING",
         f"cells=356949 genes={len(genes)} permutations={args.permutations} chunk_size={args.chunk_size} gpu={os.environ.get('CUDA_VISIBLE_DEVICES','UNSET')} rss_gib={psutil.Process().memory_info().rss/(1<<30):.3f}")

    rng = np.random.default_rng(12345)
    perm_beta = np.column_stack([rng.permutation(beta) for _ in range(args.permutations)])
    a = ad.read_h5ad(h5ad_path, backed="r")
    try:
        var_genes = a.var["gene_symbol"].astype(str).to_numpy()
        lookup = {g: i for i, g in enumerate(var_genes)}
        genes = [g for g in genes if g in lookup]
        if not genes:
            raise RuntimeError("no final-model genes found in h5ad var['gene_symbol']")
        beta = beta_map.loc[genes].to_numpy(dtype=np.float64)
        means = scaling.loc[genes, "mean"].to_numpy(dtype=np.float64)
        sds = scaling.loc[genes, "sd"].to_numpy(dtype=np.float64)
        idx = np.asarray([lookup[g] for g in genes], dtype=int)
        n = a.n_obs
        cell_id = a.obs_names.astype(str).to_numpy()
        six_group = a.obs["celltype_six_group"].astype(str).to_numpy()
        sample_id = a.obs["sample_id"].astype(str).to_numpy()
        raw_score = np.empty(n, dtype=np.float64)
        zscore = np.empty(n, dtype=np.float64)
        pvalue = np.empty(n, dtype=np.float64)
        g_beta = cp.asarray(beta)
        g_perm = cp.asarray(np.column_stack([rng.permutation(beta) for _ in range(args.permutations)]))
        g_means = cp.asarray(means)
        g_sds = cp.asarray(sds)
        chunks = (n + args.chunk_size - 1) // args.chunk_size
        for ci, start in enumerate(range(0, n, args.chunk_size), 1):
            stop = min(start + args.chunk_size, n)
            host = a.X[start:stop, :]
            host = host[:, idx]
            if sp.issparse(host):
                host = host.toarray()
            x = cp.asarray(np.asarray(host, dtype=np.float32))
            zexpr = (x - g_means) / g_sds
            score = zexpr @ g_beta
            background = zexpr @ g_perm
            bg_mean = background.mean(axis=1)
            bg_sd = background.std(axis=1, ddof=1)
            z = cp.where(bg_sd > 0, (score - bg_mean) / bg_sd, 0.0)
            raw_score[start:stop] = cp.asnumpy(score)
            zscore[start:stop] = cp.asnumpy(z)
            pvalue[start:stop] = ndtr(-np.abs(zscore[start:stop]))
            elapsed = time.time() - started_wall
            emit(task, "ALL_CELL_SCORE_CHUNK", "RUNNING",
                 f"chunk={ci}/{chunks} cells_completed={stop}/{n} percentage={100*stop/n:.2f} rss_gib={psutil.Process().memory_info().rss/(1<<30):.3f} gpu_used_gib={cp.get_default_memory_pool().used_bytes()/(1<<30):.3f} elapsed_s={elapsed:.1f} eta_s={elapsed/max(stop,1)*(n-stop):.1f}")
            del x, zexpr, score, background, bg_mean, bg_sd, z, host
            cp.get_default_memory_pool().free_all_blocks()
    finally:
        a.file.close()

    fdr = bh(pvalue)
    label = np.where((zscore > 0) & (fdr <= 0.05), "scPAS+", np.where((zscore < 0) & (fdr <= 0.05), "scPAS-", "0"))
    result = pd.DataFrame({
        "cell_id": cell_id, "six_group": six_group, "donor": "NOT_AVAILABLE",
        "sample": sample_id, "task": task, "continuous_score": raw_score,
        "normalized_score": zscore, "permutation_P": pvalue, "FDR": fdr,
        "scPAS_class": label, "model_fold_or_final": "FINAL_DISCOVERY_V5",
    })
    result.to_csv(outdir / "ALL_CELL_SCORES.tsv.gz", sep="\t", index=False, compression="gzip")
    group = result.groupby("six_group", observed=True).agg(
        total_cells=("cell_id", "size"),
        mean_score=("continuous_score", "mean"),
        median_score=("continuous_score", "median"),
        positive_associated_fraction=("scPAS_class", lambda x: np.mean(x == "scPAS+")),
        negative_associated_fraction=("scPAS_class", lambda x: np.mean(x == "scPAS-")),
        sample_support=("sample", "nunique"),
    ).reset_index()
    max_frac = result.groupby("six_group", observed=True)["sample"].value_counts(normalize=True).groupby(level=0).max()
    group["max_sample_fraction"] = group["six_group"].map(max_frac)
    group["donor_support"] = "NOT_AVAILABLE"
    group.to_csv(outdir / "CELL_GROUP_SUMMARY.tsv", sep="\t", index=False)
    runtime = time.time() - started_wall
    pd.DataFrame([{
        "task": task, "permutations": args.permutations, "seed": 12345,
        "official_rule": "coefficient_permutation;one_tail_normal_abs_Z;BH_FDR",
        "positive_cells": int(np.sum(label == "scPAS+")),
        "negative_cells": int(np.sum(label == "scPAS-")),
        "background_cells": int(np.sum(label == "0")),
        "nonzero_model_genes": len(genes),
        "gpu_physical": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
        "runtime_seconds": runtime,
    }]).to_csv(outdir / "PERMUTATION_SUMMARY.tsv", sep="\t", index=False)
    write_status(status_path, {
        "task": task, "status": "COMPLETE", "rows": int(n),
        "positive_cells": int(np.sum(label == "scPAS+")),
        "negative_cells": int(np.sum(label == "scPAS-")),
        "background_cells": int(np.sum(label == "0")),
        "nonzero_model_genes": len(genes),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
        "runtime_seconds": runtime,
    })
    emit(task, "ALL_CELL_SCORE", "COMPLETE",
         f"positive={np.sum(label=='scPAS+')} negative={np.sum(label=='scPAS-')} background={np.sum(label=='0')} runtime_s={runtime:.1f}")
    print(f"[{ts()}] [STEP DONE] V5_ALL_CELL_SCORE | {task} COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
