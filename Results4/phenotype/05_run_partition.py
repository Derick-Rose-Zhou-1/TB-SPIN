from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from scpas_pycompat.r_rng import permutation_indices
from scpas_pycompat.scoring import score_cells
from scpas_pycompat.stats import pnorm_upper_abs


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_partition(h5_path: Path, lo: int, hi: int) -> sp.csc_matrix:
    with h5py.File(h5_path, "r") as handle:
        shape = tuple(int(x) for x in handle["shape"][:])
        indptr_all = handle["indptr"][:]
        start = int(indptr_all[lo])
        stop = int(indptr_all[hi])
        indptr = indptr_all[lo : hi + 1].astype(np.int64) - start
        indices = handle["indices"][start:stop].astype(np.int32)
        data = handle["data"][start:stop].astype(np.float64)
    return sp.csc_matrix((data, indices, indptr), shape=(shape[0], hi - lo))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("score", "permutation"), required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--scaling-stats", type=Path, required=True)
    parser.add_argument("--lo", type=int, required=True)
    parser.add_argument("--hi", type=int, required=True)
    parser.add_argument("--part", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    coefficients = np.load(args.coefficients)
    scaling = np.load(args.scaling_stats)
    means, sd = scaling["means"], scaling["sd"]
    expression = load_partition(args.h5, args.lo, args.hi)
    progress_path = args.out / f"PART_{args.part:02d}_PROGRESS.json"

    if args.mode == "score":
        rs, _, _ = score_cells(expression, coefficients, means=means, sd=sd, chunk_cells=8192)
        np.savez_compressed(args.out / f"SCORE_PART_{args.part:02d}.npz", rs=rs, lo=args.lo, hi=args.hi)
        atomic_json(progress_path, {"mode": "score", "part": args.part, "cells_completed": args.hi - args.lo,
                                   "status": "PASS", "elapsed_seconds": time.monotonic() - t0,
                                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        return 0

    def report(completed: int, total: int) -> None:
        if completed == total or completed // 500 != (completed - 32) // 500:
            atomic_json(progress_path, {"mode": "permutation", "part": args.part,
                                       "permutations_completed": completed,
                                       "permutations_total": total,
                                       "elapsed_seconds": time.monotonic() - t0,
                                       "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    # This is the validated independent=TRUE permutation recurrence applied to
    # a disjoint cell slice, using the frozen full-reference scaling statistics.
    # Every slice uses the same deterministic coefficient permutation for pi.
    rs, _, _ = score_cells(expression, coefficients, means=means, sd=sd, chunk_cells=2048)
    count = np.zeros(len(rs), dtype=np.int64)
    background_mean = np.zeros(len(rs), dtype=np.float64)
    m2 = np.zeros(len(rs), dtype=np.float64)
    for pi in range(1, args.permutations + 1):
        permuted = coefficients[permutation_indices(len(coefficients), pi)]
        value, _, _ = score_cells(expression, permuted, means=means, sd=sd, chunk_cells=2048)
        count += 1
        delta = value - background_mean
        background_mean += delta / count
        m2 += delta * (value - background_mean)
        if pi % 32 == 0 or pi == args.permutations:
            report(pi, args.permutations)
    background_sd = np.sqrt(m2 / (count - 1))
    nrs = (rs - background_mean) / background_sd
    pvalue = pnorm_upper_abs(nrs)
    np.savez_compressed(args.out / f"PERMUTATION_PART_{args.part:02d}.npz",
                        scPAS_RS=rs, background_mean=background_mean,
                        background_sd=background_sd, scPAS_NRS=nrs,
                        scPAS_Pvalue=pvalue, lo=args.lo, hi=args.hi)
    atomic_json(progress_path, {"mode": "permutation", "part": args.part,
                               "permutations_completed": args.permutations,
                               "permutations_total": args.permutations, "status": "PASS",
                               "elapsed_seconds": time.monotonic() - t0,
                               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
