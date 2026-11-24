from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scpas_pycompat.api import DEFAULT_ALPHA
from scpas_pycompat.apml0 import apml0
from scpas_pycompat.normalize import quantile_normalize
from scpas_pycompat.stats import bh_adjust, labels as scpas_labels


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def status(path: Path, lines: list[str]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def rss_gib() -> float:
    try:
        fields = Path("/proc/self/status").read_text().splitlines()
        return float(next(x.split()[1] for x in fields if x.startswith("VmRSS:"))) / 1024 / 1024
    except Exception:
        return float("nan")


def progress(out: Path, task: str, stage: str, started: float, **extra) -> None:
    value = {"task": task, "stage": stage, "elapsed_seconds": time.monotonic() - started,
             "rss_gib": rss_gib(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    value.update(extra)
    atomic_json(out / "PROGRESS.json", value)
    print("PROGRESS=" + json.dumps(value, sort_keys=True), flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_cell_rs(out: Path, cell_ids: list[str], part_dir: Path, n_parts: int) -> None:
    with gzip.open(out / "CELL_RS.tsv.gz", "wt", encoding="utf-8") as handle:
        handle.write("cell_id\tscPAS_RS\n")
        offset = 0
        for part in range(n_parts):
            z = np.load(part_dir / f"SCORE_PART_{part:02d}.npz")
            values = z["rs"]
            for cell, value in zip(cell_ids[offset:offset + len(values)], values):
                handle.write(f"{cell}\t{value:.17g}\n")
            offset += len(values)
    if offset != len(cell_ids):
        raise RuntimeError("cell scoring partition count mismatch")


def write_cell_scores(out: Path, cell_ids: list[str], part_dir: Path, n_parts: int, label_column: str = "scPAS_label") -> None:
    columns = ("scPAS_RS", "scPAS_NRS", "scPAS_Pvalue")
    pvalues = np.concatenate([np.load(part_dir / f"PERMUTATION_PART_{part:02d}.npz")["scPAS_Pvalue"]
                              for part in range(n_parts)])
    nrs_all = np.concatenate([np.load(part_dir / f"PERMUTATION_PART_{part:02d}.npz")["scPAS_NRS"]
                              for part in range(n_parts)])
    fdr_all = bh_adjust(pvalues)
    labels_all = scpas_labels(nrs_all, fdr_all, 0.05)
    with gzip.open(out / "CELL_SCORES.tsv.gz", "wt", encoding="utf-8") as handle:
        handle.write(f"cell_id\tscPAS_RS\tscPAS_NRS\tscPAS_Pvalue\tscPAS_FDR\t{label_column}\n")
        offset = 0
        for part in range(n_parts):
            z = np.load(part_dir / f"PERMUTATION_PART_{part:02d}.npz")
            arrays = [z[name] for name in columns]
            length = len(arrays[0])
            for i, cell in enumerate(cell_ids[offset:offset + length]):
                handle.write(cell + "\t" + "\t".join(f"{a[i]:.17g}" for a in arrays) +
                             f"\t{fdr_all[offset+i]:.17g}\t{labels_all[offset+i]}\n")
            offset += length
    if offset != len(cell_ids):
        raise RuntimeError("permutation partition count mismatch")


def run_partitions(run: Path, out: Path, task: str, mode: str, n_cells: int, n_parts: int,
                   coefficients: Path, started: float) -> None:
    part_dir = out / "partition_checkpoints"
    part_dir.mkdir(exist_ok=True)
    script = Path(__file__).with_name("05_run_partition.py")
    h5 = run / "02_FROZEN_INPUTS/SINGLE_CELL_COMMON_CSC.h5"
    scaling_stats = run / "02_FROZEN_INPUTS/SCALING_STATS.npz"
    children = []
    for part in range(n_parts):
        lo = n_cells * part // n_parts
        hi = n_cells * (part + 1) // n_parts
        target = part_dir / (f"SCORE_PART_{part:02d}.npz" if mode == "score" else f"PERMUTATION_PART_{part:02d}.npz")
        if target.exists():
            continue
        log = (out / f"{mode}_part_{part:02d}.log").open("ab")
        cmd = [sys.executable, str(script), "--mode", mode, "--h5", str(h5),
               "--coefficients", str(coefficients), "--scaling-stats", str(scaling_stats),
               "--lo", str(lo), "--hi", str(hi),
               "--part", str(part), "--out", str(part_dir), "--permutations", "5000"]
        children.append((part, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    while children:
        failed = []
        for part, child, log in list(children):
            code = child.poll()
            if code is not None:
                log.close()
                children.remove((part, child, log))
                if code != 0:
                    failed.append((part, code))
        if failed:
            for _, child, _ in children:
                child.terminate()
            raise RuntimeError(f"{mode} partition failure: {failed}")
        if mode == "permutation":
            values = []
            for part in range(n_parts):
                p = part_dir / f"PART_{part:02d}_PROGRESS.json"
                try:
                    v = json.loads(p.read_text())
                    if v.get("mode") == "permutation": values.append(int(v.get("permutations_completed", 0)))
                except Exception:
                    values.append(0)
            completed = min(values) if len(values) == n_parts else 0
            progress(out, task, "PERMUTATION", started, permutations_completed=completed,
                     permutations_total=5000, permutation_workers=n_parts)
        else:
            progress(out, task, "CELL_SCORING", started,
                     scoring_parts_completed=n_parts - len(children), scoring_parts_total=n_parts)
        if children:
            time.sleep(15)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("task")
    args = parser.parse_args()
    run, task = args.run.resolve(), args.task
    out = run / "05_PRIMARY_TASKS" / task
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status_file = out / "STATUS.txt"
    retry_marker = out / "WORKER_STARTS.log"
    with retry_marker.open("a", encoding="utf-8") as handle:
        handle.write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + f"\tpid={os.getpid()}\n")
    try:
        registry = pd.read_csv(run / "04_TASK_REGISTRY/TASK_REGISTRY.tsv", sep="\t")
        row = registry.loc[registry.task == task]
        if len(row) != 1:
            raise RuntimeError("task is not unique in frozen registry")
        row = row.iloc[0]
        if str(row.ESTIMABLE).strip().lower() not in {"true", "1"}:
            status(status_file, ["STATUS=NOT_ESTIMABLE", "ERROR_STAGE=FROZEN_TASK_REGISTRY",
                                 "ERROR=FROZEN_REGISTRY_LACKS_BOTH_CLASSES"])
            progress(out, task, "TERMINAL", started, status="NOT_ESTIMABLE")
            return 0
        manifest = pd.read_csv(run / f"04_TASK_REGISTRY/TASK_SAMPLE_MANIFEST_{task}.tsv", sep="\t")
        y = manifest.phenotype.to_numpy(dtype=float)
        if sorted(np.unique(y).tolist()) != [0.0, 1.0]:
            raise RuntimeError("frozen phenotype is not exact binary 0/1")
        if manifest.sample_id.duplicated().any():
            raise RuntimeError("duplicate sample ID in frozen manifest")
        checkpoint = out / "MODEL_CHECKPOINT.pkl"
        coefficient_npy = out / "MODEL_COEFFICIENTS.npy"
        if not checkpoint.exists():
            progress(out, task, "INPUT_LOAD", started, alpha_completed=0, alpha_total=13)
            task_bulk = run / "02_FROZEN_INPUTS" / f"{task}_BULK_COMMON_LOGCPM.tsv.gz"
            bulk_path = task_bulk if task_bulk.exists() else run / "02_FROZEN_INPUTS/SCPAS_BULK_COMMON_LOGCPM.tsv.gz"
            bulk = pd.read_csv(bulk_path, sep="\t", index_col=0)
            samples = manifest.sample_id.astype(str).tolist()
            missing = sorted(set(samples) - set(bulk.columns))
            if missing:
                raise RuntimeError(f"bulk samples missing: {missing[:5]}")
            genes = bulk.index.astype(str).to_numpy(dtype=object)
            cache_genes = np.array((run / "02_FROZEN_INPUTS/SC_CACHE_GENES.txt").read_text().splitlines(), dtype=object)
            if not np.array_equal(genes, cache_genes):
                raise RuntimeError("frozen common gene order mismatch")
            expression_bulk = quantile_normalize(bulk.loc[:, samples].to_numpy(dtype=float))
            x = np.asarray(expression_bulk.T, dtype=float, order="F")
            del bulk, expression_bulk
            network = sp.load_npz(run / "03_NETWORK_CACHE/NETWORK.npz").toarray()
            partial_path = out / "ALPHA_PARTIAL_CHECKPOINT.pkl"
            if partial_path.exists():
                with partial_path.open("rb") as handle: partial = pickle.load(handle)
                lambdas = list(partial["lambdas"]); coefficients = partial["coefficients"]
                start_alpha = len(lambdas)
            else:
                lambdas, coefficients, start_alpha = [], None, 0
            trajectory_path = out / "ALPHA_LAMBDA_TRAJECTORY.tsv"
            for index in range(start_alpha, len(DEFAULT_ALPHA)):
                alpha = float(DEFAULT_ALPHA[index])
                progress(out, task, "ALPHA_SEARCH", started, alpha_completed=index,
                         alpha_total=len(DEFAULT_ALPHA), current_alpha=alpha, current_lambda=None)
                fit0 = apml0(x=x, y=y, family="binomial", penalty="Net", Omega=network,
                             alpha=alpha, nlambda=100, nfolds=min(10, len(x)),
                             iL0=True, icutB=False, cv_seed=123)
                selected_lambda = float(fit0.lambda_min)
                fit1 = apml0(x=x, y=y, family="binomial", penalty="Net", Omega=network,
                             alpha=alpha, lambda_=np.array([selected_lambda]),
                             iL0=True, icutB=False)
                beta = np.asarray(fit1.Beta)
                coefficients = beta[1:, 0] if beta.ndim == 2 else beta[1:]
                lambdas.append(selected_lambda)
                with partial_path.open("wb") as handle:
                    pickle.dump({"lambdas": lambdas, "coefficients": coefficients}, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
                pd.DataFrame({"alpha": DEFAULT_ALPHA[:len(lambdas)], "lambda": lambdas}).to_csv(
                    trajectory_path, sep="\t", index=False)
                progress(out, task, "ALPHA_SEARCH", started, alpha_completed=index + 1,
                         alpha_total=len(DEFAULT_ALPHA), current_alpha=alpha,
                         current_lambda=selected_lambda)
                del fit0, fit1, beta
                gc.collect()
            np.save(coefficient_npy, coefficients)
            coefficient_sha = hashlib.sha256(np.asarray(coefficients, dtype=np.float64).tobytes()).hexdigest()
            network_sha = (run / "03_NETWORK_CACHE/NETWORK_SHA256.txt").read_text().strip()
            model = {"task": task, "common_genes": genes, "sample_ids": samples,
                     "phenotype": y, "alpha": DEFAULT_ALPHA.copy(), "lambda": np.asarray(lambdas),
                     "coefficients": coefficients, "coefficient_sha256": coefficient_sha,
                     "network_path": str(run / "03_NETWORK_CACHE/NETWORK.npz"),
                     "network_sha256": network_sha, "family": "binomial",
                     "permutation_times": 5000, "fdr_threshold": 0.05, "independent": True}
            with checkpoint.open("wb") as handle: pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pd.DataFrame({"gene": genes, "coefficient": coefficients,
                          "sign": np.sign(coefficients).astype(int)}).to_csv(
                              out / "MODEL_COEFFICIENTS.tsv.gz", sep="\t", index=False)
            metadata = {k: v for k, v in model.items() if k not in {"common_genes", "sample_ids", "phenotype", "coefficients", "lambda", "alpha"}}
            metadata.update({"genes": len(genes), "samples": len(samples), "class_0": int(np.sum(y == 0)),
                             "class_1": int(np.sum(y == 1)), "alpha": DEFAULT_ALPHA.tolist(),
                             "lambda": lambdas, "model_checkpoint_sha256": sha256(checkpoint),
                             "bulk_expression_path": str(bulk_path)})
            atomic_json(out / "MODEL_METADATA.json", metadata)
            status(out / "MODEL_CHECKPOINT_STATUS.txt", ["MODEL_CHECKPOINT=PASS"])
            del network, x
            gc.collect()
        else:
            with checkpoint.open("rb") as handle: model = pickle.load(handle)
            if sha256(run / "03_NETWORK_CACHE/NETWORK.npz") != model["network_sha256"]:
                raise RuntimeError("model checkpoint network checksum mismatch")
            if not coefficient_npy.exists(): np.save(coefficient_npy, model["coefficients"])
            progress(out, task, "RESUME_MODEL_CHECKPOINT", started, alpha_completed=13, alpha_total=13)

        cell_ids = (run / "02_FROZEN_INPUTS/SC_CACHE_CELLS.txt").read_text().splitlines()
        if len(cell_ids) != 356949 or len(set(cell_ids)) != len(cell_ids):
            raise RuntimeError("frozen cell ID validation failed")
        n_parts = int(os.environ.get("PERMUTATION_WORKERS", "4"))
        if not (out / "CELL_SCORING_STATUS.txt").exists():
            run_partitions(run, out, task, "score", len(cell_ids), n_parts, coefficient_npy, started)
            write_cell_rs(out, cell_ids, out / "partition_checkpoints", n_parts)
            status(out / "CELL_SCORING_STATUS.txt", ["CELL_SCORING=PASS", f"CELLS={len(cell_ids)}"])
        if not (out / "PERMUTATION_STATUS.txt").exists():
            run_partitions(run, out, task, "permutation", len(cell_ids), n_parts, coefficient_npy, started)
            write_cell_scores(out, cell_ids, out / "partition_checkpoints", n_parts,
                              label_column="scPAS" if task == "T02_G3_vs_MATCHED_CONTROL" else "scPAS_label")
            status(out / "PERMUTATION_STATUS.txt", ["PERMUTATION=PASS", "PERMUTATIONS=5000"])
        if task == "T02_G3_vs_MATCHED_CONTROL":
            scores = pd.read_csv(out / "CELL_SCORES.tsv.gz", sep="\t")
            mandatory = {
                "INPUT_MEMBERSHIP_PASS": len(manifest) == 74 and manifest.sample_id.is_unique,
                "CLASS_COUNTS_PASS": int(np.sum(y == 0)) == 39 and int(np.sum(y == 1)) == 35,
                "SINGLE_SOURCE_PASS": set(manifest.source_cohort) == {"GSE193777"},
                "COMMON_GENE_PASS": len(model["common_genes"]) == 16608,
                "NETWORK_SHA_PASS": sha256(run / "03_NETWORK_CACHE/NETWORK.npz") == model["network_sha256"],
                "ALPHA_13_13_PASS": len(model["alpha"]) == 13 and len(model["lambda"]) == 13,
                "MODEL_CHECKPOINT_PASS": checkpoint.exists() and checkpoint.stat().st_size > 0,
                "SCORING_356949_PASS": len(scores) == 356949 and scores.cell_id.is_unique,
                "PERMUTATION_5000_PASS": (out / "PERMUTATION_STATUS.txt").exists(),
                "FDR_PASS": np.isfinite(scores.scPAS_FDR).all() and scores.scPAS_FDR.between(0, 1).all(),
                "CELL_LABEL_PASS": set(scores.scPAS.unique()) <= {"scPAS+", "scPAS-", "0"},
            }
            pd.DataFrame({"qa_check": mandatory.keys(), "status": ["PASS" if value else "FAIL" for value in mandatory.values()]}).to_csv(
                out / "TASK_QA.tsv", sep="\t", index=False)
            if not all(mandatory.values()): raise RuntimeError("T02 mandatory final QA failed")
        status(status_file, ["STATUS=SUCCESS", "ERROR_STAGE=NONE", "ERROR=NONE"])
        progress(out, task, "TERMINAL", started, status="SUCCESS", cells=len(cell_ids), permutations_completed=5000)
        print(f"TASK={task}\nSTATUS=SUCCESS\nSTEP DONE", flush=True)
        return 0
    except Exception as exc:
        resumable = "YES" if (out / "MODEL_CHECKPOINT.pkl").exists() else "NO"
        current_stage = "UNKNOWN"
        try: current_stage = json.loads((out / "PROGRESS.json").read_text()).get("stage", "UNKNOWN")
        except Exception: pass
        status(status_file, ["STATUS=FAILED", f"ERROR_STAGE={current_stage}", f"ERROR={type(exc).__name__}: {exc}",
                             f"RESUMABLE_FROM_MODEL_CHECKPOINT={resumable}"])
        print(f"TASK={task}\nSTATUS=FAILED\nERROR={type(exc).__name__}: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
