#!/usr/bin/env python3
"""Fit participant-balanced positive reference distributions per OOF fold/seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SEEDS = (20260828, 20260829, 20260830)
TASKS = ("T1", "T2")
CELLTYPES = ("B cell", "Monocyte", "NK cell", "Neutrophil", "T cell", "pDC")
DISTROS = {
    "GAMMA": stats.gamma,
    "LOGNORMAL": stats.lognorm,
    "WEIBULL": stats.weibull_min,
    "EXPONENTIAL": stats.expon,
}
QUANTILES = (0.90, 0.95, 0.975)


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def fit_dist(name: str, values: np.ndarray) -> tuple[tuple[float, ...], dict[str, float]]:
    dist = DISTROS[name]
    values = np.maximum(np.asarray(values, dtype=np.float64), np.finfo(np.float64).tiny)
    params = tuple(float(x) for x in dist.fit(values, floc=0.0))
    logpdf = dist.logpdf(values, *params)
    if not np.isfinite(logpdf).all():
        raise FloatingPointError(f"non-finite log likelihood for {name}")
    ll = float(logpdf.sum())
    k = len(params) - 1  # loc is fixed, not estimated
    n = len(values)
    ks = float(stats.kstest(values, dist.cdf, args=params).statistic)
    probs = (np.arange(n, dtype=np.float64) + 0.5) / n
    empirical = np.sort(values)
    fitted = dist.ppf(probs, *params)
    finite = np.isfinite(fitted)
    qq_r2 = float(np.corrcoef(empirical[finite], fitted[finite])[0, 1] ** 2)
    out = {
        "log_likelihood": ll,
        "parameter_n": float(k),
        "AIC": float(2 * k - 2 * ll),
        "BIC": float(k * math.log(n) - 2 * ll),
        "KS": ks,
        "QQ_R2": qq_r2,
    }
    for q in QUANTILES:
        label = {0.90: "Q90", 0.95: "Q95", 0.975: "Q975"}[q]
        observed = float(np.quantile(values, q))
        fitted_q = float(dist.ppf(q, *params))
        out[f"EMPIRICAL_{label}"] = observed
        out[f"FITTED_{label}"] = fitted_q
        out[f"RELATIVE_TAIL_ERROR_{label}"] = float((fitted_q - observed) / max(observed, 1e-12))
    return params, out


def balanced_sample(frame: pd.DataFrame, n_balance: int, seed: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for participant, group in frame.groupby("sample_id_std", sort=True):
        values = group["Y"].to_numpy(dtype=np.float64)
        rng = np.random.default_rng(stable_seed(seed, participant))
        take = rng.choice(len(values), size=n_balance, replace=False)
        chunks.append(values[take])
    return np.concatenate(chunks)


def bootstrap_ks(name: str, values: np.ndarray, observed_ks: float, seed: int, b: int) -> float:
    dist = DISTROS[name]
    params = tuple(float(x) for x in dist.fit(values, floc=0.0))
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(b):
        sample = np.asarray(dist.rvs(*params, size=len(values), random_state=rng), dtype=np.float64)
        sample = np.maximum(sample, np.finfo(np.float64).tiny)
        boot_params = tuple(float(x) for x in dist.fit(sample, floc=0.0))
        boot_ks = float(stats.kstest(sample, dist.cdf, args=boot_params).statistic)
        ge += int(boot_ks >= observed_ks)
    return float((1 + ge) / (b + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()

    root = Path(args.root)
    fold, seed = divmod(args.index, len(SEEDS))[0], SEEDS[args.index % len(SEEDS)]
    ref_path = root / "11_FISHER_COORDINATE" / f"fold_{fold}_seed_{seed}_training_reference.tsv.gz"
    held_path = root / "11_FISHER_COORDINATE" / f"fold_{fold}_seed_{seed}_heldout.tsv.gz"
    directions_path = root / "11_FISHER_COORDINATE" / f"fold_{fold}_seed_{seed}_directions.tsv"
    out_dir = root / "12_REFERENCE_DISTRIBUTION" / f"fold_{fold}_seed_{seed}"
    risk_dir = root / "13_RISK_CELL_CALL" / "per_fold_seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_dir.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(ref_path, sep="\t")
    heldout = pd.read_csv(held_path, sep="\t")
    directions = pd.read_csv(directions_path, sep="\t")
    detail_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    fitted_models: dict[str, dict[str, object]] = {}

    for task in TASKS:
        for celltype in CELLTYPES:
            key = f"{task}|{celltype}"
            direction = directions[(directions.task == task) & (directions.celltype_raw == celltype)]
            if direction.empty or direction.iloc[0]["status"] != "PASS":
                selection_rows.append({
                    "fold": fold, "seed": seed, "task": task, "celltype_raw": celltype,
                    "status": "LOW_SUPPORT", "winner": "", "parametric_fit_warning": True,
                })
                continue
            group = reference[(reference.task == task) & (reference.celltype_raw == celltype)].copy()
            counts = group.groupby("sample_id_std").size()
            n_balance = int(min(2000, counts.min()))
            low_reference_support = n_balance < 100
            fits: dict[str, list[tuple[tuple[float, ...], dict[str, float]]]] = {name: [] for name in DISTROS}
            primary_values: np.ndarray | None = None
            for resample in range(20):
                rs = stable_seed("balance", fold, seed, task, celltype, resample)
                values = balanced_sample(group, n_balance, rs)
                if primary_values is None:
                    primary_values = values
                for name in DISTROS:
                    params, diagnostics = fit_dist(name, values)
                    fits[name].append((params, diagnostics))
                    detail_rows.append({
                        "fold": fold, "seed": seed, "task": task, "celltype_raw": celltype,
                        "resample": resample, "distribution": name, "n_balance": n_balance,
                        "reference_participant_n": int(len(counts)), "sample_n": int(len(values)),
                        "parameters_json": json.dumps(params), **diagnostics,
                    })
            assert primary_values is not None
            summaries: list[dict[str, object]] = []
            for name, items in fits.items():
                primary_params, primary_diag = items[0]
                ks_p = bootstrap_ks(
                    name, primary_values, primary_diag["KS"],
                    stable_seed("bootstrap", fold, seed, task, celltype, name), args.bootstrap,
                )
                summary: dict[str, object] = {
                    "distribution": name,
                    "bootstrap_KS_P": ks_p,
                    "primary_parameters_json": json.dumps(primary_params),
                }
                for metric in ("log_likelihood", "AIC", "BIC", "KS", "QQ_R2",
                               "FITTED_Q90", "FITTED_Q95", "FITTED_Q975",
                               "RELATIVE_TAIL_ERROR_Q90", "RELATIVE_TAIL_ERROR_Q95",
                               "RELATIVE_TAIL_ERROR_Q975"):
                    vals = np.array([item[1][metric] for item in items], dtype=np.float64)
                    summary[f"median_{metric}"] = float(np.median(vals))
                    summary[f"iqr_{metric}"] = float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25))
                summaries.append(summary)
            passing = [x for x in summaries if float(x["bootstrap_KS_P"]) >= 0.05]
            candidates = passing if passing else summaries
            winner = min(candidates, key=lambda x: float(x["median_BIC"]))
            warning = not bool(passing)
            final_name = str(winner["distribution"])
            final_params = json.loads(str(winner["primary_parameters_json"]))
            fitted_models[key] = {"distribution": final_name, "parameters": final_params}
            for summary in summaries:
                selection_rows.append({
                    "fold": fold, "seed": seed, "task": task, "celltype_raw": celltype,
                    "status": "PASS", "n_balance": n_balance,
                    "reference_participant_n": int(len(counts)),
                    "low_reference_support": low_reference_support,
                    "candidate": summary["distribution"], "winner": final_name,
                    "is_winner": summary["distribution"] == final_name,
                    "parametric_fit_warning": warning, **summary,
                })

    risk_parts: list[pd.DataFrame] = []
    for (task, celltype), group in heldout.groupby(["task", "celltype_raw"], sort=False):
        key = f"{task}|{celltype}"
        piece = group.copy()
        if key not in fitted_models:
            piece["R"] = np.nan
            piece["distribution"] = ""
            piece["risk_layer_status"] = "LOW_SUPPORT"
        else:
            model = fitted_models[key]
            piece["R"] = DISTROS[str(model["distribution"])].cdf(
                piece["Y"].to_numpy(dtype=np.float64), *model["parameters"]
            )
            piece["distribution"] = str(model["distribution"])
            piece["risk_layer_status"] = "PASS"
        for label, cutoff in (("P90", 0.90), ("P95", 0.95), ("P975", 0.975)):
            piece[f"risk_{label}_seed"] = np.where(piece["R"].notna(), piece["R"] >= cutoff, False)
        risk_parts.append(piece)
    risk = pd.concat(risk_parts, ignore_index=True)
    risk.to_csv(risk_dir / f"fold_{fold}_seed_{seed}.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(detail_rows).to_csv(out_dir / "resample_fits.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(selection_rows).to_csv(out_dir / "distribution_selection.tsv", sep="\t", index=False)
    with open(out_dir / "fitted_models.json", "w", encoding="utf-8") as handle:
        json.dump(fitted_models, handle, indent=2, sort_keys=True)
    with open(out_dir / "DONE.json", "w", encoding="utf-8") as handle:
        json.dump({"fold": fold, "seed": seed, "bootstrap_B": args.bootstrap, "status": "PASS"}, handle, indent=2)
    print(f"V3_REFERENCE_FIT=PASS fold={fold} seed={seed} risk_rows={len(risk)}", flush=True)


if __name__ == "__main__":
    main()
