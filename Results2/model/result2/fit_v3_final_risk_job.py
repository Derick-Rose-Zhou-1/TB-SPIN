#!/usr/bin/env python3
"""Fit frozen participant-balanced reference distributions for final full models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from fit_v3_reference_job import DISTROS, balanced_sample, bootstrap_ks, fit_dist, stable_seed


ROOT = Path(os.environ.get("TBSPIN_R2_RUN", "results2_run"))
SEEDS = (20260828, 20260829, 20260830)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True, choices=range(3))
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    seed = SEEDS[args.index]
    output = ROOT / f"17_FINAL_FULL_MODEL/seed_{seed}"
    reference = pd.read_csv(output / "training_reference.tsv.gz", sep="\t")
    directions = pd.read_csv(output / "fisher_directions.tsv", sep="\t")
    details: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    fitted_models: dict[str, dict[str, object]] = {}
    for _, direction in directions.iterrows():
        task = str(direction.task)
        celltype = str(direction.celltype_raw)
        key = f"{task}|{celltype}"
        if direction.status != "PASS":
            selections.append({
                "seed": seed, "task": task, "celltype_raw": celltype, "status": "LOW_SUPPORT",
                "winner": "", "parametric_fit_warning": True,
            })
            continue
        group = reference[(reference.task == task) & (reference.celltype_raw == celltype)].copy()
        counts = group.groupby("sample_id_std").size()
        n_balance = int(min(2000, counts.min()))
        fits = {name: [] for name in DISTROS}
        primary_values = None
        for resample in range(20):
            values = balanced_sample(group, n_balance, stable_seed("final-balance", seed, task, celltype, resample))
            if primary_values is None:
                primary_values = values
            for name in DISTROS:
                params, diagnostics = fit_dist(name, values)
                fits[name].append((params, diagnostics))
                details.append({
                    "seed": seed, "task": task, "celltype_raw": celltype, "resample": resample,
                    "distribution": name, "n_balance": n_balance,
                    "reference_participant_n": len(counts), "sample_n": len(values),
                    "parameters_json": json.dumps(params), **diagnostics,
                })
        summaries: list[dict[str, object]] = []
        assert primary_values is not None
        for name, items in fits.items():
            primary_params, primary_diag = items[0]
            ks_p = bootstrap_ks(
                name, primary_values, primary_diag["KS"],
                stable_seed("final-bootstrap", seed, task, celltype, name), args.bootstrap,
            )
            summary: dict[str, object] = {
                "candidate": name, "bootstrap_KS_P": ks_p,
                "primary_parameters_json": json.dumps(primary_params),
            }
            for metric in ("AIC", "BIC", "KS", "QQ_R2", "FITTED_Q90", "FITTED_Q95", "FITTED_Q975",
                           "RELATIVE_TAIL_ERROR_Q90", "RELATIVE_TAIL_ERROR_Q95", "RELATIVE_TAIL_ERROR_Q975"):
                values = np.asarray([item[1][metric] for item in items], dtype=float)
                summary[f"median_{metric}"] = float(np.median(values))
                summary[f"iqr_{metric}"] = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
            summaries.append(summary)
        passing = [item for item in summaries if float(item["bootstrap_KS_P"]) >= 0.05]
        candidates = passing if passing else summaries
        winner = min(candidates, key=lambda item: float(item["median_BIC"]))
        final_name = str(winner["candidate"])
        final_params = json.loads(str(winner["primary_parameters_json"]))
        fitted_models[key] = {
            "distribution": final_name, "parameters": final_params,
            "n_balance": n_balance, "reference_participant_n": len(counts),
            "parametric_fit_warning": not bool(passing),
        }
        for summary in summaries:
            selections.append({
                "seed": seed, "task": task, "celltype_raw": celltype, "status": "PASS",
                "n_balance": n_balance, "reference_participant_n": len(counts),
                "low_reference_support": n_balance < 100, "winner": final_name,
                "is_winner": summary["candidate"] == final_name,
                "parametric_fit_warning": not bool(passing), **summary,
            })
    pd.DataFrame(details).to_csv(output / "final_distribution_resamples.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(selections).to_csv(output / "final_distribution_selection.tsv", sep="\t", index=False)
    (output / "fitted_risk_models.json").write_text(json.dumps(fitted_models, indent=2, sort_keys=True), encoding="utf-8")
    (output / "RISK_DONE.json").write_text(json.dumps({"seed": seed, "status": "PASS", "bootstrap_B": args.bootstrap}, indent=2), encoding="utf-8")
    print(f"V3_FINAL_RISK=PASS seed={seed} layer_n={len(fitted_models)}", flush=True)


if __name__ == "__main__":
    main()
