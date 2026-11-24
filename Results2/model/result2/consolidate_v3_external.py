#!/usr/bin/env python3
"""Consolidate three frozen ncumap projections at participant level."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from consolidate_v3_risk import bootstrap_metric_ci, cliffs_delta, permutation_p, stable_seed


ROOT = Path(os.environ.get("TBSPIN_R2_RUN", "results2_run"))
SEEDS = (20260828, 20260829, 20260830)
TASK_STAGE = {"T1": ("HC", "TBINF"), "T2": ("TBINF", "ATB")}


def main() -> None:
    folder = ROOT / "18_EXTERNAL_VALIDATION"
    files = [folder / f"seed_{seed}/external_risk.tsv.gz" for seed in SEEDS]
    if not all(path.exists() for path in files):
        raise RuntimeError("not all three external risk projections exist")
    long = pd.concat([pd.read_csv(path, sep="\t") for path in files], ignore_index=True)
    keys = ["cell_id", "task"]
    meta = long.sort_values("seed").drop_duplicates(keys)[
        keys + ["sample_id_std", "analysis_stage", "celltype_raw", "projection_status"]
    ]
    values = long.pivot(index=keys, columns="seed", values="R").reset_index()
    values = values.rename(columns={seed: f"R_seed_{seed}" for seed in SEEDS})
    values.columns.name = None
    master = meta.merge(values, on=keys, validate="one_to_one")
    rcols = [f"R_seed_{seed}" for seed in SEEDS]
    master["R_ensemble"] = master[rcols].median(axis=1, skipna=True)
    master.loc[master[rcols].isna().all(axis=1), "R_ensemble"] = np.nan
    for label, cutoff in (("P90", 0.90), ("P95", 0.95), ("P975", 0.975)):
        master[f"risk_{label}"] = pd.array(master.R_ensemble >= cutoff, dtype="boolean")
        master.loc[master.R_ensemble.isna(), f"risk_{label}"] = pd.NA
    master.to_csv(folder / "RESULT2_EXTERNAL_RISK_CELL_MASTER.tsv.gz", sep="\t", index=False, compression="gzip")

    valid = master[master.R_ensemble.notna()].copy()
    burden_rows: list[dict[str, object]] = []
    for (task, participant, stage, celltype), group in valid.groupby(
        ["task", "sample_id_std", "analysis_stage", "celltype_raw"], observed=True
    ):
        burden_rows.append({
            "task": task, "participant": participant, "analysis_stage": stage,
            "celltype_raw": celltype, "eligible_n": len(group),
            "mean_R": float(group.R_ensemble.mean()), "median_R": float(group.R_ensemble.median()),
            "P90_fraction": float(group.risk_P90.mean()), "P95_fraction": float(group.risk_P95.mean()),
            "P975_fraction": float(group.risk_P975.mean()),
        })
    for (task, participant, stage), group in valid.groupby(
        ["task", "sample_id_std", "analysis_stage"], observed=True
    ):
        burden_rows.append({
            "task": task, "participant": participant, "analysis_stage": stage,
            "celltype_raw": "OVERALL", "eligible_n": len(group),
            "mean_R": float(group.R_ensemble.mean()), "median_R": float(group.R_ensemble.median()),
            "P90_fraction": float(group.risk_P90.mean()), "P95_fraction": float(group.risk_P95.mean()),
            "P975_fraction": float(group.risk_P975.mean()),
        })
    burden = pd.DataFrame(burden_rows)
    burden.to_csv(folder / "RESULT2_EXTERNAL_PARTICIPANT_BURDEN.tsv", sep="\t", index=False)

    comparison_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for task, (reference_stage, target_stage) in TASK_STAGE.items():
        task_burden = burden[burden.task == task]
        for celltype, group in task_burden.groupby("celltype_raw", observed=True):
            reference = group.loc[group.analysis_stage.eq(reference_stage), "P95_fraction"].dropna().to_numpy(float)
            target = group.loc[group.analysis_stage.eq(target_stage), "P95_fraction"].dropna().to_numpy(float)
            if len(reference) < 2 or len(target) < 2:
                continue
            comparison_rows.append({
                "task": task, "celltype_raw": celltype, "reference_stage": reference_stage,
                "target_stage": target_stage, "reference_participant_n": len(reference),
                "target_participant_n": len(target), "reference_median_P95_burden": float(np.median(reference)),
                "target_median_P95_burden": float(np.median(target)),
                "median_difference": float(np.median(target) - np.median(reference)),
                "cliffs_delta": cliffs_delta(reference, target),
                "permutation_p": permutation_p(reference, target, stable_seed("external", task, celltype)),
            })
        overall = task_burden[task_burden.celltype_raw.eq("OVERALL")]
        clinical = overall[overall.analysis_stage.isin([reference_stage, target_stage])].copy()
        y = clinical.analysis_stage.eq(target_stage).astype(int).to_numpy()
        for score_name in ("P95_fraction", "mean_R", "median_R"):
            score = clinical[score_name].to_numpy(float)
            if len(np.unique(y)) < 2:
                continue
            for metric_name, metric in (("AUROC", roc_auc_score), ("AUPRC", average_precision_score)):
                value = float(metric(y, score))
                low, high = bootstrap_metric_ci(y, score, metric, stable_seed("external-metric", task, score_name, metric_name))
                metric_rows.append({
                    "task": task, "score": score_name, "metric": metric_name, "value": value,
                    "CI_low": low, "CI_high": high, "participant_n": len(clinical),
                })
        ref_values = overall.loc[overall.analysis_stage.eq(reference_stage), "P95_fraction"].dropna().to_numpy(float)
        calibration_rows.append({
            "task": task, "reference_stage": reference_stage, "participant_n": len(ref_values),
            "median_P95_fraction": float(np.median(ref_values)) if len(ref_values) else np.nan,
            "q25": float(np.quantile(ref_values, 0.25)) if len(ref_values) else np.nan,
            "q75": float(np.quantile(ref_values, 0.75)) if len(ref_values) else np.nan,
            "reference_calibration_warning": bool(len(ref_values) and np.median(ref_values) > 0.15),
        })
    comparisons = pd.DataFrame(comparison_rows)
    metrics = pd.DataFrame(metric_rows)
    calibration = pd.DataFrame(calibration_rows)
    comparisons.to_csv(folder / "RESULT2_EXTERNAL_LINEAGE_COMPARISON.tsv", sep="\t", index=False)
    metrics.to_csv(folder / "RESULT2_EXTERNAL_PARTICIPANT_METRICS.tsv", sep="\t", index=False)
    calibration.to_csv(folder / "RESULT2_EXTERNAL_REFERENCE_CALIBRATION.tsv", sep="\t", index=False)

    internal = pd.read_csv(ROOT / "14_PARTICIPANT_BURDEN/RESULT2_PARTICIPANT_BURDEN_COMPARISON.tsv", sep="\t")
    internal = internal[(internal.pcut == "P95") & (internal.celltype_raw != "OVERALL")]
    external = comparisons[comparisons.celltype_raw != "OVERALL"]
    concordance = internal[["task", "celltype_raw", "median_difference", "cliffs_delta"]].merge(
        external[["task", "celltype_raw", "median_difference", "cliffs_delta"]],
        on=["task", "celltype_raw"], suffixes=("_internal", "_external"), how="outer",
    )
    concordance["direction_concordance"] = np.where(
        concordance.median_difference_internal.notna() & concordance.median_difference_external.notna(),
        np.sign(concordance.median_difference_internal) == np.sign(concordance.median_difference_external), pd.NA,
    )
    concordance["status"] = np.where(
        concordance.direction_concordance.eq(True), "CONCORDANT",
        np.where(concordance.direction_concordance.eq(False), "NOT_CONCORDANT", "NOT_ESTIMABLE"),
    )
    concordance.to_csv(folder / "RESULT2_EXTERNAL_CELLTYPE_CONCORDANCE.tsv", sep="\t", index=False)

    train_participants = set(pd.read_csv(ROOT / "02_INPUT_AUDIT/MODEL_READY_METADATA.tsv.gz", sep="\t", usecols=["sample_id_std"]).sample_id_std.astype(str))
    external_participants = set(master.sample_id_std.astype(str))
    overlap = sorted(train_participants & external_participants)
    audits = [json.loads((folder / f"seed_{seed}/AUDIT.json").read_text(encoding="utf-8")) for seed in SEEDS]
    match_fraction = min(float(item["match_fraction"]) for item in audits)
    status = {
        "status": "PASS", "seed_n": 3, "external_cell_n": int(master.cell_id.nunique()),
        "external_participant_n": len(external_participants), "training_participant_overlap_n": len(overlap),
        "training_participant_overlap": overlap, "external_independence_status": "PASS" if not overlap else "FAIL",
        "external_gene_match_fraction": match_fraction,
        "external_gene_alignment_status": "PASS" if match_fraction >= 0.80 else "WARNING",
        "reference_calibration_status": "PASS" if not calibration.reference_calibration_warning.any() else "WARNING",
        "fine_tuning": False,
    }
    (folder / "DONE.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    pd.concat([metrics.assign(table="METRIC"), comparisons.assign(table="COMPARISON")], ignore_index=True).to_csv(
        ROOT / "24_TABLES/RESULT2_EXTERNAL_VALIDATION.tsv", sep="\t", index=False
    )
    print("V3_EXTERNAL_CONSOLIDATION=PASS " + json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
