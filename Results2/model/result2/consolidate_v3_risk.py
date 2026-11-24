#!/usr/bin/env python3
"""Consolidate OOF risk percentiles and run participant-level burden analyses."""

from __future__ import annotations

import argparse
from burden_permutation import stable_seed, permutation_p
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score


SEEDS = (20260828, 20260829, 20260830)
TASK_STAGE = {"T1": ("HC", "TBINF"), "T2": ("TBINF", "ATB")}
PCUTS = {"P90": 0.90, "P95": 0.95, "P975": 0.975}


def bh(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    n = len(valid)
    adjusted = (valid.to_numpy() * n / np.arange(1, n + 1))[::-1]
    adjusted = np.minimum.accumulate(adjusted)[::-1]
    result.loc[valid.index] = np.minimum(adjusted, 1.0)
    return result


def cliffs_delta(reference: np.ndarray, target: np.ndarray) -> float:
    if not len(reference) or not len(target):
        return float("nan")
    ranks = stats.rankdata(np.concatenate([target, reference]))
    rank_target = ranks[: len(target)].sum()
    u = rank_target - len(target) * (len(target) + 1) / 2
    return float(2 * u / (len(target) * len(reference)) - 1)


def bootstrap_ci(reference: np.ndarray, target: np.ndarray, seed: int, b: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    effects = np.empty(b, dtype=np.float64)
    for i in range(b):
        r = rng.choice(reference, size=len(reference), replace=True)
        t = rng.choice(target, size=len(target), replace=True)
        effects[i] = np.median(t) - np.median(r)
    return tuple(float(x) for x in np.quantile(effects, [0.025, 0.975]))


def bootstrap_metric_ci(y: np.ndarray, score: np.ndarray, metric, seed: int, b: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(b):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(metric(y[idx], score[idx])))
    return tuple(float(x) for x in np.quantile(vals, [0.025, 0.975]))


def load_risk(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    risk_files = sorted((root / "13_RISK_CELL_CALL" / "per_fold_seed").glob("fold_*_seed_*.tsv.gz"))
    if len(risk_files) != 15:
        raise RuntimeError(f"expected 15 risk files, found {len(risk_files)}")
    long = pd.concat([pd.read_csv(path, sep="\t") for path in risk_files], ignore_index=True)
    keys = ["cell_id", "task"]
    meta_cols = keys + ["fold", "sample_id_std", "analysis_stage", "celltype_raw", "coordinate_status"]
    meta = long.sort_values("seed").drop_duplicates(keys)[meta_cols]
    values = long.pivot(index=keys, columns="seed", values=["D", "Y", "R"]).reset_index()
    values.columns = [f"{a}_seed_{b}" if b else str(a) for a, b in values.columns]
    master = meta.merge(values, on=keys, how="left", validate="one_to_one")
    rcols = [f"R_seed_{seed}" for seed in SEEDS]
    master["R_ensemble"] = master[rcols].median(axis=1, skipna=True)
    master.loc[master[rcols].isna().all(axis=1), "R_ensemble"] = np.nan
    master["risk_layer_status"] = np.where(master.R_ensemble.notna(), "PASS", "LOW_SUPPORT")
    for label, cutoff in PCUTS.items():
        master[f"risk_{label}"] = pd.array(master.R_ensemble >= cutoff, dtype="boolean")
        master.loc[master.R_ensemble.isna(), f"risk_{label}"] = pd.NA

    selections = pd.concat(
        [pd.read_csv(path, sep="\t") for path in sorted((root / "12_REFERENCE_DISTRIBUTION").glob("fold_*_seed_*/distribution_selection.tsv"))],
        ignore_index=True,
    )
    details = pd.concat(
        [pd.read_csv(path, sep="\t") for path in sorted((root / "12_REFERENCE_DISTRIBUTION").glob("fold_*_seed_*/resample_fits.tsv.gz"))],
        ignore_index=True,
    )
    return master, selections, details


def participant_burden(master: pd.DataFrame) -> pd.DataFrame:
    eligible = master[master.R_ensemble.notna()].copy()
    rows: list[dict[str, object]] = []
    for (participant, task, stage, celltype), group in eligible.groupby(
        ["sample_id_std", "task", "analysis_stage", "celltype_raw"], observed=True
    ):
        row: dict[str, object] = {
            "participant": participant, "task": task, "analysis_stage": stage,
            "celltype_raw": celltype, "level": "CELLTYPE_RAW", "eligible_n": len(group),
        }
        for label in PCUTS:
            row[f"{label}_risk_n"] = int(group[f"risk_{label}"].sum())
            row[f"{label}_risk_fraction"] = float(group[f"risk_{label}"].mean())
        rows.append(row)
    for (participant, task, stage), group in eligible.groupby(
        ["sample_id_std", "task", "analysis_stage"], observed=True
    ):
        row = {
            "participant": participant, "task": task, "analysis_stage": stage,
            "celltype_raw": "OVERALL", "level": "OVERALL", "eligible_n": len(group),
        }
        for label in PCUTS:
            row[f"{label}_risk_n"] = int(group[f"risk_{label}"].sum())
            row[f"{label}_risk_fraction"] = float(group[f"risk_{label}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def burden_comparisons(burden: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, (reference_stage, target_stage) in TASK_STAGE.items():
        task_data = burden[burden.task == task]
        for celltype in sorted(task_data.celltype_raw.unique()):
            subset = task_data[task_data.celltype_raw == celltype]
            for label in PCUTS:
                column = f"{label}_risk_fraction"
                reference = subset.loc[subset.analysis_stage == reference_stage, column].dropna().to_numpy(float)
                target = subset.loc[subset.analysis_stage == target_stage, column].dropna().to_numpy(float)
                if len(reference) < 2 or len(target) < 2:
                    continue
                seed = stable_seed(task, celltype, label)
                ci_low, ci_high = bootstrap_ci(reference, target, seed)
                rows.append({
                    "task": task, "reference_stage": reference_stage, "target_stage": target_stage,
                    "celltype_raw": celltype, "pcut": label,
                    "reference_participant_n": len(reference), "target_participant_n": len(target),
                    "reference_median_burden": float(np.median(reference)),
                    "target_median_burden": float(np.median(target)),
                    "median_difference": float(np.median(target) - np.median(reference)),
                    "median_difference_CI_low": ci_low, "median_difference_CI_high": ci_high,
                    "cliffs_delta": cliffs_delta(reference, target),
                    "permutation_p": permutation_p(reference, target, seed),
                    "mannwhitney_p": float(stats.mannwhitneyu(target, reference, alternative="two-sided").pvalue),
                })
    result = pd.DataFrame(rows)
    result["BH_FDR_within_task_pcut"] = np.nan
    for (_, _), idx in result[result.celltype_raw != "OVERALL"].groupby(["task", "pcut"]).groups.items():
        result.loc[idx, "BH_FDR_within_task_pcut"] = bh(result.loc[idx, "permutation_p"])
    return result


def calibration(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, (reference_stage, _) in TASK_STAGE.items():
        subset = master[(master.task == task) & (master.analysis_stage == reference_stage) & master.R_ensemble.notna()]
        participant = subset.groupby("sample_id_std")[[f"risk_{x}" for x in PCUTS]].mean().reset_index()
        for label in PCUTS:
            values = participant[f"risk_{label}"].to_numpy(float)
            rows.append({
                "task": task, "reference_stage": reference_stage, "pcut": label,
                "participant_n": len(values), "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)), "q75": float(np.quantile(values, 0.75)),
                "minimum": float(values.min()), "maximum": float(values.max()),
                "nominal_tail": 1 - PCUTS[label],
                "reference_calibration_warning": bool(label == "P95" and np.median(values) > 0.15),
            })
    return pd.DataFrame(rows)


def composition(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comp_rows: list[dict[str, object]] = []
    within_rows: list[dict[str, object]] = []
    for task, (_, target_stage) in TASK_STAGE.items():
        subset = master[(master.task == task) & (master.analysis_stage == target_stage) & master.R_ensemble.notna()]
        for label in PCUTS:
            risk = subset[subset[f"risk_{label}"] == True]
            total_risk = len(risk)
            for celltype, group in subset.groupby("celltype_raw", observed=True):
                risk_n = int(group[f"risk_{label}"].sum())
                comp_rows.append({
                    "task": task, "target_stage": target_stage, "pcut": label,
                    "celltype_raw": celltype, "risk_n": risk_n, "all_target_risk_n": total_risk,
                    "risk_cell_composition": risk_n / total_risk if total_risk else np.nan,
                })
                within_rows.append({
                    "task": task, "target_stage": target_stage, "pcut": label,
                    "celltype_raw": celltype, "risk_n": risk_n, "eligible_n": len(group),
                    "within_celltype_risk_fraction": risk_n / len(group),
                })
    return pd.DataFrame(comp_rows), pd.DataFrame(within_rows)


def pcut_sensitivity(master: pd.DataFrame, comparisons: pd.DataFrame, composition_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task in TASK_STAGE:
        subset = master[(master.task == task) & master.R_ensemble.notna()]
        sets = {label: set(subset.loc[subset[f"risk_{label}"] == True, "cell_id"]) for label in PCUTS}
        for label, cutoff in PCUTS.items():
            current = sets[label]
            p95 = sets["P95"]
            union = len(current | p95)
            overlap = len(current & p95)
            effect = comparisons[(comparisons.task == task) & (comparisons.celltype_raw == "OVERALL") & (comparisons.pcut == label)]
            ranking = composition_df[(composition_df.task == task) & (composition_df.pcut == label)].sort_values(
                "risk_cell_composition", ascending=False
            ).celltype_raw.tolist()
            rows.append({
                "task": task, "pcut": label, "cutoff": cutoff, "risk_cell_n": len(current),
                "risk_fraction": len(current) / len(subset),
                "jaccard_vs_P95": overlap / union if union else np.nan,
                "overlap_coefficient_vs_P95": overlap / min(len(current), len(p95)) if current and p95 else np.nan,
                "P975_subset_P95": sets["P975"].issubset(sets["P95"]),
                "P95_subset_P90": sets["P95"].issubset(sets["P90"]),
                "overall_median_difference": float(effect.iloc[0].median_difference) if len(effect) else np.nan,
                "composition_ranking": " > ".join(ranking),
            })
    return pd.DataFrame(rows)


def seed_stability(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task in TASK_STAGE:
        task_data = master[master.task == task]
        for celltype in ["OVERALL"] + sorted(task_data.celltype_raw.unique()):
            subset = task_data if celltype == "OVERALL" else task_data[task_data.celltype_raw == celltype]
            for seed_a, seed_b in itertools.combinations(SEEDS, 2):
                a = subset[f"R_seed_{seed_a}"]
                b = subset[f"R_seed_{seed_b}"]
                valid = a.notna() & b.notna()
                if not valid.any():
                    continue
                row: dict[str, object] = {
                    "task": task, "celltype_raw": celltype, "seed_a": seed_a, "seed_b": seed_b,
                    "cell_n": int(valid.sum()), "risk_percentile_spearman": float(stats.spearmanr(a[valid], b[valid]).statistic),
                }
                for label, cutoff in PCUTS.items():
                    set_a = set(subset.loc[valid & (a >= cutoff), "cell_id"])
                    set_b = set(subset.loc[valid & (b >= cutoff), "cell_id"])
                    row[f"{label}_label_jaccard"] = len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else np.nan
                pa = subset.loc[valid].assign(_r=a[valid]).groupby("sample_id_std")["_r"].mean()
                pb = subset.loc[valid].assign(_r=b[valid]).groupby("sample_id_std")["_r"].mean()
                common = pa.index.intersection(pb.index)
                row["participant_burden_spearman"] = float(stats.spearmanr(pa.loc[common], pb.loc[common]).statistic)
                rows.append(row)
    return pd.DataFrame(rows)


def oof_performance(root: Path) -> pd.DataFrame:
    files = sorted((root / "10_OOF_PARTICIPANT").glob("fold_*_seed_*.tsv"))
    files = [x for x in files if not x.name.endswith("_metrics.tsv")]
    data = pd.concat([pd.read_csv(path, sep="\t") for path in files], ignore_index=True)
    participant = data.groupby(["task", "participant", "analysis_stage", "y_true"], as_index=False).probability.median()
    rows: list[dict[str, object]] = []
    for task, group in participant.groupby("task"):
        y = group.y_true.to_numpy(int)
        score = group.probability.to_numpy(float)
        for name, metric in (("AUROC", roc_auc_score), ("AUPRC", average_precision_score)):
            value = float(metric(y, score))
            low, high = bootstrap_metric_ci(y, score, metric, stable_seed(task, name))
            rows.append({"task": task, "metric": name, "value": value, "CI_low": low, "CI_high": high, "participant_n": len(group)})
        rows.append({"task": task, "metric": "Brier", "value": float(brier_score_loss(y, score)), "CI_low": np.nan, "CI_high": np.nan, "participant_n": len(group)})
        rows.append({"task": task, "metric": "balanced_accuracy", "value": float(balanced_accuracy_score(y, score >= 0.5)), "CI_low": np.nan, "CI_high": np.nan, "participant_n": len(group)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    for name in ("13_RISK_CELL_CALL", "14_PARTICIPANT_BURDEN", "15_CELLTYPE_RAW_ANALYSIS", "22_ROBUSTNESS", "24_TABLES"):
        (root / name).mkdir(parents=True, exist_ok=True)

    master, selections, details = load_risk(root)
    master.to_csv(root / "13_RISK_CELL_CALL" / "RESULT2_FINAL_RISK_CELL_MASTER.tsv.gz", sep="\t", index=False, compression="gzip")
    selections.to_csv(root / "12_REFERENCE_DISTRIBUTION" / "RESULT2_DISTRIBUTION_SELECTION.tsv", sep="\t", index=False)
    details.to_csv(root / "12_REFERENCE_DISTRIBUTION" / "RESULT2_DISTRIBUTION_RESAMPLE_STABILITY.tsv.gz", sep="\t", index=False, compression="gzip")

    burden = participant_burden(master)
    comparisons = burden_comparisons(burden)
    calibration_df = calibration(master)
    comp, within = composition(master)
    pcut = pcut_sensitivity(master, comparisons, comp)
    stability = seed_stability(master)
    perf = oof_performance(root)

    burden.to_csv(root / "14_PARTICIPANT_BURDEN" / "RESULT2_PARTICIPANT_RISK_BURDEN.tsv", sep="\t", index=False)
    comparisons.to_csv(root / "14_PARTICIPANT_BURDEN" / "RESULT2_PARTICIPANT_BURDEN_COMPARISON.tsv", sep="\t", index=False)
    calibration_df.to_csv(root / "14_PARTICIPANT_BURDEN" / "RESULT2_REFERENCE_CALIBRATION.tsv", sep="\t", index=False)
    comp.to_csv(root / "15_CELLTYPE_RAW_ANALYSIS" / "RESULT2_RISK_CELL_COMPOSITION.tsv", sep="\t", index=False)
    within.to_csv(root / "15_CELLTYPE_RAW_ANALYSIS" / "RESULT2_WITHIN_CELLTYPE_RISK_FRACTION.tsv", sep="\t", index=False)
    pcut.to_csv(root / "22_ROBUSTNESS" / "RESULT2_PCUT_SENSITIVITY.tsv", sep="\t", index=False)
    stability.to_csv(root / "22_ROBUSTNESS" / "RESULT2_SEED_STABILITY.tsv", sep="\t", index=False)
    perf.to_csv(root / "10_OOF_PARTICIPANT" / "RESULT2_OOF_PERFORMANCE.tsv", sep="\t", index=False)

    table_map = {
        "RESULT2_OOF_PERFORMANCE.tsv": perf,
        "RESULT2_DISTRIBUTION_SELECTION.tsv": selections,
        "RESULT2_REFERENCE_CALIBRATION.tsv": calibration_df,
        "RESULT2_P95_PARTICIPANT_BURDEN.tsv": burden,
        "RESULT2_P95_CELLTYPE_RAW_COMPOSITION.tsv": comp[comp.pcut == "P95"],
        "RESULT2_P95_WITHIN_CELLTYPE_FRACTION.tsv": within[within.pcut == "P95"],
        "RESULT2_PCUT_SENSITIVITY.tsv": pcut,
        "RESULT2_SEED_STABILITY.tsv": stability,
    }
    risk_counts = master.groupby(["task", "analysis_stage"], as_index=False).agg(
        eligible_n=("R_ensemble", "count"), P95_risk_n=("risk_P95", "sum")
    )
    risk_counts["P95_risk_fraction"] = risk_counts.P95_risk_n / risk_counts.eligible_n
    table_map["RESULT2_P95_RISK_CELL_COUNTS.tsv"] = risk_counts
    for filename, frame in table_map.items():
        frame.to_csv(root / "24_TABLES" / filename, sep="\t", index=False)

    status = {
        "risk_master_rows": len(master), "participant_burden_rows": len(burden),
        "comparison_rows": len(comparisons), "reference_calibration_warning": bool(calibration_df.reference_calibration_warning.any()),
        "P975_subset_P95": bool(pcut.P975_subset_P95.all()), "P95_subset_P90": bool(pcut.P95_subset_P90.all()),
        "status": "PASS",
    }
    with open(root / "14_PARTICIPANT_BURDEN" / "DONE.json", "w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)
    print("V3_RISK_CONSOLIDATION=PASS " + json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
