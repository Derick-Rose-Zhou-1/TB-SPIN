#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import asdict
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf

from tb_state_sc_v3_training import (
    TrainConfig,
    encode_cell_states,
    evaluate_model,
    fit_model_fixed_epochs,
    load_result2_fold,
    train_model,
)


ROOT = Path(os.environ.get("TBSPIN_R2_RUN", "results2_run"))
SEEDS = [20260828, 20260829, 20260830]
TASKS = {
    "T1": {"reference": "HC", "target": "TBINF"},
    "T2": {"reference": "TBINF", "target": "ATB"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def best_epochs(history: pd.DataFrame) -> tuple[int, int]:
    pre = history.loc[history["phase"].eq("pretrain")]
    pre_eval = pre.loc[pre["validation_reconstruction"].notna()]
    pre_epoch = (
        int(pre_eval.loc[pre_eval["validation_reconstruction"].idxmin(), "epoch"])
        if len(pre_eval) else int(pre["epoch"].max())
    )
    fine = history.loc[history["phase"].eq("finetune")]
    fine_eval = fine.loc[fine["validation_T2_AUPRC"].notna()]
    fine_epoch = (
        int(fine_eval.loc[fine_eval["validation_T2_AUPRC"].idxmax(), "epoch"])
        if len(fine_eval) else int(fine["epoch"].max())
    )
    return max(1, pre_epoch), max(1, fine_epoch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True, choices=range(15))
    args = parser.parse_args()
    fold = args.index // 3
    seed = SEEDS[args.index % 3]
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    output = ROOT / f"08_OOF_MODELS/fold_{fold}/seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    latent_path = ROOT / f"09_OOF_LATENT/fold_{fold}_seed_{seed}.tsv.gz"
    participant_path = ROOT / f"10_OOF_PARTICIPANT/fold_{fold}_seed_{seed}.tsv"
    metric_path = ROOT / f"10_OOF_PARTICIPANT/fold_{fold}_seed_{seed}_metrics.tsv"

    config = TrainConfig(
        state_dim=8,
        tau=0.5,
        lambda_cross=0.01,
        lambda_id=0.20,
        beta_kl=0.05,
        lr_finetune_encoder=1e-4,
        likelihood="zinb",
        identity_dim=16,
        hidden_dim=256,
        lambda_ordinal=0.15,
        lr_pretrain=5e-4,
        lr_heads=5e-4,
        weight_decay=1e-4,
        max_pretrain_epochs=100,
        max_finetune_epochs=150,
        participants_per_batch=8,
        cells_per_participant=512,
        gradient_clip=5.0,
        eval_every=5,
        early_stop_checks=4,
    )
    data = load_result2_fold(ROOT, fold)
    outer = pd.read_csv(ROOT / "04_SPLITS/PARTICIPANT_OUTER_FOLDS.tsv", sep="\t")
    inner = pd.read_csv(ROOT / "04_SPLITS/PARTICIPANT_INNER_FOLDS.tsv", sep="\t")
    outer_train = outer.loc[outer["outer_fold"].ne(fold), "sample_id_std"].astype(str).tolist()
    outer_test = outer.loc[outer["outer_fold"].eq(fold), "sample_id_std"].astype(str).tolist()
    inner = inner.loc[inner["heldout_outer_fold"].eq(fold)].copy()
    validation_fold = args.index % 3
    inner_train = inner.loc[inner["inner_fold"].ne(validation_fold), "sample_id_std"].astype(str).tolist()
    inner_validation = inner.loc[inner["inner_fold"].eq(validation_fold), "sample_id_std"].astype(str).tolist()
    if set(outer_train) != set(inner["sample_id_std"].astype(str)):
        raise RuntimeError("Inner/outer participant mismatch")

    print(f"[TUNE] fold={fold} seed={seed} train={len(inner_train)} val={len(inner_validation)}", flush=True)
    tune_model, tune_history, tune_metrics, tune_quality = train_model(
        data, inner_train, inner_validation, config, seed, device
    )
    pretrain_epochs, finetune_epochs = best_epochs(tune_history)
    tune_history.to_csv(output / "inner_training_history.tsv", sep="\t", index=False)
    tune_metrics.to_csv(output / "inner_validation_metrics.tsv", sep="\t", index=False)
    (output / "inner_validation_quality.json").write_text(
        json.dumps(tune_quality, indent=2), encoding="utf-8"
    )
    del tune_model
    torch.cuda.empty_cache()
    gc.collect()

    print(f"[REFIT] fold={fold} seed={seed} pre={pretrain_epochs} fine={finetune_epochs}", flush=True)
    model, _ = fit_model_fixed_epochs(
        data,
        outer_train,
        config,
        seed,
        device,
        pretrain_epochs,
        finetune_epochs,
        output,
    )
    predictions, scores, metrics, quality = evaluate_model(
        model, data, outer_test, config, device, return_cell_scores=True, chunk_size=4096
    )
    if scores is None:
        raise RuntimeError("Outer-test cell scores were not produced")
    latent = encode_cell_states(model, data, outer_test, device, chunk_size=4096)
    training_latent = encode_cell_states(model, data, outer_train, device, chunk_size=4096)
    predictions.insert(0, "outer_fold", fold)
    predictions.insert(1, "seed", seed)
    metrics.insert(0, "outer_fold", fold)
    metrics.insert(1, "seed", seed)
    latent.insert(0, "outer_fold", fold)
    latent.insert(1, "seed", seed)
    predictions.to_csv(participant_path, sep="\t", index=False)
    metrics.to_csv(metric_path, sep="\t", index=False)
    latent.to_csv(latent_path, sep="\t", index=False, compression="gzip")

    z_columns = [column for column in latent if column.startswith("z_state_")]
    direction_rows = []
    heldout_coordinate_frames = []
    reference_frames = []
    for task, definition in TASKS.items():
        for lineage in data.celltype_levels:
            train_lineage = training_latent.loc[training_latent["celltype_raw"].eq(lineage)].copy()
            test_lineage = latent.loc[latent["celltype_raw"].eq(lineage)].copy()
            eligible_counts = train_lineage.groupby("sample_id_std", observed=True).size()
            eligible_ids = eligible_counts.loc[eligible_counts.ge(20)].index.astype(str)
            train_lineage = train_lineage.loc[train_lineage["sample_id_std"].isin(eligible_ids)]
            summaries = train_lineage.groupby(
                ["sample_id_std", "analysis_stage"], observed=True
            )[z_columns].median().reset_index()
            reference = summaries.loc[summaries["analysis_stage"].eq(definition["reference"])]
            target = summaries.loc[summaries["analysis_stage"].eq(definition["target"])]
            row = {
                "fold": fold, "seed": seed, "task": task, "celltype_raw": lineage,
                "reference_stage": definition["reference"], "target_stage": definition["target"],
                "reference_participant_n": len(reference), "target_participant_n": len(target),
                "minimum_cells_per_participant": 20,
            }
            if len(reference) < 3 or len(target) < 3:
                row.update({"status": "LOW_SUPPORT", "low_support": True})
                direction_rows.append(row)
                if len(test_lineage):
                    current = test_lineage[["cell_id", "sample_id_std", "analysis_stage", "celltype_raw"]].copy()
                    current["task"] = task
                    current["D"] = np.nan
                    current["c"] = np.nan
                    current["Y"] = np.nan
                    current["coordinate_status"] = "LOW_SUPPORT"
                    heldout_coordinate_frames.append(current)
                continue
            mu0 = reference[z_columns].to_numpy(float).mean(axis=0)
            mu1 = target[z_columns].to_numpy(float).mean(axis=0)
            residuals = np.vstack(
                [reference[z_columns].to_numpy(float) - mu0, target[z_columns].to_numpy(float) - mu1]
            )
            estimator = LedoitWolf(assume_centered=True).fit(residuals)
            delta = mu1 - mu0
            try:
                direction = np.linalg.solve(estimator.covariance_, delta)
            except np.linalg.LinAlgError as error:
                raise RuntimeError(f"Fisher solve failed fold={fold} seed={seed} task={task} lineage={lineage}") from error
            intercept = -0.5 * float((mu1 + mu0) @ direction)
            train_lineage["D"] = train_lineage[z_columns].to_numpy(float) @ direction + intercept
            reference_cells = train_lineage.loc[
                train_lineage["analysis_stage"].eq(definition["reference"])
            ].copy()
            shift = float(reference_cells["D"].median())
            reference_cells["c"] = shift
            reference_cells["Y"] = np.exp(np.clip(reference_cells["D"].to_numpy(float) - shift, -80, 80))
            reference_cells["task"] = task
            reference_cells["fold"] = fold
            reference_cells["seed"] = seed
            reference_frames.append(
                reference_cells[["fold", "seed", "task", "cell_id", "sample_id_std", "celltype_raw", "D", "c", "Y"]]
            )
            if len(test_lineage):
                current = test_lineage[["cell_id", "sample_id_std", "analysis_stage", "celltype_raw"]].copy()
                current["task"] = task
                current["D"] = test_lineage[z_columns].to_numpy(float) @ direction + intercept
                current["c"] = shift
                current["Y"] = np.exp(np.clip(current["D"].to_numpy(float) - shift, -80, 80))
                current["coordinate_status"] = "PASS"
                heldout_coordinate_frames.append(current)
            reference_per_participant = reference_cells.groupby("sample_id_std", observed=True).size()
            n_balance = int(min(2000, reference_per_participant.min()))
            row.update(
                {
                    "status": "PASS", "low_support": False,
                    "low_reference_support": n_balance < 100, "n_balance": n_balance,
                    "shrinkage_lambda": float(estimator.shrinkage_), "intercept_b": intercept,
                    "reference_median_c": shift, "mu0_json": json.dumps(mu0.tolist()),
                    "mu1_json": json.dumps(mu1.tolist()), "w_json": json.dumps(direction.tolist()),
                }
            )
            direction_rows.append(row)

    coordinate = pd.concat(heldout_coordinate_frames, ignore_index=True)
    coordinate.insert(0, "fold", fold)
    coordinate.insert(1, "seed", seed)
    reference_cells = pd.concat(reference_frames, ignore_index=True) if reference_frames else pd.DataFrame()
    coordinate.to_csv(
        ROOT / f"11_FISHER_COORDINATE/fold_{fold}_seed_{seed}_heldout.tsv.gz",
        sep="\t", index=False, compression="gzip"
    )
    reference_cells.to_csv(
        ROOT / f"11_FISHER_COORDINATE/fold_{fold}_seed_{seed}_training_reference.tsv.gz",
        sep="\t", index=False, compression="gzip"
    )
    pd.DataFrame(direction_rows).to_csv(
        ROOT / f"11_FISHER_COORDINATE/fold_{fold}_seed_{seed}_directions.tsv",
        sep="\t", index=False
    )
    (output / "outer_test_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    model_path = output / "model.pt"
    manifest = {
        "status": "PASS",
        "fold": fold,
        "seed": seed,
        "outer_train_participant_n": len(outer_train),
        "outer_test_participant_n": len(outer_test),
        "outer_test_cell_n": len(latent),
        "pretrain_epochs": pretrain_epochs,
        "finetune_epochs": finetune_epochs,
        "model_sha256": sha256(model_path),
        "fisher_direction_pass_n": int(sum(row.get("status") == "PASS" for row in direction_rows)),
        "config": asdict(config),
        "annotation_column": "celltype_raw",
    }
    (output / "DONE.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"V3_OOF_JOB=PASS fold={fold} seed={seed} cells={len(latent)}", flush=True)


if __name__ == "__main__":
    main()
