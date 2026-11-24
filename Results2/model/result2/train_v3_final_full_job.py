#!/usr/bin/env python3
"""Train frozen final TB-STATE V3 models on all clean training participants."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf

from tb_state_sc_v3_training import TrainConfig, encode_cell_states, fit_model_fixed_epochs, load_result2_fold


ROOT = Path(os.environ.get("TBSPIN_R2_RUN", "results2_run"))
SEEDS = (20260828, 20260829, 20260830)
TASKS = {"T1": {"reference": "HC", "target": "TBINF"}, "T2": {"reference": "TBINF", "target": "ATB"}}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_epochs(seed: int) -> tuple[int, int, list[dict[str, int]]]:
    rows: list[dict[str, int]] = []
    for fold in range(5):
        path = ROOT / f"08_OOF_MODELS/fold_{fold}/seed_{seed}/DONE.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "fold": fold,
            "pretrain_epochs": int(payload["pretrain_epochs"]),
            "finetune_epochs": int(payload["finetune_epochs"]),
        })
    pre = max(1, int(np.rint(np.median([x["pretrain_epochs"] for x in rows]))))
    fine = max(1, int(np.rint(np.median([x["finetune_epochs"] for x in rows]))))
    return pre, fine, rows


def fisher_layers(latent: pd.DataFrame, data, seed: int, output: Path) -> None:
    z_columns = [column for column in latent if column.startswith("z_state_")]
    direction_rows: list[dict[str, object]] = []
    reference_frames: list[pd.DataFrame] = []
    for task, definition in TASKS.items():
        for lineage in data.celltype_levels:
            lineage_data = latent.loc[latent.celltype_raw.eq(lineage)].copy()
            eligible_counts = lineage_data.groupby("sample_id_std", observed=True).size()
            eligible_ids = eligible_counts.loc[eligible_counts.ge(20)].index.astype(str)
            lineage_data = lineage_data.loc[lineage_data.sample_id_std.isin(eligible_ids)]
            summaries = lineage_data.groupby(["sample_id_std", "analysis_stage"], observed=True)[z_columns].median().reset_index()
            reference = summaries.loc[summaries.analysis_stage.eq(definition["reference"])]
            target = summaries.loc[summaries.analysis_stage.eq(definition["target"])]
            row: dict[str, object] = {
                "seed": seed, "task": task, "celltype_raw": lineage,
                "reference_stage": definition["reference"], "target_stage": definition["target"],
                "reference_participant_n": len(reference), "target_participant_n": len(target),
                "minimum_cells_per_participant": 20,
            }
            if len(reference) < 3 or len(target) < 3:
                row.update({"status": "LOW_SUPPORT", "low_support": True})
                direction_rows.append(row)
                continue
            mu0 = reference[z_columns].to_numpy(float).mean(axis=0)
            mu1 = target[z_columns].to_numpy(float).mean(axis=0)
            residuals = np.vstack([reference[z_columns].to_numpy(float) - mu0, target[z_columns].to_numpy(float) - mu1])
            estimator = LedoitWolf(assume_centered=True).fit(residuals)
            direction = np.linalg.solve(estimator.covariance_, mu1 - mu0)
            intercept = -0.5 * float((mu1 + mu0) @ direction)
            lineage_data["D"] = lineage_data[z_columns].to_numpy(float) @ direction + intercept
            reference_cells = lineage_data.loc[lineage_data.analysis_stage.eq(definition["reference"])].copy()
            shift = float(reference_cells.D.median())
            reference_cells["c"] = shift
            reference_cells["Y"] = np.exp(np.clip(reference_cells.D.to_numpy(float) - shift, -80, 80))
            reference_cells["task"] = task
            reference_cells["seed"] = seed
            reference_frames.append(reference_cells[["seed", "task", "cell_id", "sample_id_std", "celltype_raw", "D", "c", "Y"]])
            per_participant = reference_cells.groupby("sample_id_std", observed=True).size()
            n_balance = int(min(2000, per_participant.min()))
            row.update({
                "status": "PASS", "low_support": False, "low_reference_support": n_balance < 100,
                "n_balance": n_balance, "shrinkage_lambda": float(estimator.shrinkage_),
                "intercept_b": intercept, "reference_median_c": shift,
                "mu0_json": json.dumps(mu0.tolist()), "mu1_json": json.dumps(mu1.tolist()),
                "w_json": json.dumps(direction.tolist()),
            })
            direction_rows.append(row)
    pd.DataFrame(direction_rows).to_csv(output / "fisher_directions.tsv", sep="\t", index=False)
    pd.concat(reference_frames, ignore_index=True).to_csv(
        output / "training_reference.tsv.gz", sep="\t", index=False, compression="gzip"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True, choices=range(3))
    args = parser.parse_args()
    seed = SEEDS[args.index]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    output = ROOT / f"17_FINAL_FULL_MODEL/seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    final_hvg_alias = ROOT / "05_FEATURES/FOLD_FINAL_HVG5000.tsv"
    if not final_hvg_alias.exists():
        raise FileNotFoundError(f"missing final-HVG alias: {final_hvg_alias}")
    data = load_result2_fold(ROOT, "FINAL")
    participants = sorted(data.participant_to_cells)
    pretrain_epochs, finetune_epochs, fold_epochs = selected_epochs(seed)
    config = TrainConfig(
        state_dim=8, tau=0.5, lambda_cross=0.01, lambda_id=0.20, beta_kl=0.05,
        lr_finetune_encoder=1e-4, likelihood="zinb", identity_dim=16, hidden_dim=256,
        lambda_ordinal=0.15, lr_pretrain=5e-4, lr_heads=5e-4, weight_decay=1e-4,
        max_pretrain_epochs=100, max_finetune_epochs=150, participants_per_batch=8,
        cells_per_participant=512, gradient_clip=5.0, eval_every=5, early_stop_checks=4,
    )
    print(f"[FINAL] seed={seed} participants={len(participants)} pre={pretrain_epochs} fine={finetune_epochs}", flush=True)
    model, history = fit_model_fixed_epochs(
        data, participants, config, seed, device, pretrain_epochs, finetune_epochs, output
    )
    latent = encode_cell_states(model, data, participants, device, chunk_size=4096)
    latent.insert(0, "seed", seed)
    latent.to_csv(output / "training_latent.tsv.gz", sep="\t", index=False, compression="gzip")
    fisher_layers(latent, data, seed, output)
    model_path = output / "model.pt"
    payload = {
        "status": "PASS", "seed": seed, "participant_n": len(participants), "cell_n": len(latent),
        "pretrain_epochs": pretrain_epochs, "finetune_epochs": finetune_epochs,
        "fold_epoch_inputs": fold_epochs, "epoch_selection_rule": "MEDIAN_ACROSS_OUTER_FOLDS",
        "model_sha256": sha256(model_path), "annotation_column": "celltype_raw",
        "hvg_n": 5000, "config": asdict(config),
    }
    (output / "DONE.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"V3_FINAL_FULL=PASS seed={seed} model_sha256={payload['model_sha256']}", flush=True)


if __name__ == "__main__":
    main()
