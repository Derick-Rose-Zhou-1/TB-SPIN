#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROD_ROOT = Path("inputs/TB_STATE_SC_V3_NC_FINAL_20260828")
AUDIT_ROOT = Path("inputs/TB_STATE_RESULT5_2_10_INDEPENDENT_SEEDS_20260913")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Results2/model/result2"))
import tb_state_sc_v3_training as training
training.R2_H5AD = PROD_ROOT / "TB_STATE_SC_V3_INPUT_20260828/train_new.h5ad"
from tbstate_sc_v3_model import TBStateSCV3
from tb_state_sc_v3_training import (
    TASKS, TrainConfig, encode_cell_states, fit_model_fixed_epochs,
    load_result2_fold, seed_everything,
)


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    seed = args.seed
    out = AUDIT_ROOT / "02_NEW_TRAINING" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for new independent training")
    data = load_result2_fold(PROD_ROOT, "FINAL")
    participants = sorted(data.participant_to_cells)
    config = TrainConfig(
        state_dim=8, tau=0.5, lambda_cross=0.01, lambda_id=0.20,
        beta_kl=0.05, lr_finetune_encoder=1e-4, likelihood="zinb",
        identity_dim=16, hidden_dim=256, lambda_ordinal=0.15,
        lr_pretrain=5e-4, lr_heads=5e-4, weight_decay=1e-4,
        max_pretrain_epochs=100, max_finetune_epochs=150,
        participants_per_batch=8, cells_per_participant=512,
        gradient_clip=5.0, eval_every=5, early_stop_checks=4,
    )
    init_model = TBStateSCV3(
        n_genes=data.counts.shape[1], n_major_celltypes=len(data.celltype_levels),
        n_technical_batches=1, transition_names=TASKS,
        state_dim=config.state_dim, tau=config.tau,
        identity_dim=config.identity_dim, hidden_dim=config.hidden_dim,
        likelihood=config.likelihood,
    ).to(device)
    init_hash = state_hash(init_model)
    del init_model
    torch.cuda.empty_cache()
    (out / "initial_parameter_sha256.txt").write_text(init_hash + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps({
        "seed": seed, "source_architecture": str(PROD_ROOT / "scripts/tbstate_sc_v3_model.py"),
        "source_training": str(PROD_ROOT / "scripts/tb_state_sc_v3_training.py"),
        "input_h5ad": str(PROD_ROOT / "TB_STATE_SC_V3_INPUT_20260828/train_new.h5ad"),
        "participant_n": len(participants), "cell_n": int(data.counts.shape[0]),
        "fixed_pretrain_epochs": 85, "fixed_finetune_epochs": 10,
        "config": asdict(config), "resume": False, "checkpoint_load": False,
    }, indent=2), encoding="utf-8")
    model, history = fit_model_fixed_epochs(
        data, participants, config, seed, device, 85, 10, out
    )
    latent = encode_cell_states(model, data, participants, device, chunk_size=4096)
    latent.insert(0, "training_seed", seed)
    latent_path = AUDIT_ROOT / "03_LATENTS" / f"seed_{seed}_latents.parquet"
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    latent.to_parquet(latent_path, index=False)
    final_hash = file_hash(out / "model.pt")
    latent_hash = file_hash(latent_path)
    (out / "final_checkpoint_sha256.txt").write_text(final_hash + "\n", encoding="utf-8")
    (out / "latent_sha256.txt").write_text(latent_hash + "\n", encoding="utf-8")
    (out / "DONE.json").write_text(json.dumps({
        "status": "PASS", "training_seed": seed, "training_from_scratch": True,
        "resume": False, "checkpoint_load": False,
        "initial_parameter_sha256": init_hash,
        "checkpoint": str(out / "model.pt"), "checkpoint_sha256": final_hash,
        "latent": str(latent_path), "latent_sha256": latent_hash,
        "n_cells": len(latent), "z_id_dim": 16, "z_state_dim": 8,
        "participant_n": len(participants),
    }, indent=2), encoding="utf-8")
    print(f"TRAIN_COMPLETE seed={seed} checkpoint_sha256={final_hash} latent_sha256={latent_hash}", flush=True)


if __name__ == "__main__":
    main()
