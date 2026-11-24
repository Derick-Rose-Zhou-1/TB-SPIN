from __future__ import annotations
import copy,math,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
import tb_state_sc_v3_training as training
from tb_state_sc_v3_training import PreparedData, TrainConfig
from tbstate_sc_v3_model import TBStateSCV3 as TBStateSCV2
TASKS={"B1":{"negative":"ATB","positive":"DTB"},"B2":{"negative":"ATB","positive":"TPE"},"B3":{"negative":"ATB","positive":"TBM"}}
def build_model(
    data: PreparedData,
    config: TrainConfig,
    device: torch.device,
    result2_checkpoint: Path | None,
    freeze_encoder: bool,
) -> TBStateSCV2:
    model = TBStateSCV2(
        n_genes=data.counts.shape[1],
        n_major_celltypes=len(data.celltype_levels),
        n_technical_batches=1,
        transition_names=TASKS,
        state_dim=config.state_dim,
        tau=config.tau,
        identity_dim=config.identity_dim,
        hidden_dim=config.hidden_dim,
        likelihood=config.likelihood,
    ).to(device)
    if result2_checkpoint is not None:
        checkpoint = torch.load(result2_checkpoint, map_location="cpu")
        if checkpoint["gene_ids"] != data.gene_ids:
            raise RuntimeError("Warm-start Result2 genes do not exactly match Result6 genes/order")
        source = checkpoint["state_dict"]
        target = model.state_dict()
        excluded = ("transition_heads.", "ordinal_head.", "identity_head.")
        transferable = {
            key: value
            for key, value in source.items()
            if not key.startswith(excluded) and key in target and target[key].shape == value.shape
        }
        target.update(transferable)
        source_major = [str(value) for value in checkpoint["celltype_levels"]]
        for level in sorted(set(source_major) & set(data.celltype_levels)):
            old_index = source_major.index(level)
            new_index = data.celltype_levels.index(level)
            target["identity_head.weight"][new_index] = source["identity_head.weight"][old_index]
            target["identity_head.bias"][new_index] = source["identity_head.bias"][old_index]
        model.load_state_dict(target)
    if freeze_encoder:
        for module in (model.encoder, model.id_mu, model.id_logvar, model.state_mu, model.state_logvar):
            for parameter in module.parameters():
                parameter.requires_grad = False
    return model

def train_result6_model(
    data: PreparedData,
    train_participants: list[str],
    validation_participants: list[str] | None,
    config: TrainConfig,
    seed: int,
    device: torch.device,
    checkpoint: Path | None,
    freeze_encoder: bool,
    adaptation_epochs: int,
    finetune_epochs: int,
    early_stop: bool,
    output_dir: Path,
) -> tuple[TBStateSCV2, int, pd.DataFrame]:
    training.seed_everything(seed)
    model = build_model(data, config, device, checkpoint, freeze_encoder)
    amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    rng = np.random.default_rng(seed)
    history: list[dict[str, object]] = []
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if adaptation_epochs:
        optimizer = torch.optim.AdamW(
            trainable, lr=config.lr_finetune_encoder, weight_decay=config.weight_decay
        )
        for epoch in range(1, adaptation_epochs + 1):
            rows = []
            model.train()
            for indices, local_participants, stage_codes in training.participant_batches(
                data, train_participants, config, rng
            ):
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp):
                    loss, components = training.amp_safe_phase_loss(
                        model,
                        data,
                        indices,
                        local_participants,
                        stage_codes,
                        config,
                        device,
                        config.beta_kl,
                        supervised=False, use_amp=amp,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                rows.append(components)
            history.append({"phase": "result6_adaptation", "epoch": epoch, "loss": np.mean([x["loss"] for x in rows])})

    head_parameters = list(model.transition_heads.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    base_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    groups = [{"params": head_parameters, "lr": config.lr_heads}]
    if base_parameters:
        groups.insert(0, {"params": base_parameters, "lr": config.lr_finetune_encoder})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    best_state = None
    best_objective = -math.inf
    best_epoch = finetune_epochs
    stale = 0
    best_tie_state = None
    best_tie_objective = -math.inf
    best_tie_bce = math.inf
    best_tie_epoch = None
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, finetune_epochs + 1):
        rows = []
        model.train()
        for indices, local_participants, stage_codes in training.participant_batches(
            data, train_participants, config, rng
        ):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loss, components = training.amp_safe_phase_loss(
                    model,
                    data,
                    indices,
                    local_participants,
                    stage_codes,
                    config,
                    device,
                    config.beta_kl,
                    supervised=True, use_amp=amp,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            rows.append(components)
        record = {
            "phase": "result6_branch_finetune",
            "epoch": epoch,
            **{k:float(np.mean([x[k] for x in rows])) for k in rows[0]},
        }
        if early_stop and validation_participants and epoch % 2 == 0:
            validation_pred, _, metrics, _ = training.evaluate_model(
                model, data, validation_participants, config, device, False
            )
            values = metrics.loc[metrics["metric"].eq("AUPRC"), "value"].to_numpy(float)
            objective = float(values.mean()) if len(values) == len(TASKS) else -math.inf
            record["validation_mean_branch_AUPRC"] = objective
            validation_pred.to_csv(output_dir / f"validation_epoch_{epoch}.tsv", sep="\t", index=False)
            branch_bce = []
            for task, sub in validation_pred.groupby("task"):
                p = np.clip(sub.probability.to_numpy(float), 1e-7, 1-1e-7)
                y = sub.y_true.to_numpy(float)
                bce = float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p)))
                branch_bce.append(bce)
                record[f"validation_{task}_BCE"] = bce
                record[f"validation_{task}_AUPRC"] = float(metrics.loc[(metrics.task==task)&(metrics.metric=="AUPRC"),"value"].iloc[0])
            bce = float(np.mean(branch_bce))
            record["validation_mean_branch_BCE"] = bce
            if objective > best_tie_objective + 1e-4 or (abs(objective-best_tie_objective)<=1e-4 and bce < best_tie_bce-1e-8):
                best_tie_objective = objective
                best_tie_bce = bce
                best_tie_epoch = epoch
                best_tie_state = copy.deepcopy(model.state_dict())
            if objective > best_objective + 1e-4:
                best_objective = objective
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                # Diagnostic: observe the full locked 100-epoch budget; no patience break.
                record["stale_evaluations"] = stale
        history.append(record)
    if early_stop:
        if best_state is None:
            raise RuntimeError("Result6 inner training produced no three-task validation objective")
        torch.save({"state_dict":best_tie_state}, output_dir / "tie_bce_state.pt")
        (output_dir / "diagnostic_selection.json").write_text(json.dumps({"strict_epoch":best_epoch,"tie_bce_epoch":best_tie_epoch,"strict_AUPRC":best_objective,"tie_AUPRC":best_tie_objective,"tie_BCE":best_tie_bce},indent=2))
        model.load_state_dict(best_tie_state)
        best_epoch = best_tie_epoch
    output_dir.mkdir(parents=True, exist_ok=True)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.tsv", sep="\t", index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "train_config": asdict(config),
            "effective_run_config": {"inner_budget":100,"eval_every":2,"patience_enabled":False,"selection":"mean_3task_AUPRC_then_BCE_then_earliest","AUPRC_tolerance":1e-4,"BCE_tolerance":1e-8,"adaptation_epochs":adaptation_epochs,"selected_or_fixed_epochs":best_epoch if early_stop else finetune_epochs},
            "seed": seed,
            "gene_ids": data.gene_ids,
            "celltype_levels": data.celltype_levels,
            "tasks": TASKS,
            "warm_start": checkpoint is not None,
            "freeze_encoder": freeze_encoder,
            "adaptation_epochs": adaptation_epochs,
            "finetune_epochs": best_epoch if early_stop else finetune_epochs,
        },
        output_dir / "model.pt",
    )
    return model, best_epoch, history_frame
