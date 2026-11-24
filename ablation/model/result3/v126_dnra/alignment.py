"""Full-gene pseudobulk alignment with isolated CORAL/MMD ablations."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from v126_dnra.sc_representation import _canonical_module, build_sc_model


def alignment_loss_weights(variant: str, config: dict) -> dict[str, float]:
    weights = {"coral_weight": float(config["coral_weight"]), "mmd_weight": float(config["mmd_weight"])}
    if variant == "NO_CORAL":
        weights["coral_weight"] = 0.0
    elif variant == "NO_MMD":
        weights["mmd_weight"] = 0.0
    return weights


def validate_bulk_gene_order(observed: np.ndarray, expected: list[str]) -> None:
    if len(observed) != len(expected) or not np.array_equal(np.asarray(observed, dtype=str), np.asarray(expected, dtype=str)):
        raise ValueError("BULK_GENE_ORDER_CONTRACT_FAILED")


def bulk_log2cpm_to_ln(values: np.ndarray) -> np.ndarray:
    """Use the historical natural-log scale expected by the alignment adapter."""
    return np.asarray(values, dtype=np.float32) * np.float32(np.log(2.0))


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.T @ source_centered / max(1, len(source) - 1)
    target_cov = target_centered.T @ target_centered / max(1, len(target) - 1)
    return functional.mse_loss(source.mean(dim=0), target.mean(dim=0)) + functional.mse_loss(source_cov, target_cov)


def mmd_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = source.new_tensor(0.0)
    for bandwidth in (0.5, 1.0, 2.0, 4.0):
        gamma = 1.0 / (2.0 * bandwidth * bandwidth)
        xx = torch.cdist(source, source).square()
        yy = torch.cdist(target, target).square()
        xy = torch.cdist(source, target).square()
        result = result + torch.exp(-gamma * xx).mean() + torch.exp(-gamma * yy).mean() - 2.0 * torch.exp(-gamma * xy).mean()
    return result / 4.0


def fit_alignment(
    anchor_expression: np.ndarray,
    anchor_latent: np.ndarray,
    anchor_partition: np.ndarray,
    bulk_expression: np.ndarray,
    sc_checkpoint: dict,
    alignment_config: dict,
    variant: str,
    seed: int,
    device: str | torch.device,
) -> tuple[dict, list[dict]]:
    if anchor_expression.ndim != 2 or anchor_expression.shape[1] != 16930 or bulk_expression.shape[1] != 16930:
        raise ValueError("ALIGNMENT_REQUIRES_16930_FEATURES")
    if anchor_latent.shape != (len(anchor_expression), 32):
        raise ValueError("ALIGNMENT_LATENT_SHAPE_INVALID")
    train = np.flatnonzero(anchor_partition == "train")
    validation = np.flatnonzero(anchor_partition == "validation")
    if not len(train) or not len(validation):
        raise ValueError("ALIGNMENT_PARTITION_EMPTY")
    torch.manual_seed(seed)
    np.random.seed(seed)
    target = torch.device(device)
    sc = sc_checkpoint["config"]
    sc_model = build_sc_model(16930, sc["latent_dim"], sc["hidden_dim"], sc["dropout"], sc_checkpoint["stage_loss_weight"])
    sc_model.load_state_dict(sc_checkpoint["model_state"], strict=True)
    canonical = _canonical_module()
    model = canonical.PseudobulkAlignmentModel(16930, sc["hidden_dim"], 32, sc["dropout"])
    model.initialize_from_single_cell(sc_model)
    del sc_model
    model.to(target)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in {"sc_gene_scale", "sc_gene_shift", "bulk_gene_scale", "bulk_gene_shift"})
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=alignment_config["learning_rate"], weight_decay=alignment_config["weight_decay"])
    weights = alignment_loss_weights(variant, alignment_config)
    rng = np.random.default_rng(seed)
    best_loss, best_state, stale, rows = np.inf, None, 0, []
    for epoch in range(1, alignment_config["epochs"] + 1):
        size = min(alignment_config["batch_size"], len(train))
        ai = rng.choice(train, size=size, replace=False)
        bi = rng.choice(len(bulk_expression), size=size, replace=len(bulk_expression) < size)
        ax = torch.as_tensor(anchor_expression[ai], dtype=torch.float32, device=target)
        az = torch.as_tensor(anchor_latent[ai], dtype=torch.float32, device=target)
        bx = torch.as_tensor(bulk_expression[bi], dtype=torch.float32, device=target)
        model.train(); optimizer.zero_grad(set_to_none=True)
        predicted = model(ax, "single_cell_pseudobulk")
        bulk_latent = model(bx, "bulk")
        mse = functional.mse_loss(predicted, az)
        coral = coral_loss(bulk_latent, az)
        mmd = mmd_loss(bulk_latent, az)
        prior = (model.sc_gene_scale - 1).square().mean() + model.sc_gene_shift.square().mean() + (model.bulk_gene_scale - 1).square().mean() + model.bulk_gene_shift.square().mean()
        total = mse + weights["coral_weight"] * coral + weights["mmd_weight"] * mmd + alignment_config["adapter_prior_weight"] * prior
        if not torch.isfinite(total):
            raise RuntimeError("NONFINITE_ALIGNMENT_LOSS")
        total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        optimizer.step()
        model.eval()
        with torch.no_grad():
            vx = torch.as_tensor(anchor_expression[validation], dtype=torch.float32, device=target)
            vz = torch.as_tensor(anchor_latent[validation], dtype=torch.float32, device=target)
            validation_loss = float(functional.mse_loss(model(vx, "single_cell_pseudobulk"), vz).cpu())
        rows.append({"epoch": epoch, "total_loss": float(total.detach().cpu()), "mse_loss": float(mse.detach().cpu()), "coral_loss": float(coral.detach().cpu()), "mmd_loss": float(mmd.detach().cpu()), "validation_mse": validation_loss, "gradient_norm": gradient_norm, "variant": variant, "seed": seed})
        if validation_loss < best_loss - alignment_config["early_stopping_min_delta"]:
            best_loss, stale = validation_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
        else:
            stale += 1
        if epoch >= alignment_config["early_stopping_minimum_epochs"] and stale >= alignment_config["early_stopping_patience"]:
            break
    if best_state is None:
        raise RuntimeError("ALIGNMENT_BEST_STATE_MISSING")
    return {"format": "V126_DE_NOVO_ALIGNMENT", "variant": variant, "seed": seed, "n_genes": 16930, "latent_dim": 32, "model_state": best_state, "best_epoch": best_epoch, "best_validation_mse": best_loss, "loss_weights": weights}, rows
