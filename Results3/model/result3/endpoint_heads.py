import copy
import numpy as np
import torch
from torch import nn
import v126_training as t

def fit_last_binary_artifact(bundle, alignment, training, weights, train_mean):
    t.seed_all(); model = copy.deepcopy(alignment).to(bundle.device)
    for p in model.parameters(): p.requires_grad = False
    for module in (model.shared_linear, model.shared_norm, model.latent_projection):
        for p in module.parameters(): p.requires_grad = True
    head = nn.Sequential(nn.Linear(model.latent_dim, 16), nn.GELU(), nn.Linear(16, 1)).to(bundle.device)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    x = bundle.bulk.loc[training["GSM"], bundle.genes].to_numpy(dtype=np.float32)
    x = train_mean + (x - train_mean) * weights
    tx = torch.from_numpy(x).to(bundle.device)
    y = torch.from_numpy(training["y_true"].to_numpy(dtype=np.float32)).to(bundle.device)
    sw = torch.from_numpy(training["participant_metric_weight"].to_numpy(dtype=np.float32)).to(bundle.device)
    best, state, stale = np.inf, None, 0
    for _ in range(120):
        model.train(); head.train(); opt.zero_grad(set_to_none=True)
        logits = head(model(tx, "bulk")).reshape(-1)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none") * sw).sum() / sw.sum()
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        value = float(loss.detach().cpu())
        if value < best - 1e-6:
            best, stale = value, 0; state = {"alignment": copy.deepcopy(model.state_dict()), "head": copy.deepcopy(head.state_dict())}
        else: stale += 1
        if stale >= 12: break
    model.load_state_dict(state["alignment"]); head.load_state_dict(state["head"]); model.eval(); head.eval()
    return {"kind": "last_block_binary", "alignment": model.cpu(), "head": head.cpu(), "best_loss": best}


def fit_last_cox_artifact(bundle, alignment, training, weights, train_mean):
    t.seed_all(); model = copy.deepcopy(alignment).to(bundle.device)
    for p in model.parameters(): p.requires_grad = False
    for module in (model.shared_linear, model.shared_norm, model.latent_projection):
        for p in module.parameters(): p.requires_grad = True
    head = t.TorchHead(model.latent_dim, "B_LAST_BLOCK_COX16").to(bundle.device)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    x = bundle.bulk.loc[training["GSM"], bundle.genes].to_numpy(dtype=np.float32)
    x = train_mean + (x - train_mean) * weights
    tx = torch.from_numpy(x).to(bundle.device)
    duration = torch.from_numpy(training["duration"].to_numpy(dtype=np.float32)).to(bundle.device)
    event = torch.from_numpy(training["event"].to_numpy(dtype=np.float32)).to(bundle.device)
    sw = torch.from_numpy(training["participant_metric_weight"].to_numpy(dtype=np.float32)).to(bundle.device)
    best, state, stale = np.inf, None, 0
    for _ in range(160):
        model.train(); head.train(); opt.zero_grad(set_to_none=True)
        loss = t.cox_loss(head(model(tx, "bulk")), duration, event, sw)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        value = float(loss.detach().cpu())
        if value < best - 1e-6:
            best, stale = value, 0; state = {"alignment": copy.deepcopy(model.state_dict()), "head": copy.deepcopy(head.state_dict())}
        else: stale += 1
        if stale >= 15: break
    model.load_state_dict(state["alignment"]); head.load_state_dict(state["head"]); model.eval(); head.eval()
    return {"kind": "last_block_cox", "alignment": model.cpu(), "head": head.cpu(), "best_loss": best}


def predict_artifact(artifact, x, endpoint):
    if artifact["kind"] == "sklearn":
        return artifact["estimator"].predict_proba(artifact["scaler"].transform(x))[:, 1]
    if artifact["kind"] == "cox": return t.predict_cox({"scaler": artifact["scaler"], "head": artifact["head"]}, x)
    tx = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        if artifact["kind"] == "last_block_binary": return torch.sigmoid(artifact["head"](artifact["alignment"](tx, "bulk")).reshape(-1)).numpy()
        return artifact["head"](artifact["alignment"](tx, "bulk")).reshape(-1).numpy()
