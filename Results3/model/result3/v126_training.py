from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.special import expit
from scipy.stats import norm, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from v126_common import (
    CODE_ROOT,
    FORMAL_COHORTS,
    PROJECT_ROOT,
    assert_no_figures,
    atomic_json,
    atomic_tsv,
    read_current_run,
    resolve_sources,
    sha256,
    utc_now,
)


SEED = 20260815
GENE_CONDITIONS = ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
BINARY_MODELS = (
    "A1_LINEAR", "A2_MLP16", "A3_MLP32", "A4_MLP32_8", "A5_ELASTIC_NET",
    "B_LAST_BLOCK_MLP16", "C_MULTITASK_SHARED16",
)
COX_MODELS = ("C1_LINEAR", "C2_PENALISED", "C3_DEEP16", "C4_DEEP32", "C5_DEEP32_8", "B_LAST_BLOCK_COX16", "C_MULTITASK_SHARED16")
MODEL_COMPLEXITY = {
    "A1_LINEAR": 1, "A5_ELASTIC_NET": 2, "A2_MLP16": 3, "A3_MLP32": 4,
    "A4_MLP32_8": 5, "B_LAST_BLOCK_MLP16": 6,
    "C1_LINEAR": 1, "C2_PENALISED": 2, "C3_DEEP16": 3, "C4_DEEP32": 4,
    "C5_DEEP32_8": 5, "B_LAST_BLOCK_COX16": 6, "C_MULTITASK_SHARED16": 4,
}


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    values = p[valid]
    if not len(values):
        return out
    order = np.argsort(values)
    ranks = np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate((values[order] * len(values) / ranks)[::-1])[::-1]
    restored = np.empty_like(values)
    restored[order] = np.clip(adjusted, 0, 1)
    out[valid] = restored
    return out


def load_canonical_module():
    path = Path(__file__).resolve().with_name("tbstate_v12_models.py")
    spec = importlib.util.spec_from_file_location("tbstate_v12_models_v126", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CANONICAL = load_canonical_module()
PseudobulkAlignmentModel = CANONICAL.PseudobulkAlignmentModel


@dataclass
class DataBundle:
    run: Path
    sources: dict[str, Path]
    endpoints: pd.DataFrame
    bulk: pd.DataFrame
    genes: list[str]
    device: torch.device


def load_bundle() -> DataBundle:
    run = read_current_run()
    sources = resolve_sources()
    endpoints = pd.read_csv(run / "02_ENDPOINTS" / "PARTICIPANT_ENDPOINTS.tsv", sep="\t", low_memory=False)
    endpoints["participant_id"] = endpoints["participant_id"].astype(str)
    endpoints["GSM"] = endpoints["GSM"].astype(str)
    bulk = joblib.load(sources["bulk_expression"])
    bulk.index = bulk.index.astype(str)
    genes = pd.read_csv(sources["gene_list"], sep="\t")["gene_symbol"].astype(str).tolist()
    if list(bulk.columns.astype(str)) != genes:
        raise RuntimeError("BULK_GENE_ORDER_DRIFT")
    missing = sorted(set(endpoints["GSM"]) - set(bulk.index))
    if missing:
        raise RuntimeError(f"ENDPOINT_GSM_MISSING_FROM_BULK:{missing[:10]}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return DataBundle(run, sources, endpoints, bulk, genes, device)


def outer_key(cohort: str) -> str:
    return cohort


def load_alignment(bundle: DataBundle, heldout: str) -> nn.Module:
    payload = torch.load(bundle.sources[f"v125_model_{outer_key(heldout)}"], map_location="cpu", weights_only=False)
    model = PseudobulkAlignmentModel(**payload["alignment_config"])
    model.load_state_dict(payload["alignment_state"], strict=True)
    return model.to(bundle.device)


def frame_for(bundle: DataBundle, endpoint: str, cohorts: set[str]) -> pd.DataFrame:
    frame = bundle.endpoints.loc[
        bundle.endpoints["endpoint"].astype(str).eq(endpoint)
        & bundle.endpoints["cohort"].astype(str).isin(cohorts)
        & bundle.endpoints["eligible"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    frame["participant_metric_weight"] = pd.to_numeric(frame["participant_metric_weight"], errors="coerce").fillna(1.0)
    if endpoint == "E_COX":
        frame["duration"] = pd.to_numeric(frame["time_to_event_or_censor_days"], errors="raise")
        frame["event"] = frame["y_true"].astype(int)
    return frame.reset_index(drop=True)


def build_fold_manifest(bundle: DataBundle) -> pd.DataFrame:
    rows = []
    for heldout in FORMAL_COHORTS:
        training_cohorts = set(FORMAL_COHORTS) - {heldout}
        for endpoint in ("E_ALL", "E_6M", "E_12M", "E_COX"):
            frame = frame_for(bundle, endpoint, training_cohorts)
            if frame.empty:
                continue
            available = sorted(frame["cohort"].unique())
            cohort_splits = []
            if len(available) >= 2:
                for validation_cohort in available:
                    validation = frame[frame["cohort"].eq(validation_cohort)]
                    training = frame[~frame["cohort"].eq(validation_cohort)]
                    if validation["y_true"].nunique() == 2 and training["y_true"].nunique() == 2:
                        cohort_splits.append((f"LODO_{validation_cohort}", set(validation["participant_id"])))
            if not cohort_splits:
                participant = frame.groupby("participant_id", as_index=False).agg(y=("y_true", "max"))
                n_events = int(participant["y"].sum())
                n_splits = min(3, n_events, len(participant) - n_events)
                if n_splits < 2:
                    rows.append({"outer_heldout": heldout, "endpoint": endpoint, "inner_split": "STRUCTURAL_NE", "role": "NA", "participant_id": "", "cohort": "", "status": "STRUCTURAL_NE_INSUFFICIENT_CLASSES"})
                    continue
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
                for fold, (_, val_idx) in enumerate(splitter.split(participant, participant["y"])):
                    cohort_splits.append((f"PARTICIPANT_CV_{fold}", set(participant.iloc[val_idx]["participant_id"])))
            for split_name, validation_ids in cohort_splits:
                for _, row in frame[["participant_id", "cohort"]].drop_duplicates().iterrows():
                    role = "validation" if row["participant_id"] in validation_ids else "training"
                    rows.append({"outer_heldout": heldout, "endpoint": endpoint, "inner_split": split_name, "role": role, "participant_id": row["participant_id"], "cohort": row["cohort"], "status": "PASS"})
    manifest = pd.DataFrame(rows)
    valid = manifest[manifest["status"].eq("PASS")]
    check = valid.groupby(["outer_heldout", "endpoint", "inner_split", "participant_id"])["role"].nunique()
    if (check > 1).any():
        raise RuntimeError("PARTICIPANT_LEAKAGE_IN_MANIFEST")
    return manifest


def _standardised_effect(x1: np.ndarray, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n1, n0 = len(x1), len(x0)
    m1, m0 = x1.mean(axis=0), x0.mean(axis=0)
    v1, v0 = x1.var(axis=0, ddof=1), x0.var(axis=0, ddof=1)
    pooled = np.sqrt(np.maximum(((n1 - 1) * v1 + (n0 - 1) * v0) / max(n1 + n0 - 2, 1), 1e-12))
    effect = (m1 - m0) / pooled
    se = np.sqrt(np.maximum(v1 / max(n1, 1) + v0 / max(n0, 1), 1e-12)) / pooled
    return effect, se


def binary_gene_statistics(bundle: DataBundle, frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    effects, variances, cohort_signs = [], [], []
    x_all = bundle.bulk.loc[frame["GSM"], bundle.genes].to_numpy(dtype=np.float32)
    for cohort in sorted(frame["cohort"].unique()):
        idx = frame["cohort"].eq(cohort).to_numpy()
        y = frame.loc[idx, "y_true"].to_numpy(dtype=int)
        x = x_all[idx]
        if len(np.unique(y)) < 2 or y.sum() < 2:
            continue
        effect, se = _standardised_effect(x[y == 1], x[y == 0])
        effects.append(effect)
        variances.append(np.maximum(se**2, 1e-8))
        cohort_signs.append(np.sign(effect))
    if not effects:
        return pd.DataFrame({"gene": bundle.genes, "meta_effect": np.nan, "meta_p": np.nan, "meta_fdr": np.nan, "direction_consistent_n": 0, "available_contexts": 0, "bootstrap_stability": np.nan})
    e = np.vstack(effects)
    v = np.vstack(variances)
    w = 1.0 / v
    meta = (w * e).sum(axis=0) / w.sum(axis=0)
    meta_se = np.sqrt(1.0 / w.sum(axis=0))
    p = 2.0 * norm.sf(np.abs(meta / np.maximum(meta_se, 1e-12)))
    fdr = bh_fdr(p)
    signs = np.vstack(cohort_signs)
    consistency = (signs == np.sign(meta)).sum(axis=0)
    stability = norm.cdf(np.abs(meta) / np.maximum(meta_se, 1e-12))
    return pd.DataFrame({"gene": bundle.genes, "meta_effect": meta, "meta_p": p, "meta_fdr": fdr, "direction_consistent_n": consistency, "available_contexts": e.shape[0], "bootstrap_stability": stability, "stability_method": "normal_approximation_to_training_only_participant_bootstrap"})


def cox_gene_statistics(bundle: DataBundle, frame: pd.DataFrame) -> pd.DataFrame:
    x = bundle.bulk.loc[frame["GSM"], bundle.genes].to_numpy(dtype=np.float64)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0, ddof=1), 1e-8)
    time = frame["duration"].to_numpy(dtype=float)
    event = frame["event"].to_numpy(dtype=int)
    order = np.argsort(-time)
    xo, eo = x[order], event[order]
    cumulative_x = np.cumsum(xo, axis=0)
    cumulative_x2 = np.cumsum(xo * xo, axis=0)
    risk_n = np.arange(1, len(xo) + 1)[:, None]
    mean_risk = cumulative_x / risk_n
    var_risk = np.maximum(cumulative_x2 / risk_n - mean_risk**2, 1e-10)
    mask = eo.astype(bool)
    score = (xo[mask] - mean_risk[mask]).sum(axis=0)
    info = var_risk[mask].sum(axis=0)
    beta = score / np.maximum(info, 1e-10)
    se = 1.0 / np.sqrt(np.maximum(info, 1e-10))
    p = 2.0 * norm.sf(np.abs(beta / se))
    fdr = bh_fdr(p)
    stability = norm.cdf(np.abs(beta) / np.maximum(se, 1e-12))
    return pd.DataFrame({"gene": bundle.genes, "meta_effect": beta, "HR": np.exp(np.clip(beta, -10, 10)), "meta_p": p, "meta_fdr": fdr, "direction_consistent_n": 1, "available_contexts": 1, "bootstrap_stability": stability, "stability_method": "Cox_score_normal_approximation_training_only"})


def condition_mask(stats: pd.DataFrame, condition: str, endpoint: str) -> np.ndarray:
    if condition == "G0":
        return np.ones(len(stats), dtype=bool)
    fdr = stats["meta_fdr"].to_numpy(dtype=float)
    effect = np.abs(stats["meta_effect"].to_numpy(dtype=float))
    consistency = stats["direction_consistent_n"].to_numpy(dtype=float)
    available = stats["available_contexts"].to_numpy(dtype=float)
    stability = stats["bootstrap_stability"].to_numpy(dtype=float)
    required = np.minimum(2, np.maximum(1, np.ceil(2 * available / 3)))
    rules = {
        "G1": (fdr < .10) & (effect > .20),
        "G2": (fdr < .05) & (effect > .20),
        "G3": (fdr < .05) & (effect > .35),
        "G4": (fdr < .05) & (effect > .50),
        "G5": (fdr < .01) & (effect > .35),
        "G6": (fdr < .01) & (effect > .50),
        "G7": (fdr < .05) & (effect > .35) & (consistency >= required) & (stability >= .60),
        "G8": (fdr < .05) & (effect > .50) & (consistency >= required) & (stability >= .70),
    }
    return rules[condition]


def gene_weights(stats: pd.DataFrame, condition: str, endpoint: str) -> tuple[np.ndarray, int]:
    mask = condition_mask(stats, condition, endpoint)
    if condition == "G0":
        return np.ones(len(stats), dtype=np.float32), len(stats)
    weights = np.ones(len(stats), dtype=np.float32)
    effect = np.abs(stats["meta_effect"].to_numpy(dtype=float))
    weights[mask] = 1.0 + np.minimum(effect[mask], 1.5)
    return weights, int(mask.sum())


def latent_features(bundle: DataBundle, alignment: nn.Module, frame: pd.DataFrame, weights: np.ndarray, train_mean: np.ndarray) -> np.ndarray:
    x = bundle.bulk.loc[frame["GSM"], bundle.genes].to_numpy(dtype=np.float32)
    x = train_mean + (x - train_mean) * weights
    alignment.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            tensor = torch.from_numpy(x[start:start + 256]).to(bundle.device)
            outputs.append(alignment(tensor, "bulk").cpu().numpy())
    return np.concatenate(outputs, axis=0)


def binary_estimator(name: str):
    if name == "A1_LINEAR":
        return LogisticRegression(C=.1, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED)
    if name == "A5_ELASTIC_NET":
        return LogisticRegression(C=.1, penalty="elasticnet", l1_ratio=.5, solver="saga", max_iter=2000, random_state=SEED)
    hidden = {"A2_MLP16": (16,), "A3_MLP32": (32,), "A4_MLP32_8": (32, 8)}[name]
    return MLPClassifier(hidden_layer_sizes=hidden, alpha=.01, learning_rate_init=.001, max_iter=200, early_stopping=True, validation_fraction=.2, n_iter_no_change=12, random_state=SEED)


def binary_metrics(y, score, weight, threshold=.5) -> dict:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if len(np.unique(y)) < 2:
        return {"AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan, "status": "STRUCTURAL_NE_SINGLE_CLASS"}
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, sample_weight=weight, labels=[0, 1]).ravel()
    return {
        "AUROC": roc_auc_score(y, score, sample_weight=weight),
        "AUPRC": average_precision_score(y, score, sample_weight=weight),
        "Brier": np.average((score - y) ** 2, weights=weight),
        "sensitivity": tp / max(tp + fn, 1e-12), "specificity": tn / max(tn + fp, 1e-12),
        "PPV": tp / max(tp + fp, 1e-12), "NPV": tn / max(tn + fn, 1e-12),
        "status": "ESTIMABLE",
    }


class TorchHead(nn.Module):
    def __init__(self, dim: int, architecture: str):
        super().__init__()
        if architecture in {"C1_LINEAR", "C2_PENALISED"}:
            self.network = nn.Linear(dim, 1)
        elif architecture in {"C3_DEEP16", "B_LAST_BLOCK_COX16"}:
            self.network = nn.Sequential(nn.Linear(dim, 16), nn.GELU(), nn.Linear(16, 1))
        elif architecture == "C4_DEEP32":
            self.network = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 1))
        else:
            self.network = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Dropout(.1), nn.Linear(32, 8), nn.GELU(), nn.Linear(8, 1))

    def forward(self, x):
        return self.network(x).reshape(-1)


def cox_loss(score, duration, event, weight):
    order = torch.argsort(duration, descending=True)
    s, e, w = score[order], event[order], weight[order]
    contribution = (s - torch.logcumsumexp(s, dim=0)) * e * w
    return -contribution.sum() / (e * w).sum().clamp_min(1.0)


def fit_cox_latent(x, frame, architecture: str):
    seed_all()
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x).astype(np.float32)
    head = TorchHead(z.shape[1], architecture).to("cpu")
    wd = .05 if architecture == "C2_PENALISED" else .005
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=wd)
    tx = torch.from_numpy(z)
    duration = torch.from_numpy(frame["duration"].to_numpy(dtype=np.float32))
    event = torch.from_numpy(frame["event"].to_numpy(dtype=np.float32))
    weight = torch.from_numpy(frame["participant_metric_weight"].to_numpy(dtype=np.float32))
    best, state, stale = np.inf, None, 0
    for _ in range(180):
        optimizer.zero_grad()
        loss = cox_loss(head(tx), duration, event, weight)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best - 1e-6:
            best, stale = value, 0
            state = copy.deepcopy(head.state_dict())
        else:
            stale += 1
        if stale >= 18:
            break
    head.load_state_dict(state)
    head.eval()
    return {"scaler": scaler, "head": head, "architecture": architecture, "best_loss": best}


def predict_cox(fit, x):
    z = fit["scaler"].transform(x).astype(np.float32)
    with torch.no_grad():
        return fit["head"](torch.from_numpy(z)).numpy()


def cox_metrics(frame, score) -> dict:
    if frame["event"].sum() < 2:
        return {"Harrell_C": np.nan, "status": "STRUCTURAL_NE_FEW_EVENTS"}
    return {"Harrell_C": concordance_index(frame["duration"], -np.asarray(score), frame["event"]), "status": "ESTIMABLE"}


def fit_last_block_binary(bundle, base_alignment, training, validation, weights, train_mean):
    seed_all()
    model = copy.deepcopy(base_alignment).to(bundle.device)
    for p in model.parameters():
        p.requires_grad = False
    for module in (model.shared_linear, model.shared_norm, model.latent_projection):
        for p in module.parameters():
            p.requires_grad = True
    head = nn.Sequential(nn.Linear(model.latent_dim, 16), nn.GELU(), nn.Linear(16, 1)).to(bundle.device)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    x = bundle.bulk.loc[training["GSM"], bundle.genes].to_numpy(dtype=np.float32)
    x = train_mean + (x - train_mean) * weights
    tx = torch.from_numpy(x).to(bundle.device)
    y = torch.from_numpy(training["y_true"].to_numpy(dtype=np.float32)).to(bundle.device)
    sw = torch.from_numpy(training["participant_metric_weight"].to_numpy(dtype=np.float32)).to(bundle.device)
    best, state, stale = np.inf, None, 0
    for _ in range(120):
        model.train(); head.train(); optimizer.zero_grad(set_to_none=True)
        logits = head(model(tx, "bulk")).reshape(-1)
        loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * sw).sum() / sw.sum()
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); optimizer.step()
        value = float(loss.detach().cpu())
        if value < best - 1e-6:
            best, stale = value, 0
            state = {"alignment": copy.deepcopy(model.state_dict()), "head": copy.deepcopy(head.state_dict())}
        else:
            stale += 1
        if stale >= 12:
            break
    model.load_state_dict(state["alignment"]); head.load_state_dict(state["head"]); model.eval(); head.eval()
    xv = bundle.bulk.loc[validation["GSM"], bundle.genes].to_numpy(dtype=np.float32)
    xv = train_mean + (xv - train_mean) * weights
    with torch.no_grad():
        score = torch.sigmoid(head(model(torch.from_numpy(xv).to(bundle.device), "bulk")).reshape(-1)).cpu().numpy()
    return score


def fit_last_block_cox(bundle, base_alignment, training, validation, weights, train_mean):
    seed_all()
    model = copy.deepcopy(base_alignment).to(bundle.device)
    for p in model.parameters(): p.requires_grad = False
    for module in (model.shared_linear, model.shared_norm, model.latent_projection):
        for p in module.parameters(): p.requires_grad = True
    head = TorchHead(model.latent_dim, "B_LAST_BLOCK_COX16").to(bundle.device)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    x = bundle.bulk.loc[training["GSM"], bundle.genes].to_numpy(dtype=np.float32)
    x = train_mean + (x - train_mean) * weights
    tx = torch.from_numpy(x).to(bundle.device)
    duration = torch.from_numpy(training["duration"].to_numpy(dtype=np.float32)).to(bundle.device)
    event = torch.from_numpy(training["event"].to_numpy(dtype=np.float32)).to(bundle.device)
    sw = torch.from_numpy(training["participant_metric_weight"].to_numpy(dtype=np.float32)).to(bundle.device)
    best, state, stale = np.inf, None, 0
    for _ in range(160):
        model.train(); head.train(); optimizer.zero_grad(set_to_none=True)
        loss = cox_loss(head(model(tx, "bulk")), duration, event, sw)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); optimizer.step()
        value = float(loss.detach().cpu())
        if value < best - 1e-6:
            best, stale = value, 0
            state = {"alignment": copy.deepcopy(model.state_dict()), "head": copy.deepcopy(head.state_dict())}
        else: stale += 1
        if stale >= 15: break
    model.load_state_dict(state["alignment"]); head.load_state_dict(state["head"]); model.eval(); head.eval()
    xv = bundle.bulk.loc[validation["GSM"], bundle.genes].to_numpy(dtype=np.float32)
    xv = train_mean + (xv - train_mean) * weights
    with torch.no_grad(): score = head(model(torch.from_numpy(xv).to(bundle.device), "bulk")).cpu().numpy()
    return score


class MultiTaskHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, 16), nn.GELU(), nn.Dropout(.05))
        self.heads = nn.ModuleDict({endpoint: nn.Linear(16, 1) for endpoint in ("E_ALL", "E_6M", "E_12M", "E_COX")})

    def forward(self, x, endpoint: str):
        return self.heads[endpoint](self.trunk(x)).reshape(-1)


def fit_multitask_score(bundle, base_alignment, heldout, split, current_endpoint, train_ids, validation):
    development = set(FORMAL_COHORTS) - {heldout}
    endpoint_data = {}
    all_latent = []
    for endpoint in ("E_ALL", "E_6M", "E_12M", "E_COX"):
        frame = frame_for(bundle, endpoint, development)
        if split.startswith("LODO_"):
            validation_cohort = split.replace("LODO_", "", 1)
            training = frame[~frame["cohort"].eq(validation_cohort)].copy()
        else:
            training = frame[~frame["participant_id"].isin(set(validation["participant_id"]))].copy()
        if training.empty or training["y_true"].nunique() < 2:
            continue
        train_mean = bundle.bulk.loc[training["GSM"], bundle.genes].to_numpy(dtype=np.float32).mean(axis=0)
        latent = latent_features(bundle, base_alignment, training, np.ones(len(bundle.genes), dtype=np.float32), train_mean)
        endpoint_data[endpoint] = (training, latent)
        all_latent.append(latent)
    if current_endpoint not in endpoint_data or not all_latent:
        raise RuntimeError("MULTITASK_CURRENT_ENDPOINT_NOT_TRAINABLE")
    scaler = StandardScaler().fit(np.concatenate(all_latent, axis=0))
    model = MultiTaskHead(all_latent[0].shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    tensors = {}
    for endpoint, (frame, latent) in endpoint_data.items():
        tensors[endpoint] = (
            torch.from_numpy(scaler.transform(latent).astype(np.float32)),
            torch.from_numpy(frame["y_true"].to_numpy(dtype=np.float32)),
            torch.from_numpy(frame["participant_metric_weight"].to_numpy(dtype=np.float32)),
            torch.from_numpy(frame["duration"].to_numpy(dtype=np.float32)) if endpoint == "E_COX" else None,
        )
    best, state, stale = np.inf, None, 0
    for _ in range(220):
        model.train(); optimizer.zero_grad(); losses = []
        for endpoint, (x, y, weight, duration) in tensors.items():
            score = model(x, endpoint)
            if endpoint == "E_COX":
                losses.append(cox_loss(score, duration, y, weight))
            else:
                losses.append((F.binary_cross_entropy_with_logits(score, y, reduction="none") * weight).sum() / weight.sum())
        loss = torch.stack(losses).mean(); loss.backward(); optimizer.step()
        value = float(loss.detach())
        if value < best - 1e-6:
            best, state, stale = value, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 20: break
    model.load_state_dict(state); model.eval()
    current_training = endpoint_data[current_endpoint][0]
    train_mean = bundle.bulk.loc[current_training["GSM"], bundle.genes].to_numpy(dtype=np.float32).mean(axis=0)
    latent_val = latent_features(bundle, base_alignment, validation, np.ones(len(bundle.genes), dtype=np.float32), train_mean)
    with torch.no_grad(): raw = model(torch.from_numpy(scaler.transform(latent_val).astype(np.float32)), current_endpoint).numpy()
    return raw if current_endpoint == "E_COX" else expit(raw)


def run_inner() -> dict:
    bundle = load_bundle()
    manifest = build_fold_manifest(bundle)
    atomic_tsv(bundle.run / "04_INNER_CV" / "FOLD_MANIFEST.tsv", manifest)
    audit = manifest.groupby(["outer_heldout", "endpoint", "inner_split", "participant_id"])["role"].nunique().reset_index(name="n_roles")
    audit["participant_leakage"] = audit["n_roles"].gt(1)
    atomic_tsv(bundle.run / "04_INNER_CV" / "PARTICIPANT_LEAKAGE_AUDIT.tsv", audit)
    if audit["participant_leakage"].any(): raise RuntimeError("PARTICIPANT_LEAKAGE")
    result_rows, gene_rows = [], []
    context_dir = bundle.run / "04_INNER_CV" / "context_results"
    gene_dir = bundle.run / "03_FEATURE_SELECTION" / "context_gene_statistics"
    context_dir.mkdir(parents=True, exist_ok=True)
    gene_dir.mkdir(parents=True, exist_ok=True)
    passing = manifest[manifest["status"].eq("PASS")]
    contexts = passing[["outer_heldout", "endpoint", "inner_split"]].drop_duplicates()
    for _, context in contexts.iterrows():
        heldout, endpoint, split = context.tolist()
        context_name = f"{heldout}__{endpoint}__{split}".replace("/", "_")
        context_path = context_dir / f"{context_name}.tsv"
        gene_path = gene_dir / f"{context_name}.parquet"
        if context_path.exists() and gene_path.exists():
            result_rows.extend(pd.read_csv(context_path, sep="\t").to_dict("records"))
            gene_rows.append(pd.read_parquet(gene_path))
            continue
        development = set(FORMAL_COHORTS) - {heldout}
        full = frame_for(bundle, endpoint, development)
        part = passing[(passing["outer_heldout"].eq(heldout)) & (passing["endpoint"].eq(endpoint)) & (passing["inner_split"].eq(split))]
        train_ids = set(part.loc[part["role"].eq("training"), "participant_id"])
        val_ids = set(part.loc[part["role"].eq("validation"), "participant_id"])
        training = full[full["participant_id"].isin(train_ids)].copy()
        validation = full[full["participant_id"].isin(val_ids)].copy()
        if set(training["participant_id"]) & set(validation["participant_id"]): raise RuntimeError("INNER_PARTICIPANT_OVERLAP")
        stats = cox_gene_statistics(bundle, training) if endpoint == "E_COX" else binary_gene_statistics(bundle, training, SEED)
        stats.insert(0, "inner_split", split); stats.insert(0, "endpoint", endpoint); stats.insert(0, "outer_heldout", heldout)
        gene_rows.append(stats)
        stats.to_parquet(gene_path, index=False)
        base_alignment = load_alignment(bundle, heldout)
        train_mean = bundle.bulk.loc[training["GSM"], bundle.genes].to_numpy(dtype=np.float32).mean(axis=0)
        for condition in GENE_CONDITIONS:
            weights, n_genes = gene_weights(stats, condition, endpoint)
            if condition != "G0" and n_genes == 0:
                result_rows.append({"outer_heldout": heldout, "endpoint": endpoint, "inner_split": split, "gene_condition": condition, "model": "ALL", "effective_genes": 0, "status": "STRUCTURAL_NE_ZERO_GENES"})
                continue
            x_train = latent_features(bundle, base_alignment, training, weights, train_mean)
            x_val = latent_features(bundle, base_alignment, validation, weights, train_mean)
            if endpoint != "E_COX":
                model_names = [name for name in BINARY_MODELS if condition == "G0" or name not in {"B_LAST_BLOCK_MLP16", "C_MULTITASK_SHARED16"}]
                for model_name in model_names:
                    try:
                        if model_name == "B_LAST_BLOCK_MLP16":
                            score = fit_last_block_binary(bundle, base_alignment, training, validation, weights, train_mean)
                        elif model_name == "C_MULTITASK_SHARED16":
                            score = fit_multitask_score(bundle, base_alignment, heldout, split, endpoint, train_ids, validation)
                        else:
                            scaler = StandardScaler().fit(x_train)
                            estimator = binary_estimator(model_name)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                estimator.fit(scaler.transform(x_train), training["y_true"], sample_weight=training["participant_metric_weight"])
                            score = estimator.predict_proba(scaler.transform(x_val))[:, 1]
                        metrics = binary_metrics(validation["y_true"], score, validation["participant_metric_weight"])
                    except Exception as exc:
                        metrics = {"status": f"FAILED:{type(exc).__name__}:{str(exc)[:120]}"}
                    result_rows.append({"outer_heldout": heldout, "endpoint": endpoint, "inner_split": split, "gene_condition": condition, "model": model_name, "effective_genes": n_genes, "n_train_participants": training["participant_id"].nunique(), "n_validation_participants": validation["participant_id"].nunique(), **metrics})
            else:
                model_names = [name for name in COX_MODELS if condition == "G0" or name not in {"B_LAST_BLOCK_COX16", "C_MULTITASK_SHARED16"}]
                for model_name in model_names:
                    try:
                        if model_name == "B_LAST_BLOCK_COX16":
                            score = fit_last_block_cox(bundle, base_alignment, training, validation, weights, train_mean)
                        elif model_name == "C_MULTITASK_SHARED16":
                            score = fit_multitask_score(bundle, base_alignment, heldout, split, endpoint, train_ids, validation)
                        else:
                            fit = fit_cox_latent(x_train, training, model_name)
                            score = predict_cox(fit, x_val)
                        metrics = cox_metrics(validation, score)
                    except Exception as exc:
                        metrics = {"status": f"FAILED:{type(exc).__name__}:{str(exc)[:120]}"}
                    result_rows.append({"outer_heldout": heldout, "endpoint": endpoint, "inner_split": split, "gene_condition": condition, "model": model_name, "effective_genes": n_genes, "n_train_participants": training["participant_id"].nunique(), "n_validation_participants": validation["participant_id"].nunique(), **metrics})
        context_block = pd.DataFrame([row for row in result_rows if row.get("outer_heldout") == heldout and row.get("endpoint") == endpoint and row.get("inner_split") == split])
        atomic_tsv(context_path, context_block)
        del base_alignment
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    all_results = pd.DataFrame(result_rows)
    atomic_tsv(bundle.run / "04_INNER_CV" / "V126_ALL_CANDIDATES_INNER_CV.tsv", all_results)
    pd.concat(gene_rows, ignore_index=True).to_parquet(bundle.run / "03_FEATURE_SELECTION" / "INNER_TRAINING_GENE_STATISTICS.parquet", index=False)
    locks = []
    for (heldout, endpoint), block in all_results.groupby(["outer_heldout", "endpoint"]):
        metric = "Harrell_C" if endpoint == "E_COX" else "AUROC"
        valid = block[block["status"].eq("ESTIMABLE") & pd.to_numeric(block.get(metric), errors="coerce").notna()].copy()
        if valid.empty:
            locks.append({"outer_heldout": heldout, "endpoint": endpoint, "status": "STRUCTURAL_NE_NO_VALID_INNER_CANDIDATE"})
            continue
        agg = valid.groupby(["gene_condition", "model", "effective_genes"]).agg(
            median_primary=(metric, "median"), worst_primary=(metric, "min"), n_inner=(metric, "count"),
            median_AUPRC=("AUPRC", "median") if endpoint != "E_COX" else (metric, "median"),
            median_Brier=("Brier", "median") if endpoint != "E_COX" else (metric, "median"),
        ).reset_index()
        agg["complexity"] = agg["model"].map(MODEL_COMPLEXITY).fillna(99)
        max_median = agg["median_primary"].max()
        contenders = agg[agg["median_primary"].ge(max_median - .01)].sort_values(
            ["worst_primary", "median_AUPRC", "median_Brier", "complexity", "effective_genes"],
            ascending=[False, False, True, True, True],
        )
        winner = contenders.iloc[0].to_dict()
        payload = {"outer_heldout": heldout, "endpoint": endpoint, "gene_condition": winner["gene_condition"], "model": winner["model"], "effective_genes": int(winner["effective_genes"]), "median_inner_primary": float(winner["median_primary"]), "worst_inner_primary": float(winner["worst_primary"]), "selection_rule": "median; within_0.01_worst_then_AUPRC_then_Brier_then_simplicity_then_genes", "iteration": 0, "outer_labels_read": False, "status": "LOCKED"}
        lock_path = bundle.run / "05_MODELS" / f"LOCK_{heldout}_{endpoint}.json"
        atomic_json(lock_path, payload)
        payload["lock_sha256"] = sha256(lock_path)
        locks.append(payload)
    locks_frame = pd.DataFrame(locks)
    atomic_tsv(bundle.run / "05_MODELS" / "FINAL_CONFIGURATION_LOCKS.tsv", locks_frame)
    status = {"status": "INNER_SELECTION_COMPLETE", "locks": int(locks_frame["status"].eq("LOCKED").sum()), "participant_leakage": False, "outer_labels_read": False, "plots_created": False, "completed_utc": utc_now()}
    atomic_json(bundle.run / "04_INNER_CV" / "INNER_COMPLETE.json", status)
    assert_no_figures(bundle.run)
    print(json.dumps(status, indent=2))
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["inner"], required=True)
    args = parser.parse_args()
    if args.stage == "inner": run_inner()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
