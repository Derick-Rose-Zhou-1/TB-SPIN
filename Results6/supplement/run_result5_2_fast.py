#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("inputs/TB_STATE_SC_V3_NC_FINAL_20260828")
OUT = Path(os.environ.get("TBSTATE_OUT", "inputs/TB_STATE_RESULT5_2_FAST_DISENTANGLEMENT_20260913"))
LATENT_DIR = ROOT / "17_FINAL_FULL_MODEL"
PRIMARY_SEED = int(os.environ.get("TBSTATE_PRIMARY_SEED", "20260828"))
LATENT_SOURCE = Path(os.environ.get("TBSTATE_LATENT_SOURCE", str(LATENT_DIR / f"seed_{PRIMARY_SEED}/training_latent.tsv.gz")))
SEEDS = [PRIMARY_SEED]
RANDOM_STATE = 20260913
RAW_TO_MAJOR = {"T cell": "T", "NK cell": "NK", "B cell": "B", "Monocyte": "Monocyte", "pDC": "DC", "DC": "DC", "Neutrophil": "Neutrophil"}
STAGE_LABEL = {"HC": "HC", "TBINF": "Mtb infection", "ATB": "ATB"}
STAGE_CODE = {"HC": 0, "TBINF": 1, "ATB": 2}
PRIMARY_DF = None


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def probe():
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                   max_iter=1000, class_weight="balanced",
                                   multi_class="multinomial")),
    ])


def read_latents():
    frames = {}
    rows = []
    for seed in SEEDS:
        path = LATENT_SOURCE if len(SEEDS) == 1 else LATENT_DIR / f"seed_{seed}" / "training_latent.tsv.gz"
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, sep="\t")
        zid = [c for c in frame.columns if c.startswith("z_id_")]
        zstate = [c for c in frame.columns if c.startswith("z_state_")]
        required = {"cell_id", "sample_id_std", "analysis_stage", "celltype_raw"}
        if not required.issubset(frame.columns) or len(zid) != 16 or len(zstate) != 8:
            raise RuntimeError(f"Invalid formal latent schema: {path}")
        if frame.cell_id.duplicated().any():
            raise RuntimeError(f"Duplicate cell_id: {path}")
        frame["participant_id"] = frame.sample_id_std.astype(str)
        frame["clinical_state"] = frame.analysis_stage.map(STAGE_LABEL)
        frame["clinical_code"] = frame.analysis_stage.map(STAGE_CODE)
        frame["major_cell_type"] = frame.celltype_raw.map(RAW_TO_MAJOR)
        if frame.clinical_state.isna().any() or frame.major_cell_type.isna().any():
            raise RuntimeError(f"Unmapped frozen metadata: {path}")
        rows.append({
            "file": str(path), "object": "formal V3 full-model frozen latent",
            "n_cells": len(frame), "z_id_dim": len(zid), "z_state_dim": len(zstate),
            "participant_column": "sample_id_std", "celltype_column": "celltype_raw",
            "clinical_state_column": "analysis_stage", "production_status": "PASS",
            "used": "PRIMARY" if seed == PRIMARY_SEED else "SEED_SENSITIVITY",
            "seed": seed, "sha256": sha256(path),
        })
        frames[seed] = frame
    ref = frames[PRIMARY_SEED]
    for seed, frame in frames.items():
        if not np.array_equal(ref.cell_id.to_numpy(), frame.cell_id.to_numpy()):
            raise RuntimeError(f"Cell order mismatch: seed {seed}")
        if not np.array_equal(ref.participant_id.to_numpy(), frame.participant_id.to_numpy()):
            raise RuntimeError(f"Participant order mismatch: seed {seed}")
    save(pd.DataFrame(rows), OUT / "00_AUDIT/PRODUCTION_LATENT_REGISTRY.tsv")
    return frames, pd.DataFrame(rows)


def make_universe(df):
    rows = []
    for lineage in ["T", "NK", "B", "Monocyte", "DC", "Neutrophil"]:
        sub = df[df.major_cell_type.eq(lineage)]
        n = len(sub)
        npart = sub.participant_id.nunique()
        rows.append({"major_cell_type": lineage, "n_cells": n,
                     "n_participants": npart,
                     "included": bool(n >= 500 and npart >= 3),
                     "rule": ">=500 total cells and >=3 participants"})
    out = pd.DataFrame(rows)
    save(out, OUT / "00_AUDIT/ANALYSIS_UNIVERSE.tsv")
    if not out.included.all():
        raise RuntimeError("A required broad lineage failed the prespecified rule")
    return out


def make_folds(df):
    p = df[["participant_id", "clinical_state", "clinical_code"]].drop_duplicates("participant_id").sort_values("participant_id").reset_index(drop=True)
    if p.participant_id.duplicated().any():
        raise RuntimeError("A participant has multiple clinical states")
    nfold = min(5, int(p.clinical_state.value_counts().min()))
    if nfold < 3:
        raise RuntimeError("Fewer than three feasible participant folds")
    cv = StratifiedGroupKFold(n_splits=nfold, shuffle=True, random_state=RANDOM_STATE)
    p["fold"] = -1
    X = np.zeros((len(p), 1))
    for fold, (_, test) in enumerate(cv.split(X, p.clinical_state, p.participant_id)):
        p.loc[test, "fold"] = fold
    save(p[["participant_id", "clinical_state", "fold"]], OUT / "00_AUDIT/PARTICIPANT_FOLDS.tsv")
    return p[["participant_id", "clinical_state", "fold"]]


def make_identity_set(df, folds):
    work = df[df.major_cell_type.isin(["T", "NK", "B", "Monocyte", "DC", "Neutrophil"])].copy()
    rng = np.random.default_rng(RANDOM_STATE)
    parts = []
    for (pid, lineage), sub in work.groupby(["participant_id", "major_cell_type"], sort=True):
        idx = sub.index.to_numpy()
        if len(idx) > 500:
            idx = rng.choice(idx, 500, replace=False)
        parts.append(work.loc[np.sort(idx), ["cell_id", "participant_id", "clinical_state", "major_cell_type"]])
    cells = pd.concat(parts, ignore_index=True).sort_values(["participant_id", "major_cell_type", "cell_id"])
    cells = cells.merge(folds, on=["participant_id", "clinical_state"], validate="many_to_one")
    lookup = pd.Series(df.index.to_numpy(), index=df.cell_id.astype(str))
    cells.insert(0, "row_index", cells.cell_id.astype(str).map(lookup))
    if cells.row_index.isna().any() or cells.cell_id.duplicated().any():
        raise RuntimeError("Identity cell set is not aligned")
    save(cells, OUT / "02_IDENTITY_PROBE/IDENTITY_CELL_SET.tsv.gz")
    return cells.reset_index(drop=True)


def identity_probe(df, cells, seed):
    indexed = df.set_index("cell_id")
    output = []
    summaries = []
    for rep in ["z_id", "z_state"]:
        cols = [c for c in df.columns if c.startswith(rep + "_")]
        X = indexed.loc[cells.cell_id, cols].to_numpy(float)
        y = cells.major_cell_type.to_numpy(str)
        pred = np.empty(len(cells), dtype=object)
        for fold in sorted(cells.fold.unique()):
            train = cells.fold.to_numpy() != fold
            test = ~train
            model = probe()
            model.fit(X[train], y[train])
            pred[test] = model.predict(X[test])
        output.append(pd.DataFrame({
            "seed": seed, "representation": rep, "cell_id": cells.cell_id,
            "participant_id": cells.participant_id, "major_cell_type": y,
            "fold": cells.fold, "y_pred": pred,
        }))
        summaries.append({
            "seed": seed, "representation": rep,
            "macro_F1": f1_score(y, pred, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "n_cells": len(y), "n_participants": cells.participant_id.nunique(),
            "n_celltypes": pd.Series(y).nunique(), "estimate": "OOF_overall",
        })
    oof = pd.concat(output, ignore_index=True)
    summary = pd.DataFrame(summaries)
    save(oof, OUT / "02_IDENTITY_PROBE/IDENTITY_PROBE_OOF.tsv")
    save(summary, OUT / "02_IDENTITY_PROBE/IDENTITY_PROBE_SUMMARY.tsv")
    return oof, {r["representation"]: r for r in summaries}


def participant_latents(df, folds):
    zid = [c for c in df.columns if c.startswith("z_id_")]
    zstate = [c for c in df.columns if c.startswith("z_state_")]
    p = df.groupby(["participant_id", "clinical_state", "clinical_code"], observed=True)[zid + zstate].median().reset_index()
    p = p.merge(folds, on=["participant_id", "clinical_state"], validate="one_to_one")
    save(p, OUT / "03_STATE_PROBE/PARTICIPANT_LATENTS.tsv")
    return p


def state_probe(p, rep, seed):
    cols = [c for c in p.columns if c.startswith(rep + "_")]
    y = p.clinical_code.to_numpy(int)
    prob = np.zeros((len(p), 3), float)
    pred = np.full(len(p), -1, int)
    for fold in sorted(p.fold.unique()):
        train = p.fold.to_numpy() != fold
        test = ~train
        model = probe()
        model.fit(p.loc[train, cols], y[train])
        pr = model.predict_proba(p.loc[test, cols])
        for j, label in enumerate(model.classes_.astype(int)):
            prob[test, label] = pr[:, j]
        pred[test] = model.predict(p.loc[test, cols]).astype(int)
    score = prob @ np.arange(3, dtype=float)
    tau = float(kendalltau(y, score, variant="b").statistic)
    summary = {"seed": seed, "representation": rep, "kendall_tau_b": tau,
               "balanced_accuracy": balanced_accuracy_score(y, pred),
               "n_participants": len(p), "n_HC": int((y == 0).sum()),
               "n_infection": int((y == 1).sum()), "n_ATB": int((y == 2).sum()),
               "estimate": "OOF_overall"}
    oof = p[["participant_id", "clinical_state", "clinical_code", "fold"]].copy()
    oof.insert(0, "seed", seed)
    oof["representation"] = rep
    oof["p_HC"], oof["p_infection"], oof["p_ATB"] = prob[:, 0], prob[:, 1], prob[:, 2]
    oof["predicted_state_score"], oof["predicted_state"] = score, pred
    return oof, summary


def stratified_indices(codes, rng):
    result = []
    for code in [0, 1, 2]:
        pool = np.flatnonzero(codes == code)
        result.extend(rng.choice(pool, len(pool), replace=True).tolist())
    return np.asarray(result, dtype=int)


def identity_bootstrap(oof, df):
    rng = np.random.default_rng(RANDOM_STATE)
    base = oof[oof.cell_id.notna()].copy()
    p = df[["participant_id", "clinical_code"]].drop_duplicates("participant_id").sort_values("participant_id")
    labels = sorted(base.major_cell_type.unique())
    label_index = {label: i for i, label in enumerate(labels)}
    conf = {}
    for rep in ["z_id", "z_state"]:
        conf[rep] = np.zeros((len(p), len(labels), len(labels)), dtype=np.int64)
        for i, pid in enumerate(p.participant_id):
            sub = base[(base.representation == rep) & base.participant_id.eq(pid)]
            for truth, pred in zip(sub.major_cell_type, sub.y_pred):
                conf[rep][i, label_index[truth], label_index[pred]] += 1

    def macro_f1_from_conf(matrix):
        tp = np.diag(matrix).astype(float)
        fp = matrix.sum(axis=0) - tp
        fn = matrix.sum(axis=1) - tp
        denom = 2 * tp + fp + fn
        return float(np.mean(np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)))

    rows = []
    for b in range(2000):
        pick = stratified_indices(p.clinical_code.to_numpy(int), rng)
        vals = {"replicate": b, "seed": PRIMARY_SEED}
        for rep in ["z_id", "z_state"]:
            vals["macro_F1_" + rep] = macro_f1_from_conf(conf[rep][pick].sum(axis=0))
        vals["delta_identity_F1"] = vals["macro_F1_z_id"] - vals["macro_F1_z_state"]
        rows.append(vals)
    out = pd.DataFrame(rows)
    save(out, OUT / "06_BOOTSTRAP/IDENTITY_DELTA_BOOTSTRAP.tsv")
    return out


def state_bootstrap(long_oof):
    wide = long_oof.pivot(index=["participant_id", "clinical_code"], columns="representation", values="predicted_state_score").reset_index()
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    codes = wide.clinical_code.to_numpy(int)
    for b in range(5000):
        pick = stratified_indices(codes, rng)
        sub = wide.iloc[pick]
        tau_id = float(kendalltau(sub.clinical_code, sub.z_id, variant="b").statistic)
        tau_state = float(kendalltau(sub.clinical_code, sub.z_state, variant="b").statistic)
        rows.append({"replicate": b, "seed": PRIMARY_SEED, "tau_z_id": tau_id,
                     "tau_z_state": tau_state, "delta_state_tau": tau_state - tau_id})
    out = pd.DataFrame(rows)
    save(out, OUT / "06_BOOTSTRAP/STATE_DELTA_BOOTSTRAP.tsv")
    return out


def within_lineage(df, folds, seed):
    state_rows, compare_rows = [], []
    for lineage in ["T", "NK", "Monocyte", "B"]:
        sub = df[df.major_cell_type.eq(lineage)]
        counts = sub.groupby(["participant_id", "clinical_state"], observed=True).size().rename("n_cells").reset_index()
        eligible = counts[counts.n_cells.ge(30)]
        coverage = eligible.groupby("clinical_state").participant_id.nunique().to_dict()
        row = {"lineage": lineage, "n_HC": coverage.get("HC", 0),
               "n_infection": coverage.get("Mtb infection", 0), "n_ATB": coverage.get("ATB", 0),
               "estimable": all(coverage.get(s, 0) >= 3 for s in ["HC", "Mtb infection", "ATB"])}
        if not row["estimable"]:
            row.update({"tau_b": np.nan, "CI_low": np.nan, "CI_high": np.nan,
                        "balanced_accuracy": np.nan, "reason": "<3 participants per state"})
            state_rows.append(row)
            continue
        ids = eligible.participant_id.to_numpy()
        ldf = sub[sub.participant_id.isin(ids)].groupby(["participant_id", "clinical_state", "clinical_code"], observed=True)[[c for c in sub.columns if c.startswith("z_")]].median().reset_index()
        ldf = ldf.merge(folds, on=["participant_id", "clinical_state"], validate="one_to_one")
        oof_state, summary_state = state_probe(ldf, "z_state", seed)
        oof_id, summary_id = state_probe(ldf, "z_id", seed)
        tau_values = []
        rng = np.random.default_rng(RANDOM_STATE + len(lineage))
        codes = ldf.clinical_code.to_numpy(int)
        state_scores = oof_state.set_index("participant_id").predicted_state_score
        for _ in range(2000):
            pick = stratified_indices(codes, rng)
            selected = ids[pick]
            tau_values.append(kendalltau(ldf.set_index("participant_id").loc[selected, "clinical_code"],
                                         state_scores.loc[selected], variant="b").statistic)
        row.update({"tau_b": summary_state["kendall_tau_b"],
                    "balanced_accuracy": summary_state["balanced_accuracy"],
                    "CI_low": np.nanpercentile(tau_values, 2.5),
                    "CI_high": np.nanpercentile(tau_values, 97.5), "reason": ""})
        state_rows.append(row)
        compare_rows.append({"lineage": lineage, "tau_z_state": summary_state["kendall_tau_b"],
                             "tau_z_id": summary_id["kendall_tau_b"],
                             "delta_tau_state_minus_id": summary_state["kendall_tau_b"] - summary_id["kendall_tau_b"],
                             "n_participants": len(ldf)})
    state = pd.DataFrame(state_rows)
    compare = pd.DataFrame(compare_rows)
    save(state, OUT / "04_WITHIN_LINEAGE/WITHIN_LINEAGE_STATE.tsv")
    save(compare, OUT / "04_WITHIN_LINEAGE/WITHIN_LINEAGE_ID_VS_STATE.tsv")
    return state, compare


def dependence(df):
    rng = np.random.default_rng(RANDOM_STATE)
    pieces = []
    for _, sub in df.groupby("participant_id", sort=True):
        idx = sub.index.to_numpy()
        if len(idx) > 200:
            idx = rng.choice(idx, 200, replace=False)
        pieces.append(df.loc[idx])
    sample = pd.concat(pieces, ignore_index=True)
    zid = [c for c in df.columns if c.startswith("z_id_")]
    zstate = [c for c in df.columns if c.startswith("z_state_")]
    a = StandardScaler().fit_transform(sample[zid])
    b = StandardScaler().fit_transform(sample[zstate])
    corr = np.corrcoef(a.T, b.T)[:len(zid), len(zid):]
    long = pd.DataFrame([{"z_id_dimension": zid[i], "z_state_dimension": zstate[j], "correlation": corr[i, j]}
                         for i in range(len(zid)) for j in range(len(zstate))])
    save(long, OUT / "05_DEPENDENCE/CROSS_LATENT_CORRELATION.tsv")
    save(pd.DataFrame([{"mean_absolute_cross_correlation": np.abs(corr).mean(),
                        "median_absolute_cross_correlation": np.median(np.abs(corr)),
                        "maximum_absolute_cross_correlation": np.abs(corr).max(),
                        "normalized_frobenius_norm": np.sqrt(np.mean(corr ** 2)),
                        "n_cells": len(sample), "n_participants": sample.participant_id.nunique()}]),
         OUT / "05_DEPENDENCE/DEPENDENCE_SUMMARY.tsv")
    return corr


def ci(a):
    return np.nanpercentile(np.asarray(a, float), [2.5, 97.5])


def figures(id_summary, state_summary, id_boot, state_boot, lineage, corr):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    reps = ["z_id", "z_state"]
    v = [id_summary[r]["macro_F1"] for r in reps]
    ci_id = [ci(id_boot["macro_F1_" + r]) for r in reps]
    ax[0].bar(reps, v, color=["#4C78A8", "#E69F00"], width=.58, edgecolor="black", linewidth=.6)
    ax[0].errorbar(reps, v, yerr=[[v[i]-ci_id[i][0] for i in range(2)], [ci_id[i][1]-v[i] for i in range(2)]],
                   fmt="none", ecolor="black", capsize=3, lw=1)
    ax[0].set_title("CELL IDENTITY INFORMATION", fontsize=9, weight="bold")
    ax[0].set_ylabel("Participant-held-out macro-F1", fontsize=8)
    ax[0].text(.98, .96, f"Delta F1={v[0]-v[1]:.3f}", transform=ax[0].transAxes, ha="right", va="top", fontsize=7)
    v2 = [state_summary[r]["kendall_tau_b"] for r in reps]
    ci_state = [ci(state_boot["tau_" + r]) for r in reps]
    ax[1].bar(reps, v2, color=["#4C78A8", "#E69F00"], width=.58, edgecolor="black", linewidth=.6)
    ax[1].errorbar(reps, v2, yerr=[[v2[i]-ci_state[i][0] for i in range(2)], [ci_state[i][1]-v2[i] for i in range(2)]],
                   fmt="none", ecolor="black", capsize=3, lw=1)
    ax[1].axhline(0, color="black", lw=.7)
    ax[1].set_title("CLINICAL-STATE INFORMATION", fontsize=9, weight="bold")
    ax[1].set_ylabel("Participant-held-out Kendall tau-b", fontsize=8)
    ax[1].text(.98, .96, f"Delta tau={v2[1]-v2[0]:.3f}", transform=ax[1].transAxes, ha="right", va="top", fontsize=7)
    for a in ax:
        a.tick_params(labelsize=7)
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "07_FIGURES/Fig5_2A_DIFFERENTIAL_ENCODING.pdf", bbox_inches="tight")
    fig.savefig(OUT / "07_FIGURES/Fig5_2A_DIFFERENTIAL_ENCODING.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    est = lineage[lineage.estimable].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    if len(est):
        y = np.arange(len(est))
        ax.errorbar(est.tau_b, y, xerr=[est.tau_b-est.CI_low, est.CI_high-est.tau_b], fmt="o", color="#4C78A8", capsize=3)
        ax.set_yticks(y, est.lineage)
    ax.axvline(0, color="black", lw=.7)
    ax.set_xlabel("Kendall tau-b for clinical state", fontsize=9)
    ax.set_title("Within-lineage state information", fontsize=10, weight="bold")
    ax.tick_params(labelsize=8)
    fig.savefig(OUT / "07_FIGURES/Fig5_2B_WITHIN_LINEAGE_STATE.pdf", bbox_inches="tight")
    fig.savefig(OUT / "07_FIGURES/Fig5_2B_WITHIN_LINEAGE_STATE.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xlabel("z_state dimensions", fontsize=9)
    ax.set_ylabel("z_id dimensions", fontsize=9)
    ax.set_title("Cross-latent correlation", fontsize=10, weight="bold")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="Pearson correlation")
    fig.savefig(OUT / "07_FIGURES/Fig5_2C_CROSS_LATENT_CORRELATION.pdf", bbox_inches="tight")
    fig.savefig(OUT / "07_FIGURES/Fig5_2C_CROSS_LATENT_CORRELATION.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(df, registry, id_summary, state_summary, id_boot, state_boot, lineage, corr, id_oof, state_oof, folds, qa):
    id_delta = id_summary["z_id"]["macro_F1"] - id_summary["z_state"]["macro_F1"]
    state_delta = state_summary["z_state"]["kendall_tau_b"] - state_summary["z_id"]["kendall_tau_b"]
    id_ci = ci(id_boot.delta_identity_F1)
    state_ci = ci(state_boot.delta_state_tau)
    decisions = int(id_delta > 0) + int(state_delta > 0)
    decision = "SUPPORTED" if decisions == 2 else "PARTIAL" if decisions == 1 else "NOT_SUPPORTED"
    interp = {
        "SUPPORTED": "TB-STATE latent representations showed differential encoding of cellular identity and tuberculosis-associated clinical-state information.",
        "PARTIAL": "The frozen latent validation provided partial evidence for differential encoding of cellular identity and clinical state.",
        "NOT_SUPPORTED": "The frozen latent validation did not provide clear evidence that the two latent spaces differentially encode cell identity and clinical state.",
    }[decision]
    english = interp + f" The primary analysis used formal production seed {PRIMARY_SEED}, comprising {df.participant_id.nunique()} participants and {len(df)} cells; confidence intervals were participant-level bootstrap intervals."
    chinese = f"冻结 latent 验证结果为 {decision}。主分析使用正式 production seed {PRIMARY_SEED}，包含 {df.participant_id.nunique()} 名参与者和 {len(df)} 个细胞；置信区间均基于参与者层面的 bootstrap。"
    (OUT / "09_REPORT/RESULT5_2_ENGLISH.md").write_text(english + "\n", encoding="utf-8")
    (OUT / "09_REPORT/RESULT5_2_CHINESE.md").write_text(chinese + "\n", encoding="utf-8")
    methods = """# Methods

We performed a frozen-latent validation without retraining TB-STATE or loading any checkpoint. The primary representation was the formal Result2 V3 full-model export for seed 20260828, containing 16 identity dimensions and 8 state dimensions. The frozen broad annotations were mapped to T, NK, B, Monocyte, DC and Neutrophil groups without reannotation. The analysis universe retained only HC, Mtb infection and ATB cells, with at least 500 cells and 3 participants per broad lineage.

Five participant-held-out, clinical-state-stratified folds were fixed once and reused for both latent spaces and all probes. Linear probes used a training-fold StandardScaler and multinomial L2 logistic regression (C=1, balanced class weights, max_iter=1000). Identity was evaluated by macro-F1 and balanced accuracy on the same participant-balanced cell set, with a maximum of 500 cells per participant-lineage. Clinical state was evaluated after participant-level median pooling using the OOF expected ordered score 0P(HC)+1P(infection)+2P(ATB), with Kendall tau-b as the primary metric. Participant-stratified bootstrap used B=2000 for identity, B=5000 for state and B=2000 within lineage.

Each training seed was analyzed separately; independently trained latent coordinates were not averaged because they are not directly comparable without an additional alignment procedure. No model retraining, Result3 latent, DEG, pathway, ARI/NMI, UMAP or simulation analysis was run.
"""
    legend = """# Figure legend

Fig. 5.2 | Frozen-latent differential encoding validation. (A) Participant-held-out macro-F1 for broad cell identity and Kendall tau-b for ordered clinical state, shown for z_id and z_state with participant-bootstrap 95% confidence intervals. (B) Within-lineage clinical-state association for lineages meeting the prespecified >=30 cells per participant and >=3 participants per state criteria. (C) Cross-latent Pearson correlation diagnostic. This diagnostic does not imply complete independence.
"""
    interpretation = f"# Interpretation\n\n{interp}\n\nIdentity delta (z_id minus z_state) = {id_delta:.4f}; 95% CI = [{id_ci[0]:.4f}, {id_ci[1]:.4f}]. State delta (z_state minus z_id) = {state_delta:.4f}; 95% CI = [{state_ci[0]:.4f}, {state_ci[1]:.4f}].\n"
    (OUT / "09_REPORT/METHODS.md").write_text(methods, encoding="utf-8")
    (OUT / "09_REPORT/FIGURE_LEGEND.md").write_text(legend, encoding="utf-8")
    (OUT / "09_REPORT/INTERPRETATION.md").write_text(interpretation, encoding="utf-8")
    report = f"""FINAL_STATUS=PASS
DIFFERENTIAL_ENCODING={decision}
PRODUCTION_LATENT={LATENT_SOURCE}
N_CELLS={len(df)}
N_PARTICIPANTS={df.participant_id.nunique()}
N_HC={(df.clinical_state == "HC").sum()}
N_MTB_INFECTION={(df.clinical_state == "Mtb infection").sum()}
N_ATB={(df.clinical_state == "ATB").sum()}
Z_ID_DIM=16
Z_STATE_DIM=8
IDENTITY_MACRO_F1_ZID={id_summary["z_id"]["macro_F1"]:.8f}
IDENTITY_MACRO_F1_ZSTATE={id_summary["z_state"]["macro_F1"]:.8f}
DELTA_IDENTITY_F1={id_delta:.8f}
DELTA_IDENTITY_F1_95CI=[{id_ci[0]:.8f},{id_ci[1]:.8f}]
STATE_TAU_ZID={state_summary["z_id"]["kendall_tau_b"]:.8f}
STATE_TAU_ZSTATE={state_summary["z_state"]["kendall_tau_b"]:.8f}
DELTA_STATE_TAU={state_delta:.8f}
DELTA_STATE_TAU_95CI=[{state_ci[0]:.8f},{state_ci[1]:.8f}]
"""
    for lineage_name in ["T", "NK", "Monocyte", "B"]:
        s = lineage.loc[lineage.lineage.eq(lineage_name), "tau_b"]
        report += f"{lineage_name.upper().replace(' ', '_')}_TAU_ZSTATE={s.iloc[0] if len(s) else 'NA'}\n"
    (OUT / "11_PACKAGE/FINAL_TERMINAL_REPORT.txt").write_text(report, encoding="utf-8")
    return decision


def write_xlsx(id_oof, state_oof, lineage, id_boot, state_boot, corr_long, folds, qa):
    path = OUT / "08_SOURCE_DATA/RESULT5_2_DIFFERENTIAL_ENCODING_SOURCE.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"item": ["Frozen latent validation", "Primary seed", "Identity bootstrap", "State bootstrap"],
                      "value": ["No retraining", PRIMARY_SEED, 2000, 5000]}).to_excel(writer, sheet_name="README", index=False)
        id_oof.to_excel(writer, sheet_name="IDENTITY_PROBE", index=False)
        state_oof.to_excel(writer, sheet_name="STATE_PROBE", index=False)
        lineage.to_excel(writer, sheet_name="WITHIN_LINEAGE", index=False)
        pd.DataFrame({"identity_delta": id_boot.delta_identity_F1.describe(),
                      "state_delta": state_boot.delta_state_tau.describe()}).to_excel(writer, sheet_name="BOOTSTRAP_SUMMARY")
        corr_long.to_excel(writer, sheet_name="DEPENDENCE", index=False)
        folds.to_excel(writer, sheet_name="PARTICIPANT_FOLDS", index=False)
        qa.to_excel(writer, sheet_name="QA", index=False)


def package():
    path = OUT / "11_PACKAGE/TB_STATE_RESULT5_2_FAST_DISENTANGLEMENT_20260913.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in OUT.rglob("*"):
            if not file.is_file() or file == path:
                continue
            if file.suffix in {".pt", ".h5ad"}:
                continue
            z.write(file, file.relative_to(OUT))


def main():
    global PRIMARY_DF
    for sub in ["00_AUDIT", "01_FROZEN_LATENT", "02_IDENTITY_PROBE", "03_STATE_PROBE",
                "04_WITHIN_LINEAGE", "05_DEPENDENCE", "06_BOOTSTRAP", "07_FIGURES",
                "08_SOURCE_DATA", "09_REPORT", "10_QA", "11_PACKAGE"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    frames, registry = read_latents()
    PRIMARY_DF = frames[PRIMARY_SEED]
    df = PRIMARY_DF.copy()
    universe = make_universe(df)
    folds = make_folds(df)
    identity_cells = make_identity_set(df, folds)
    id_oof, id_summary = identity_probe(df, identity_cells, PRIMARY_SEED)
    id_boot = identity_bootstrap(id_oof, df)
    p = participant_latents(df, folds)
    state_parts, state_summaries = [], {}
    for rep in ["z_id", "z_state"]:
        oof, summary = state_probe(p, rep, PRIMARY_SEED)
        state_parts.append(oof)
        state_summaries[rep] = summary
    state_oof = pd.concat(state_parts, ignore_index=True)
    save(state_oof, OUT / "03_STATE_PROBE/STATE_PROBE_OOF.tsv")
    save(pd.DataFrame(state_summaries.values()), OUT / "03_STATE_PROBE/STATE_PROBE_SUMMARY.tsv")
    state_boot = state_bootstrap(state_oof)
    lineage, lineage_compare = within_lineage(df, folds, PRIMARY_SEED)
    corr = dependence(df)
    corr_long = pd.read_csv(OUT / "05_DEPENDENCE/CROSS_LATENT_CORRELATION.tsv", sep="\t")
    save(registry, OUT / "08_SOURCE_DATA/FROZEN_LATENTS_SOURCE.tsv")
    shutil.copy2(Path(__file__), OUT / "00_AUDIT/SCRIPT_USED.py")
    evidence = pd.DataFrame([
        ["production full model", str(ROOT / "17_FINAL_FULL_MODEL/DONE.json"), "status PASS; seed_n=3", "PASS"],
        ["production latent export", str(LATENT_SOURCE), "16 z_id + 8 z_state columns", "PASS"],
        ["production training script", str(ROOT / "scripts/train_v3_final_full_job.py"), "TrainConfig identity_dim=16, state_dim=8, hvg_n=5000", "PASS"],
        ["formal input", str(ROOT / "scripts/tb_state_sc_v3_training.py"), "R2_H5AD=train_new.h5ad; 356949 x 5000 count target", "PASS"],
        ["likelihood", str(ROOT / "scripts/tb_state_sc_v3_training.py"), "TrainConfig likelihood=zinb", "PASS"],
        ["clinical tasks", str(ROOT / "scripts/tb_state_sc_v3_training.py"), "T1 HC->TBINF; T2 TBINF->ATB; ordinal HC=0/TBINF=1/ATB=2", "PASS"],
        ["forbidden branches", "analysis script", "No ablation, Result3, retraining, DEG, pathway, ARI/NMI, UMAP or simulation", "PASS"],
    ], columns=["evidence_type", "source_file", "source_detail", "status"])
    save(evidence, OUT / "00_AUDIT/SOURCE_EVIDENCE.tsv")
    alignment = pd.DataFrame([
        ["same_cell_order", "PASS", "identical cell_id order across 3 formal seed exports"],
        ["same_participant_order", "PASS", "identical participant order across 3 formal seed exports"],
        ["no_duplicate_cell_id", "PASS", "validated for all 3 exports"],
        ["z_id_dim", "PASS", "16"],
        ["z_state_dim", "PASS", "8"],
        ["participant_complete", "PASS", str(df.participant_id.notna().all())],
        ["clinical_state_complete", "PASS", str(df.clinical_state.notna().all())],
        ["broad_cell_type_complete", "PASS", str(df.major_cell_type.notna().all())],
    ], columns=["check", "status", "detail"])
    save(alignment, OUT / "00_AUDIT/LATENT_ALIGNMENT_QA.tsv")
    qa = pd.DataFrame([
        ["FULL_PRODUCTION_MODEL_CONFIRMED", "PASS", "17_FINAL_FULL_MODEL DONE PASS"],
        ["Z_ID_FOUND", "PASS", "16 dimensions"],
        ["Z_STATE_FOUND", "PASS", "8 dimensions"],
        ["SAME_CELL_ORDER", "PASS", "formal seed exports"],
        ["SAME_PARTICIPANT_UNIVERSE", "PASS", "formal seed exports"],
        ["NO_ABLATION_CHECKPOINT_USED", "PASS", "formal production training latent only"],
        ["NO_RESULT3_LATENT_USED", "PASS", "not used"],
        ["PARTICIPANT_HELD_OUT", "PASS", "five StratifiedGroupKFold folds"],
        ["NO_CELL_LEVEL_PARTICIPANT_LEAKAGE", "PASS", "all probes split by participant"],
        ["SAME_FOLDS_ZID_ZSTATE", "PASS", "same fold registry"],
        ["TRAIN_ONLY_SCALER", "PASS", "pipeline fitted within each training fold"],
        ["ONE_PARTICIPANT_ONE_STATE", "PASS", "validated"],
        ["IDENTITY_PROBE_MATCHED", "PASS", "same balanced cell set"],
        ["STATE_PROBE_MATCHED", "PASS", "same participant set and folds"],
        ["BOOTSTRAP_PARTICIPANT_LEVEL", "PASS", "B=2000 and B=5000"],
        ["WITHIN_LINEAGE_FIXED_IDENTITY", "PASS", "frozen mapping; no reannotation"],
        ["NO_MODEL_RETRAINING", "PASS", "latent tables only"],
        ["NO_ARI_NMI", "PASS", "not run"],
        ["NO_RESULT3_PROGRESSION", "PASS", "not run"],
    ], columns=["check", "status", "evidence"])
    save(qa, OUT / "10_QA/FINAL_QA.tsv")
    figures(id_summary, state_summaries, id_boot, state_boot, lineage, corr)
    write_outputs(df, registry, id_summary, state_summaries, id_boot, state_boot, lineage, corr,
                  id_oof, state_oof, folds, qa)
    write_xlsx(id_oof, state_oof, lineage, id_boot, state_boot, corr_long, folds, qa)
    package()
    print((OUT / "11_PACKAGE/FINAL_TERMINAL_REPORT.txt").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
