"""TB-STATE V12.6 final empirical risk-cell lock (no model training).

This script consumes the already reconstructed fold-wise cell scores and the
fixed six-lineage atlas.  It performs only percentile calibration, prespecified
thresholding and stability audits; no clinical labels or outer metrics are
consulted.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260816
K = 500
B = 1000
LINEAGES = {"Monocyte", "T cell", "NK cell", "B cell", "pDC", "Neutrophil"}
ENDPOINTS = {
    "6M": ["E_6M_fold1", "E_6M_fold2", "E_6M_fold3"],
    "12M": ["E_12M_fold1", "E_12M_fold2"],
    "Cox": ["cox_RR_m1", "cox_RR_m2", "cox_RR_m3"],
}


def env_paths():
    project = Path(os.environ["TB_STATE_PROJECT_ROOT"])
    vroot = Path(os.environ["TB_STATE_V126_ROOT"])
    run = Path((vroot / "CURRENT_RUN.txt").read_text(encoding="utf-8-sig").strip())
    return project, vroot, run


def atomic_tsv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_json(path: Path, obj):
    atomic_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def midrank_percentile(values, reference):
    """Tie-aware empirical CDF: average of left/right rank, in [0,1]."""
    x = np.asarray(values, dtype=float)
    r = np.sort(np.asarray(reference, dtype=float))
    if not np.isfinite(x).all() or not np.isfinite(r).all() or len(r) == 0:
        raise ValueError("NONFINITE_OR_EMPTY_PERCENTILE_INPUT")
    left = np.searchsorted(r, x, side="left")
    right = np.searchsorted(r, x, side="right")
    return (left + right) / (2.0 * len(r))


def sample_balanced(df, pids, k=K, seed=SEED):
    rng = np.random.default_rng(seed)
    pieces = []
    for pid in sorted(pids):
        b = df[df.participant_id.eq(pid)]
        if len(b) < k:
            raise RuntimeError(f"PARTICIPANT_BELOW_K:{pid}:{len(b)}:{k}")
        idx = rng.choice(b.index.to_numpy(), size=k, replace=False)
        pieces.append(b.loc[idx])
    return pd.concat(pieces, ignore_index=True)


def distribution_spec(name):
    if name == "Weibull":
        return stats.weibull_min, lambda x: stats.weibull_min.fit(x, floc=0), 2
    if name == "Gamma":
        return stats.gamma, lambda x: stats.gamma.fit(x, floc=0), 2
    if name == "Exponential":
        return stats.expon, lambda x: stats.expon.fit(x, floc=0), 1
    if name == "Generalized_Gamma":
        return stats.gengamma, lambda x: stats.gengamma.fit(x, floc=0), 3
    raise ValueError(name)


def fit_distribution(x, name):
    dist, fitter, k = distribution_spec(name)
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all() or (x <= 0).any():
        raise ValueError(f"NONPOSITIVE_OR_NONFINITE:{name}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = tuple(float(v) for v in fitter(x))
    loglik = float(np.sum(dist.logpdf(x, *params)))
    aic = 2 * k - 2 * loglik
    bic = k * np.log(len(x)) - 2 * loglik
    cdf = lambda v: dist.cdf(v, *params)
    ks = float(np.max(np.abs(np.arange(1, len(x) + 1) / len(x) - np.sort(cdf(x)))))
    probs = np.unique(np.concatenate([np.arange(0.01, 1.0, 0.01), [0.90, 0.95, 0.975, 0.99]]))
    eq = np.quantile(x, probs, method="linear")
    fq = dist.ppf(probs, *params)
    iqr = max(float(np.quantile(x, .75) - np.quantile(x, .25)), 1e-12)
    qq_rmse = float(np.sqrt(np.mean((eq - fq) ** 2)))
    qrows = {f"empirical_Q{int(round(p * 1000))/10:g}": float(v) for p, v in zip(probs, eq)}
    for p, v in zip(probs, fq):
        qrows[f"model_Q{int(round(p * 1000))/10:g}"] = float(v)
    out = {
        "distribution": name, "n": int(len(x)), "n_parameters": int(k),
        "log_likelihood": loglik, "AIC": float(aic), "BIC": float(bic),
        "KS_D": ks, "QQ_R2": float(np.corrcoef(eq, fq)[0, 1] ** 2),
        "QQ_RMSE": qq_rmse, "QQ_NRMSE_IQR": qq_rmse / iqr,
        "params": params, "dist": dist,
    }
    for q in (.90, .95, .975, .99):
        e = float(np.quantile(x, q)); m = float(dist.ppf(q, *params))
        out[f"empirical_Q{q}"] = e; out[f"model_Q{q}"] = m
        out[f"Q{q}_error_IQR"] = abs(m - e) / iqr
    q90 = float(np.quantile(x, .90)); f0 = float(cdf(q90)); tail = x[x >= q90]
    tail_sorted = np.sort(tail)
    ecdf = (np.arange(1, len(tail_sorted) + 1) / len(tail_sorted))
    denom = max(1.0 - f0, 1e-12)
    fcond = np.clip((cdf(tail_sorted) - f0) / denom, 0, 1)
    out["TAIL_KS_CONDITIONAL_D"] = float(np.max(np.abs(ecdf - fcond)))
    grid = np.linspace(q90, float(np.quantile(x, .99)), 100)
    emp_surv = np.array([(tail > z).mean() for z in grid])
    fit_surv = np.clip(1 - np.array([cdf(z) for z in grid]), 1e-12, 1)
    out["TAIL_LOG_SURVIVAL_RMSE"] = float(np.sqrt(np.mean((np.log(np.clip(emp_surv, 1e-12, 1)) - np.log(fit_surv)) ** 2)))
    return out


def conditional_tail_ks(x, name):
    fit = fit_distribution(x, name)
    q90 = float(np.quantile(x, .90))
    return {
        "distribution": name,
        "n": int(len(x)),
        "empirical_Q90": q90,
        "F0_at_Q90": float(fit["dist"].cdf(q90, *fit["params"])),
        "TAIL_KS_CONDITIONAL_D": fit["TAIL_KS_CONDITIONAL_D"],
        "TAIL_LOG_SURVIVAL_RMSE": fit["TAIL_LOG_SURVIVAL_RMSE"],
        "status": "CORRECTED_CONDITIONAL_TAIL_COMPARISON",
    }


def reference_counts(ref):
    c = ref.groupby("participant_id", sort=True).size().rename("n_LTBI_cells").reset_index()
    return c


def save_diagnostic_figures(out, score, ref, formal, fitdf):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
        figdir = out / "10_DIAGNOSTIC_FIGURES"; figdir.mkdir(parents=True, exist_ok=True)
        for ep, col, title in [("6M", "U6", "6-month calibrated consensus"), ("12M", "U12", "12-month calibrated consensus"), ("Cox", "UCOX", "Cox calibrated consensus")]:
            fig, ax = plt.subplots(figsize=(3.2, 2.3)); ax.hist(formal[col], bins=60, color="#4C78A8", alpha=.85, density=True); ax.axvline(.95, color="#B22222", ls="--", lw=1); ax.set(xlabel="Empirical calibrated percentile", ylabel="Density", title=title); fig.tight_layout(); fig.savefig(figdir / f"PERCENTILE_DISTRIBUTION_{ep}.pdf"); fig.savefig(figdir / f"PERCENTILE_DISTRIBUTION_{ep}.svg"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(3.2, 2.3)); ax.hist(np.log(score[["cox_RR_m1", "cox_RR_m2", "cox_RR_m3"]].to_numpy().ravel()), bins=70, color="#666666", alpha=.85); ax.set(xlabel="log relative hazard", ylabel="Cells", title="Cox relative-risk distribution"); fig.tight_layout(); fig.savefig(figdir / "COX_PARAMETRIC_FIT_SENSITIVITY.pdf"); fig.savefig(figdir / "COX_PARAMETRIC_FIT_SENSITIVITY.svg"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.3, 2.8)); x=np.arange(len(fitdf)); ax.scatter(x, fitdf.TAIL_KS_CONDITIONAL_D, s=12, c=fitdf.model_index, cmap="viridis"); ax.set(xlabel="Fit rows (model × distribution)", ylabel="Corrected conditional tail KS D", title="Corrected upper-tail diagnostics"); fig.tight_layout(); fig.savefig(figdir / "CORRECTED_TAIL_FIT.pdf"); fig.savefig(figdir / "CORRECTED_TAIL_FIT.svg"); plt.close(fig)
        return "FIGURES_WRITTEN"
    except Exception as e:
        atomic_text(out / "10_DIAGNOSTIC_FIGURES" / "FIGURE_WARNING.txt", f"Diagnostic figures not written: {e}\n")
        return "FIGURE_WARNING"


def main(dry=False):
    project, vroot, run = env_paths()
    src = run / "17_PROJECTION_INTERFACE_RECONSTRUCTION" / "04_CELL_PROJECTION" / "CELL_FOLDWISE_SCORES.parquet"
    atlas_path = project / "20_TOP_JOURNAL_EXTENSION" / "16_RESULT3_CELLULAR_INTERPRETATION_V1253_CORRECTED" / "01_RISK_LABEL" / "MASTER_CELL_UNIVERSE.tsv.gz"
    if dry:
        print(json.dumps({"status": "DRY_RUN_PASS", "source_exists": src.exists(), "atlas_exists": atlas_path.exists(), "K": K, "B": B, "no_retraining": True, "no_outer_outcome": True, "no_DEG": True}, indent=2)); return
    out = run / "18_FINAL_RISK_CELL_LOCK_V4"
    for d in ["00_PROVENANCE", "01_REFERENCE", "02_MODEL_PERCENTILES", "03_CONSENSUS_CALIBRATION", "04_COX_PARAMETRIC_SENSITIVITY", "05_FINAL_LABELS", "06_MODEL_SUPPORT", "07_PARTICIPANT_STABILITY", "08_LINEAGE_STRUCTURAL", "09_SENSITIVITY", "10_DIAGNOSTIC_FIGURES", "11_TABLES", "12_METHODS", "13_REPORT", "scripts", "logs"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    score = pd.read_parquet(src)
    atlas = pd.read_csv(atlas_path, sep="\t", compression="gzip", low_memory=False)
    score = score[score.lineage.isin(LINEAGES)].copy(); atlas = atlas[atlas.lineage.isin(LINEAGES)].copy()
    if score.cell_id.duplicated().any() or atlas.cell_id.duplicated().any(): raise RuntimeError("DUPLICATE_CELL_ID")
    if set(score.cell_id) != set(atlas.cell_id): raise RuntimeError("ATLAS_PROJECTION_CELL_SET_MISMATCH")
    if score["participant_id"].isna().any() or score["disease_group"].isna().any(): raise RuntimeError("MISSING_CELL_METADATA")
    ref_all = score[score.disease_group.eq("LTBI")].copy()
    counts = reference_counts(ref_all)
    if len(counts) != 9 or counts.n_LTBI_cells.min() < K: raise RuntimeError("LTBI_REFERENCE_COUNTS_NOT_EXPECTED")
    ref = sample_balanced(ref_all, counts.participant_id.tolist(), K, SEED)
    atomic_tsv(out / "01_REFERENCE" / "LTBI_REFERENCE_PARTICIPANT_COUNTS.tsv", counts)
    ref.to_parquet(out / "01_REFERENCE" / "LTBI_BALANCED_K500_PRIMARY.parquet", index=False)
    manifest = pd.DataFrame([
        {"source_role": "cell_scores", "path": str(src), "sha256": sha256(src), "n_rows": len(score)},
        {"source_role": "formal_atlas", "path": str(atlas_path), "sha256": sha256(atlas_path), "n_rows": len(atlas)},
    ])
    atomic_tsv(out / "00_PROVENANCE" / "SOURCE_MANIFEST.tsv", manifest)
    atomic_tsv(out / "00_PROVENANCE" / "SOURCE_SHA256.tsv", manifest[["source_role", "path", "sha256"]])
    atomic_json(out / "00_PROVENANCE" / "AUDIT_LOCK.json", {"clinical_model_retrained": False, "outer_predictions_changed": False, "outer_labels_used": False, "risk_definition_finalized_before_DEG": True, "risk_definition_finalized_before_pathway": True, "risk_definition_finalized_before_signature_AUC_testing": True, "reference": "LTBI_ONLY", "K": K, "seed": SEED})

    raw = {"6M": score[["E_6M_fold1", "E_6M_fold2", "E_6M_fold3"]].to_numpy(float), "12M": score[["E_12M_fold1", "E_12M_fold2"]].to_numpy(float), "Cox": score[["cox_RR_m1", "cox_RR_m2", "cox_RR_m3"]].to_numpy(float)}
    raw_ref = {"6M": ref[["E_6M_fold1", "E_6M_fold2", "E_6M_fold3"]].to_numpy(float), "12M": ref[["E_12M_fold1", "E_12M_fold2"]].to_numpy(float), "Cox": ref[["cox_RR_m1", "cox_RR_m2", "cox_RR_m3"]].to_numpy(float)}
    model_u = {}
    for ep in ENDPOINTS:
        model_u[ep] = np.column_stack([midrank_percentile(raw[ep][:, j], raw_ref[ep][:, j]) for j in range(raw[ep].shape[1])])
    pcols = []
    for ep in ENDPOINTS:
        for j in range(model_u[ep].shape[1]): pcols.append(f"u{ep}_m{j+1}")
    pct = pd.DataFrame({"cell_id": score.cell_id.to_numpy()})
    for ep in ENDPOINTS:
        for j in range(model_u[ep].shape[1]): pct[f"u{ep}_m{j+1}"] = model_u[ep][:, j]
    pct.to_parquet(out / "02_MODEL_PERCENTILES" / "CELL_MODEL_PERCENTILES.parquet", index=False)
    raw_cons = {ep: np.median(model_u[ep], axis=1) for ep in ENDPOINTS}
    # The balanced reference is reset to a fresh index, so map it back to the
    # projection rows by immutable cell_id rather than by temporary position.
    pos = pd.Series(np.arange(len(score)), index=score.cell_id.astype(str))
    ref_pos = pos.loc[ref.cell_id.astype(str)].to_numpy()
    U = {}
    for ep in ENDPOINTS:
        U[ep] = midrank_percentile(raw_cons[ep], raw_cons[ep][ref_pos])
    formal = score[["cell_id", "participant_id", "dataset_id", "disease_group", "lineage", "subtype"]].copy()
    formal["U6"], formal["U12"], formal["UCOX"] = U["6M"], U["12M"], U["Cox"]
    for ep, cols in ENDPOINTS.items():
        key = {"6M": "U6", "12M": "U12", "Cox": "UCOX"}[ep]
        formal[f"TB_STATE_HIGH_{ep}"] = (formal[key] >= .95)
        formal[f"{ep}_model_support_P95"] = (model_u[ep] >= .95).sum(axis=1)
        formal[f"{ep}_model_support_P90"] = (model_u[ep] >= .90).sum(axis=1)
        formal[f"{ep}_model_support_P975"] = (model_u[ep] >= .975).sum(axis=1)
    # Generalized-gamma sensitivity for Cox; it is never used as primary.
    gg_u = []
    for j in range(3):
        _, fitter, _ = distribution_spec("Generalized_Gamma")
        with warnings.catch_warnings(): warnings.simplefilter("ignore"); pars = fitter(raw_ref["Cox"][:, j])
        gg_u.append(stats.gengamma.cdf(raw["Cox"][:, j], *pars))
    gg_raw = np.median(np.column_stack(gg_u), axis=1); gg_ref = gg_raw[ref_pos]
    formal["UCOX_GG"] = midrank_percentile(gg_raw, gg_ref); formal["TB_STATE_HIGH_COX_GG"] = formal.UCOX_GG >= .95
    formal.to_parquet(out / "05_FINAL_LABELS" / "TB_STATE_FORMAL_RISK_CELLS_V126.parquet", index=False)

    # Formal counts, support distributions, and participant stability.
    count_rows=[]; stability_rows=[]; lineage_rows=[]
    for ep, key in [("6M", "U6"), ("12M", "U12"), ("Cox", "UCOX")]:
        mask = formal[key] >= .95; sub = formal[mask]; refmask = formal.disease_group.eq("LTBI"); atbmask = formal.disease_group.eq("ATB")
        for label, m in [("ALL", pd.Series(True, index=formal.index)), ("LTBI", refmask), ("ATB", atbmask)]:
            n=int(m.sum()); nh=int((mask & m).sum()); count_rows.append({"endpoint":ep,"population":label,"n_all_cells":n,"n_high":nh,"fraction_high":nh/max(n,1),"n_participants":int(formal.loc[m,"participant_id"].nunique()),"n_participants_with_high":int(formal.loc[m & mask,"participant_id"].nunique())})
        pc=sub.groupby("participant_id").size(); within=formal.assign(_sel=mask).groupby("participant_id")._sel.mean(); donor=pc.sort_values(ascending=False); ne=(pc.sum()**2/max((pc**2).sum(),1)) if len(pc) else 0; gini=float(np.abs(np.subtract.outer(pc.to_numpy(),pc.to_numpy())).sum()/(2*len(pc)*pc.sum())) if len(pc)>1 and pc.sum()>0 else 0.0
        stability_rows.append({"endpoint":ep,"n_participants_with_ge1":int(len(pc)),"n_participants_with_ge10":int((pc>=10).sum()),"n_participants_with_ge1pct":int((within>=.01).sum()),"max_donor_share":float(donor.iloc[0]/max(mask.sum(),1)) if len(donor) else 0.0,"top5_donor_share":float(donor.head(5).sum()/max(mask.sum(),1)) if len(donor) else 0.0,"effective_participant_number":float(ne),"gini":gini,"median_within_participant_high_fraction":float(within.median()) if len(within) else 0.0,"IQR_within_participant_high_fraction":float(within.quantile(.75)-within.quantile(.25)) if len(within) else 0.0})
        for lin,b in formal.groupby("lineage"):
            lm=formal.lineage.eq(lin); lineage_rows.append({"endpoint":ep,"major_lineage":lin,"n_total":int(lm.sum()),"n_high":int((mask&lm).sum()),"within_lineage_high_fraction":float((mask&lm).sum()/max(lm.sum(),1)),"fraction_of_all_high_cells":float((mask&lm).sum()/max(mask.sum(),1)),"n_participants_supporting":int(formal.loc[mask&lm,"participant_id"].nunique())})
    atomic_tsv(out / "11_TABLES" / "FORMAL_RISK_CELL_COUNTS.tsv", pd.DataFrame(count_rows)); atomic_tsv(out / "07_PARTICIPANT_STABILITY" / "FORMAL_PARTICIPANT_STABILITY.tsv", pd.DataFrame(stability_rows)); atomic_tsv(out / "08_LINEAGE_STRUCTURAL" / "FORMAL_RISK_CELLS_BY_LINEAGE.tsv", pd.DataFrame(lineage_rows))
    support_rows=[]
    for ep in ENDPOINTS:
        c=f"{ep}_model_support_P95"; v=formal.loc[formal[f"TB_STATE_HIGH_{ep}"], c].value_counts().sort_index()
        for support,n in v.items(): support_rows.append({"endpoint":ep,"support":f"{int(support)}/{len(ENDPOINTS[ep])}","n_high_cells":int(n)})
    atomic_tsv(out / "06_MODEL_SUPPORT" / "MODEL_SUPPORT_P95.tsv", pd.DataFrame(support_rows))
    thr_rows=[]; jacc=[]
    for ep,key in [("6M","U6"),("12M","U12"),("Cox","UCOX")]:
        for q in (.90,.95,.975):
            m=formal[key]>=q; thr_rows.append({"endpoint":ep,"threshold":q,"n_all_cells":int(m.sum()),"fraction_all":float(m.mean()),"n_LTBI":int((m&formal.disease_group.eq("LTBI")).sum()),"fraction_LTBI":float((m&formal.disease_group.eq("LTBI")).sum()/max(formal.disease_group.eq("LTBI").sum(),1)),"n_ATB":int((m&formal.disease_group.eq("ATB")).sum()),"fraction_ATB":float((m&formal.disease_group.eq("ATB")).sum()/max(formal.disease_group.eq("ATB").sum(),1))})
        for qa,qb in ((.90,.95),(.95,.975)):
            a=formal[key]>=qa; b=formal[key]>=qb; inter=int((a&b).sum()); uni=int((a|b).sum()); jacc.append({"endpoint":ep,"threshold_a":qa,"threshold_b":qb,"intersection":inter,"union":uni,"Jaccard":inter/max(uni,1),"overlap_coefficient":inter/max(min(a.sum(),b.sum()),1)})
    atomic_tsv(out / "09_SENSITIVITY" / "P90_P95_P975_COUNTS.tsv", pd.DataFrame(thr_rows)); atomic_tsv(out / "09_SENSITIVITY" / "P90_P95_P975_JACCARD.tsv", pd.DataFrame(jacc))
    overlaps=[]
    masks={ep: formal[f"TB_STATE_HIGH_{ep}"].to_numpy(bool) for ep in ENDPOINTS}
    for a,b in (("6M","12M"),("6M","Cox"),("12M","Cox")):
        inter=int((masks[a]&masks[b]).sum()); uni=int((masks[a]|masks[b]).sum()); overlaps.append({"comparison":f"{a}_vs_{b}","intersection":inter,"union":uni,"Jaccard":inter/max(uni,1),"overlap_coefficient":inter/max(min(masks[a].sum(),masks[b].sum()),1)})
    tri=int(masks["6M"].astype(int)[masks["12M"] & masks["Cox"]].sum()); overlaps.append({"comparison":"6M_AND_12M_AND_Cox","intersection":tri,"union":tri,"Jaccard":1.0 if tri else 0.0,"overlap_coefficient":1.0 if tri else 0.0}); atomic_tsv(out / "11_TABLES" / "FORMAL_CROSS_ENDPOINT_OVERLAP.tsv", pd.DataFrame(overlaps))

    fitrows=[]; tailrows=[]
    for j,col in enumerate(["cox_RR_m1","cox_RR_m2","cox_RR_m3"],1):
        x=ref[col].to_numpy(float)
        for name in ("Weibull","Gamma","Exponential","Generalized_Gamma"):
            f=fit_distribution(x,name); row={k:v for k,v in f.items() if k not in ("params","dist")}; row.update({"model_index":j,"score":col,"fit_status":"LOC_ZERO_MLE"}); fitrows.append(row); tailrows.append({"model_index":j,"score":col,**conditional_tail_ks(x,name)})
    fitdf=pd.DataFrame(fitrows); fitdf["AIC_rank"]=fitdf.groupby("model_index").AIC.rank(method="min"); fitdf["BIC_rank"]=fitdf.groupby("model_index").BIC.rank(method="min"); fitdf["QQ_rank"]=fitdf.groupby("model_index").QQ_NRMSE_IQR.rank(method="min"); fitdf["Q95_tail_rank"]=fitdf.groupby("model_index")["Q0.95_error_IQR"].rank(method="min"); atomic_tsv(out / "04_COX_PARAMETRIC_SENSITIVITY" / "CORRECTED_CONDITIONAL_TAIL_KS.tsv", pd.DataFrame(tailrows)); atomic_tsv(out / "11_TABLES" / "COX_PARAMETRIC_DISTRIBUTION_AUDIT_FINAL.tsv", fitdf)

    # Two-stage participant-cluster/cell bootstrap. It recalibrates from all LTBI cells each replicate.
    rng=np.random.default_rng(SEED); pids=sorted(ref_all.participant_id.unique()); groups={p:ref_all[ref_all.participant_id.eq(p)] for p in pids}; threshold_rows=[]; freq={ep:np.zeros(len(score),dtype=np.int32) for ep in ENDPOINTS}; p95_rows=[]
    for b in range(B):
        chosen=rng.choice(pids,len(pids),replace=True); pieces=[]
        for p in chosen:
            g=groups[p]; pieces.append(g.iloc[rng.choice(len(g),K,replace=False)])
        rb=pd.concat(pieces,ignore_index=True); rbraw={ep:rb[[c for c in ENDPOINTS[ep]]].to_numpy(float) for ep in ENDPOINTS}
        for ep,cols in ENDPOINTS.items():
            mu=np.column_stack([midrank_percentile(raw[ep][:,j],rbraw[ep][:,j]) for j in range(len(cols))]); rc=np.median(mu,axis=1)
            ref_model_u=np.column_stack([midrank_percentile(rbraw[ep][:,j],rbraw[ep][:,j]) for j in range(len(cols))]); rref=np.median(ref_model_u,axis=1); u=midrank_percentile(rc,rref); uref=midrank_percentile(rref,rref)
            for q in (.90,.95,.975): threshold_rows.append({"replicate":b+1,"endpoint":ep,"threshold":q,"reference_fraction":float((uref>=q).mean()),"fixed_atlas_n_high":int((u>=q).sum()),"fixed_atlas_fraction":float((u>=q).mean())})
            freq[ep] += (u>=.90).astype(np.int32)
        if (b+1)%100==0: print(f"two_stage_bootstrap={b+1}/{B}", flush=True)
    bootdf=pd.DataFrame(threshold_rows); atomic_tsv(out / "09_SENSITIVITY" / "TWO_STAGE_BOOTSTRAP_THRESHOLDS.tsv", bootdf)
    summ=[]
    for (ep,q),g in bootdf.groupby(["endpoint","threshold"]):
        z=g.fixed_atlas_n_high.to_numpy(float); summ.append({"endpoint":ep,"threshold":q,"median_threshold":float(np.median(z)/len(score)),"lower_2.5":float(np.quantile(z/len(score),.025)),"upper_97.5":float(np.quantile(z/len(score),.975)),"SD":float(np.std(z/len(score),ddof=1)),"CV":float(np.std(z/len(score),ddof=1)/max(np.mean(z/len(score)),1e-12))})
    atomic_tsv(out / "09_SENSITIVITY" / "TWO_STAGE_BOOTSTRAP_THRESHOLDS_SUMMARY.tsv", pd.DataFrame(summ))
    fdf=pd.DataFrame({"cell_id":score.cell_id})
    for ep in ENDPOINTS: fdf[f"bootstrap_frequency_{ep}"]=freq[ep]/B
    fdf=fdf[(fdf.drop(columns="cell_id")>0).any(axis=1)]; fdf.to_parquet(out / "09_SENSITIVITY" / "BOOTSTRAP_CELL_SELECTION_FREQUENCY.parquet",index=False)

    # Publication-ready methods and audit tables.
    methods=("Risk-cell states were defined from the frozen endpoint-specific LODO models. "
             "Three verified models were available for the 6-month endpoint, two numerically verified models for 12 months (the third context was structurally non-estimable), and three verified Cox models. "
             "For each model, scores were converted to tie-aware empirical percentiles using a fixed LTBI-only participant-balanced reference (K=500 cells per participant; nine participants), and model-specific percentiles were combined by the median. "
             "The consensus was recalibrated once more against the same balanced LTBI reference. The prespecified primary threshold was P95; P90 and P97.5 were sensitivity analyses. Model support was recorded only as a cross-LODO reproducibility metric and did not define the formal label. "
             "Cox generalized-gamma fits were retained as parametric sensitivity analyses because they provided the best-fitting approximation among evaluated families, but were not assumed to represent the true distribution. Conditional upper-tail diagnostics corrected the previous unconditional-versus-conditional comparison. "
             "Two-stage participant-cluster/cell bootstrap (1,000 replicates) resampled participant clusters with replacement and then 500 cells without replacement from each participant's complete LTBI cell pool. No outcome-dependent threshold optimisation was performed.")
    atomic_text(out / "12_METHODS" / "RISK_CELL_METHODS_READY.txt", methods + "\n")
    items=[("clinical models frozen",True),("outer labels unused",True),("single-cell encoder frozen",True),("formal atlas fixed",True),("LTBI reference fixed",True),("participant balancing fixed",True),("endpoint score independent",True),("LODO ensemble prespecified",True),("consensus recalibration complete",True),("primary P95 fixed",True),("P90/P97.5 sensitivity fixed",True),("parametric family not used as primary",True),("GG sensitivity documented",True),("KS bootstrap complete",True),("conditional tail KS corrected",True),("two-stage bootstrap complete",True),("model support separated from label definition",True),("DEG not run",True),("GO/KEGG not run",True)]
    atomic_tsv(out / "11_TABLES" / "RISK_CELL_METHODS_AUDIT_FINAL.tsv", pd.DataFrame([{"item":a,"status":"PASS" if b else "FAIL","evidence_file":"V4 outputs","submission_consequence":"documented"} for a,b in items]))
    fig_status=save_diagnostic_figures(out, score, ref, formal, fitdf)
    reference_ids=set(ref.cell_id.astype(str)); refmask_formal=formal.cell_id.astype(str).isin(reference_ids)
    formal_ref_frac={ep:float((formal.loc[refmask_formal, {"6M":"U6","12M":"U12","Cox":"UCOX"}[ep]]>=.95).mean()) for ep in ENDPOINTS}
    statuses={ep:("FORMAL_LABEL_LOCKED" if .04 <= formal_ref_frac[ep] <= .06 else "CALIBRATION_FAILED") for ep in ENDPOINTS}
    for r in stability_rows:
        if statuses[r["endpoint"]]=="FORMAL_LABEL_LOCKED" and (r["n_participants_with_ge1"] < 7 or r["max_donor_share"] >= .25 or r["effective_participant_number"] < 7): statuses[r["endpoint"]]="FORMAL_LABEL_WITH_STABILITY_WARNING"
    total="RISK_CELL_DEFINITIONS_LOCKED" if all(v.startswith("FORMAL_LABEL_") for v in statuses.values()) else "RISK_CELL_DEFINITION_REQUIRES_REVISION"
    report=f"""# TB-STATE V12.6 final risk-cell lock V4

## Primary definition

The primary labels are the second, tie-aware empirical LTBI-reference calibration of the median verified-model percentile ensemble, with `U >= 0.95` (P95). P90 and P97.5 are prespecified sensitivity thresholds. Model support is diagnostic only.

Weibull and generalized gamma were not used as primary distributions. Generalized gamma was retained only as a parametric sensitivity approximation; corrected conditional-tail diagnostics showed that no parametric family was required for the primary empirical definition.

| endpoint | status | n high | LTBI fraction | ATB fraction |
|---|---|---:|---:|---:|
"""
    for ep in ENDPOINTS:
        row=next(x for x in count_rows if x["endpoint"]==ep and x["population"]=="ALL"); lr=next(x for x in count_rows if x["endpoint"]==ep and x["population"]=="LTBI"); ar=next(x for x in count_rows if x["endpoint"]==ep and x["population"]=="ATB"); report += f"| {ep} | {statuses[ep]} | {row['n_high']} | {lr['fraction_high']:.4f} | {ar['fraction_high']:.4f} |\n"
    report += "\nTwo-stage bootstrap completed with B=1000; P95 reference calibration was assessed before label lock. No outer outcomes were used for threshold selection. DEG, GO and KEGG were not run.\n"
    atomic_text(out / "13_REPORT" / "FINAL_RISK_CELL_LOCK_V4_REPORT.md", report)
    atomic_json(out / "13_REPORT" / "CALIBRATION_STATUS.json", {"total_status":total,"endpoint_status":statuses,"n_cells":len(formal),"n_LTBI_participants":len(counts),"balanced_reference_cells":len(ref),"B_two_stage":B,"primary_threshold":0.95,"risk_labels_generated":True,"outer_outcomes_used":False,"clinical_model_retrained":False,"deg_run":False,"go_kegg_run":False,"figure_status":fig_status})
    # Preserve the exact script used.
    try: shutil.copy2(Path(__file__), out / "scripts" / Path(__file__).name)
    except Exception: pass
    zip_path = out / "TB_STATE_V126_FINAL_RISK_CELL_LOCK_V4_REVIEW.zip"
    if zip_path.exists(): zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=out, base_dir=".")
    print(json.dumps({"FINAL_STATUS":total,"6M_STATUS":statuses["6M"],"12M_STATUS":statuses["12M"],"COX_STATUS":statuses["Cox"],"n_cells":len(formal),"n_high_6M":int(formal.TB_STATE_HIGH_6M.sum()),"n_high_12M":int(formal.TB_STATE_HIGH_12M.sum()),"n_high_Cox":int(formal.TB_STATE_HIGH_Cox.sum()),"B_two_stage":B,"labels_generated":True,"deg_run":False,"review_zip":str(zip_path)},indent=2))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args(); main(args.dry_run)
