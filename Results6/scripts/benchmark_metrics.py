from pathlib import Path
import hashlib
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("RESULT6_INPUT", "inputs"))
PREDICTIONS = ROOT / "V126_OUTER_PREDICTIONS.tsv"
PROGRAMME_SCORES = ROOT / "PATIENT_SCORES.tsv.gz"
ENDPOINTS = ROOT / "FORMAL_ENDPOINT_OBSERVATIONS.tsv"
BULK = ROOT / "BULK_LOG2CPM.tsv"
DEG_DIR = ROOT / "DEG_TABLES"

B = 2000
BASE_SEED = 20260920

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    }
)


def stable_seed(*parts):
    token = "|".join(map(str, parts)).encode("utf-8")
    return BASE_SEED + int(hashlib.sha256(token).hexdigest()[:8], 16) % 1_000_000


def harrell_c(time_values, event_values, risk_score):
    t = np.asarray(time_values, dtype=float)
    e = np.asarray(event_values, dtype=int)
    s = np.asarray(risk_score, dtype=float)
    ok = np.isfinite(t) & np.isfinite(e) & np.isfinite(s)
    t, e, s = t[ok], e[ok], s[ok]
    upper = np.triu(np.ones((len(t), len(t)), dtype=bool), 1)
    i_earlier = upper & (t[:, None] < t[None, :]) & (e[:, None] == 1)
    j_earlier = upper & (t[None, :] < t[:, None]) & (e[None, :] == 1)
    # Harrell/lifelines convention: at equal observed time, an event and a censoring are comparable.
    tied_time = upper & (t[:, None] == t[None, :]) & (e[:, None] != e[None, :])
    i_earlier |= tied_time & (e[:, None] == 1)
    j_earlier |= tied_time & (e[None, :] == 1)
    comparable = int(i_earlier.sum() + j_earlier.sum())
    if comparable == 0:
        return np.nan
    concordant = int((i_earlier & (s[:, None] > s[None, :])).sum())
    concordant += int((j_earlier & (s[None, :] > s[:, None])).sum())
    ties = int((i_earlier & (s[:, None] == s[None, :])).sum())
    ties += int((j_earlier & (s[None, :] == s[:, None])).sum())
    return float((concordant + 0.5 * ties) / comparable)


def weighted_auc(y_true, score, sample_weight=None):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    w = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s) & np.isfinite(w) & (w > 0)
    y, s, w = y[ok], s[ok], w[ok]
    if np.unique(y).size != 2:
        return np.nan
    order = np.argsort(s, kind="stable")
    y, s, w = y[order], s[order], w[order]
    total_pos = w[y == 1].sum()
    total_neg = w[y == 0].sum()
    if total_pos <= 0 or total_neg <= 0:
        return np.nan
    numerator = 0.0
    cumulative_neg = 0.0
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and s[end] == s[start]:
            end += 1
        group_y = y[start:end]
        group_w = w[start:end]
        pos_w = group_w[group_y == 1].sum()
        neg_w = group_w[group_y == 0].sum()
        numerator += pos_w * (cumulative_neg + 0.5 * neg_w)
        cumulative_neg += neg_w
        start = end
    return float(numerator / (total_pos * total_neg))


def bootstrap_metric(frame, metric, seed):
    ids = frame["participant_id"].astype(str).to_numpy()
    groups = {p: np.flatnonzero(ids == p) for p in np.unique(ids)}
    names = np.array(sorted(groups))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(B):
        sampled = rng.choice(names, size=len(names), replace=True)
        idx = np.concatenate([groups[p] for p in sampled])
        d = frame.iloc[idx]
        if metric == "Harrell_C":
            value = harrell_c(d["time"], d["y_true"], d["score"])
        else:
            if d["y_true"].nunique() != 2:
                continue
            value = weighted_auc(
                d["y_true"].astype(int),
                d["score"].astype(float),
                d["weight"].astype(float),
            )
        if np.isfinite(value):
            values.append(float(value))
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, 0
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), int(len(arr))


def loco_mean_z_scores(expr, cohorts, genes):
    cohorts = np.asarray(cohorts).astype(str)
    use = [g for g in genes if g in expr.columns]
    scores = np.full(len(expr), np.nan, dtype=float)
    if not use:
        return scores, 0
    x = expr.loc[:, use].astype(float)
    usable_any = set()
    for cohort in sorted(np.unique(cohorts)):
        test = cohorts == cohort
        train = ~test
        mu = x.loc[train].mean(axis=0)
        sd = x.loc[train].std(axis=0, ddof=0)
        valid = sd.gt(0) & np.isfinite(sd) & np.isfinite(mu)
        usable = valid.index[valid].tolist()
        usable_any.update(usable)
        scores[test] = ((x.loc[test, usable] - mu[usable]) / sd[usable]).mean(axis=1).to_numpy()
    return scores, len(usable_any)


pred = pd.read_csv(PREDICTIONS, sep="\t")
prog = pd.read_csv(PROGRAMME_SCORES, sep="\t")
endpoints = pd.read_csv(ENDPOINTS, sep="\t")
prog = prog[(prog["threshold"] == "R1") & (prog["cohort"].isin(["GSE79362", "GSE94438"]))].copy()

# One time-to-event row per participant for Cox evaluation and for attaching time to native E_COX scores.
h1_time = (
    endpoints[endpoints["endpoint"] == "E_ALL"]
    .sort_values(["participant_id", "baseline", "sample_date"], ascending=[True, False, True], kind="stable")
    .drop_duplicates("participant_id")[["participant_id", "cohort", "time_to_event_or_censor_days", "y_true"]]
    .rename(columns={"time_to_event_or_censor_days": "time"})
)

endpoint_map = {"H1": "E_ALL", "H2": "E_6M", "H3": "E_12M"}
methods = ["TBSTATE", "SIDISH", "SCAB", "SCPAS", "DEGAS"]
display = {
    "TBSTATE": "TB-STATE",
    "SIDISH": "SIDISH",
    "SCAB": "scAB",
    "SCISSOR": "Scissor",
    "SCPAS": "scPAS",
    "DEGAS": "DEGAS",
}
colors = {
    "TBSTATE": "#D98693",
    "SIDISH": "#E7B46A",
    "SCAB": "#8FBCAD",
    "SCISSOR": "#A89BC8",
    "SCPAS": "#A9D2EA",
    "DEGAS": "#D4A3BE",
}
panel_methods = {
    "H1": ["TBSTATE", "SIDISH", "SCAB", "SCPAS"],
    "H2": ["TBSTATE", "SCAB", "SCPAS", "DEGAS"],
    "H3": ["TBSTATE", "SCAB", "SCPAS", "DEGAS"],
}

# H1 is now evaluated as broad-future-progression AUROC on the full E_ALL population.
# Recompute each comparator's R1 programme score for all E_ALL baseline samples.
h1_meta = endpoints[endpoints["endpoint"] == "E_ALL"].copy()
h1_meta = h1_meta.sort_values(["participant_id", "baseline", "sample_date"], ascending=[True, False, True], kind="stable")
h1_meta = h1_meta.drop_duplicates("participant_id").reset_index(drop=True)
bulk = pd.read_csv(BULK, sep="\t").rename(columns={"sample_id": "GSM"}).set_index("GSM")
bulk.index = bulk.index.astype(str)
if not h1_meta["GSM"].astype(str).isin(bulk.index).all():
    raise RuntimeError("E_ALL baseline sample missing from bulk expression")
h1_expr = bulk.reindex(h1_meta["GSM"].astype(str)).reset_index(drop=True)
h1_programme_rows = []
for method in ["SIDISH", "SCAB", "SCPAS"]:
    table = pd.read_csv(DEG_DIR / f"{method}_H1_ALL_GENES.tsv.gz", sep="\t")
    selected_flag = table["R1_selected"]
    if selected_flag.dtype != bool:
        selected_flag = selected_flag.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})
    selected_genes = table.loc[selected_flag, "gene"].astype(str).tolist()
    scores, usable = loco_mean_z_scores(h1_expr, h1_meta["cohort"], selected_genes)
    scored = h1_meta[["participant_id", "GSM", "cohort", "y_true"]].copy()
    scored["method"] = method
    scored["endpoint"] = "H1"
    scored["score"] = scores
    scored["n_degs_selected"] = len(selected_genes)
    scored["n_degs_bulk_usable"] = usable
    h1_programme_rows.append(scored)
h1_programmes = pd.concat(h1_programme_rows, ignore_index=True)
h1_programmes.to_csv(ROOT / "H1_E_ALL_R1_PROGRAMME_SCORES.tsv.gz", sep="\t", index=False, compression="gzip")

rows = []
for cohort in ["GSE79362", "GSE94438"]:
    for endpoint in ["H1", "H2", "H3"]:
        # Frozen native TB-STATE predictions.
        native = pred[(pred["outer_heldout"] == cohort) & (pred["endpoint"] == endpoint_map[endpoint])].copy()
        native = native.rename(columns={"participant_metric_weight": "weight"})
        native["weight"] = native["weight"].fillna(1.0)
        estimate = weighted_auc(native["y_true"], native["score"], native["weight"])
        lo, hi, valid = bootstrap_metric(native, "AUROC", stable_seed(cohort, endpoint, "TBSTATE"))
        rows.append(
            {
                "cohort": cohort,
                "endpoint": endpoint,
                "method": "TBSTATE",
                "estimate": estimate,
                "ci_low": lo,
                "ci_high": hi,
                "n_participants": native["participant_id"].nunique(),
                "n_rows": len(native),
                "events": native.groupby("participant_id")["y_true"].max().sum(),
                "bootstrap_valid": valid,
                "source_type": "FROZEN_NATIVE_RESULT3",
                "gene_rule": "NOT_APPLICABLE_NATIVE_MODEL",
                "status": "ESTIMATED",
            }
        )

        # Comparator R1 programmes, calculated independently within each cohort.
        for method in methods[1:]:
            # Structural endpoint gaps are retained only where the user has not requested the entire column removed.
            if (
                (endpoint == "H1" and method == "DEGAS")
                or (endpoint == "H2" and method == "SIDISH")
                or (endpoint == "H3" and method == "SIDISH")
            ):
                continue
            if endpoint == "H1":
                scored = h1_programmes[
                    (h1_programmes["cohort"] == cohort) & (h1_programmes["method"] == method)
                ].copy()
            else:
                scored = prog[
                    (prog["cohort"] == cohort)
                    & (prog["endpoint"] == endpoint)
                    & (prog["method"] == method)
                ].copy()
            if len(scored) == 0:
                rows.append(
                    {
                        "cohort": cohort,
                        "endpoint": endpoint,
                        "method": method,
                        "estimate": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n_participants": 0,
                        "n_rows": 0,
                        "events": 0,
                        "bootstrap_valid": 0,
                        "source_type": "R1_RISK_CELL_DEG_PROGRAMME",
                        "gene_rule": "BH_FDR_LT_0.05_AND_LOG2FC_GE_1.0",
                        "status": "STRUCTURAL_NE",
                    }
                )
                continue
            if endpoint != "H1":
                endpoint_labels = (
                    pred[
                        (pred["outer_heldout"] == cohort)
                        & (pred["endpoint"] == endpoint_map[endpoint])
                    ]
                    .groupby("participant_id", as_index=False)["y_true"]
                    .max()
                )
                scored = scored.drop(columns=["y_true"]).merge(
                    endpoint_labels,
                    on="participant_id",
                    how="left",
                    validate="one_to_one",
                )
            scored["weight"] = 1.0
            estimate = weighted_auc(scored["y_true"], scored["score"])
            lo, hi, valid = bootstrap_metric(scored, "AUROC", stable_seed(cohort, endpoint, method))
            rows.append(
                {
                    "cohort": cohort,
                    "endpoint": endpoint,
                    "method": method,
                    "estimate": estimate,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_participants": scored["participant_id"].nunique(),
                    "n_rows": len(scored),
                    "events": scored["y_true"].sum(),
                    "bootstrap_valid": valid,
                    "source_type": "R1_RISK_CELL_DEG_PROGRAMME",
                    "gene_rule": "BH_FDR_LT_0.05_AND_LOG2FC_GE_1.0",
                    "status": "ESTIMATED",
                }
            )

results = pd.DataFrame(rows)
results.to_csv(ROOT / "COHORT_SEPARATED_METRICS.tsv", sep="\t", index=False)

# Reproduction gates for the native estimates.
expected = {
    ("GSE79362", "H1"): 0.6539365452408932,
    ("GSE94438", "H1"): 0.6200772200772200,
    ("GSE79362", "H2"): 0.8530546515919082,
    ("GSE94438", "H2"): 0.6774684562216393,
    ("GSE79362", "H3"): 0.7616909389970106,
    ("GSE94438", "H3"): 0.7026736140075319,
}
audit = []
for key, value in expected.items():
    cohort, endpoint = key
    observed = float(
        results[
            (results["cohort"] == cohort)
            & (results["endpoint"] == endpoint)
            & (results["method"] == "TBSTATE")
        ]["estimate"].iloc[0]
    )
    audit.append(
        {
            "cohort": cohort,
            "endpoint": endpoint,
            "expected": value,
            "observed": observed,
            "absolute_difference": abs(value - observed),
            "match": bool(np.isclose(value, observed, atol=1e-12, rtol=0)),
        }
    )
audit_df = pd.DataFrame(audit)
audit_df.to_csv(ROOT / "TBSTATE_RESULT3_REPRODUCTION_AUDIT.tsv", sep="\t", index=False)
if not audit_df["match"].all():
    raise RuntimeError("Frozen native Result 3 reproduction gate failed")
