import hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

def set_archive(work):
    global SOURCE, GENE_LEVEL
    SOURCE = work.parent / "RESULT2_SIGNATURE_SCORE_CROSS_COHORT_20260906"
    GENE_LEVEL = SOURCE / "data/gene_level"

SEED, N_BOOT = 20260907, 2000
FSQN_SOURCE = "https://doi.org/10.1093/bioinformatics/bty026"
QN_SOURCE = "https://doi.org/10.1038/s42003-023-04588-6"

DATASETS = ["GSE101705", "GSE193777", "GSE107993", "GSE107994", "GSE19435", "GSE19439", "GSE29536", "GSE83456", "GSE89403", "LOCAL_QZ"]
TECH = {
    "GSE101705": ("RNA-seq", "RNA-seq"), "GSE193777": ("RNA-seq", "RNA-seq"), "GSE107993": ("RNA-seq", "RNA-seq"),
    "GSE107994": ("RNA-seq", "RNA-seq"), "GSE89403": ("RNA-seq", "RNA-seq"), "LOCAL_QZ": ("RNA-seq", "RNA-seq"),
    "GSE19435": ("Microarray;GPL6947", "Microarray"), "GSE19439": ("Microarray;GPL6947", "Microarray"),
    "GSE29536": ("Microarray;GPL6102", "Microarray"), "GSE83456": ("Microarray;GPL10558", "Microarray"),
}
TASKS = {
    "T1": {"negative": "HC", "positive": "LTBI", "label": "HC vs M. tuberculosis infection"},
    "T2": {"negative": "LTBI", "positive": "ATB", "label": "M. tuberculosis infection vs ATB"},
}
def stable_seed(*parts):
    return (SEED + int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)) % (2**32 - 1)


def ngene(x):
    s = "" if pd.isna(x) else str(x).strip().upper()
    return "" if s in {"", "NA", "NAN", "NONE"} else s.split(".", 1)[0] if s.startswith("ENSG") else s


def metadata(ds):
    m = pd.read_csv(GENE_LEVEL / f"{ds}.metadata.tsv", sep="\t", dtype=str)
    return m[["dataset", "sample_id", "clinical_state", "participant_id"]].copy()


def all_panels():
    f = SOURCE / "data" / "FROZEN_SIGNATURES.tsv"
    t = pd.read_csv(f, sep="\t")
    return {task: list(dict.fromkeys(t.loc[t.task.eq(task), "gene"].map(ngene))) for task in TASKS}


def fsqn_map(target, reference):
    """Published FSQN equivalent: gene-wise quantile mapping to reference."""
    out = np.empty_like(target, dtype=np.float32); n = target.shape[0]
    q = (np.arange(reference.shape[0]) + .5) / reference.shape[0]
    for j in range(target.shape[1]):
        vals = target[:, j]; ref = np.sort(reference[:, j])
        ranks = rankdata(vals, method="average"); pos = (ranks - .5) / n
        out[:, j] = np.interp(pos, q, ref, left=ref[0], right=ref[-1])
    return out


def qn_map(target, reference):
    """Foltz et al. target-distribution QN: sample-wise RNA-seq mapping."""
    ref_target = np.mean(np.sort(reference, axis=1), axis=0)
    out = np.empty_like(target, dtype=np.float32); n = target.shape[1]; q = (np.arange(n) + .5) / n
    for i in range(target.shape[0]):
        ranks = rankdata(target[i], method="average"); pos = (ranks - .5) / n
        out[i] = np.interp(pos, q, ref_target, left=ref_target[0], right=ref_target[-1])
    return out


def participant_auc(frame, score_col="score"):
    p = frame.groupby("participant_id", as_index=False).agg(binary_label=("binary_label", "first"), score=(score_col, "mean"))
    y, s = p.binary_label.to_numpy(), p.score.to_numpy()
    if len(set(y)) < 2: return (np.nan, np.nan, np.nan, 0, 0)
    rng = np.random.default_rng(stable_seed("boot", tuple(sorted(frame.dataset.unique())), len(p), float(np.nansum(s))))
    neg, pos = np.flatnonzero(y == 0), np.flatnonzero(y == 1); boots=[]
    for _ in range(N_BOOT):
        ix=np.r_[rng.choice(neg,len(neg),replace=True),rng.choice(pos,len(pos),replace=True)]; boots.append(roc_auc_score(y[ix],s[ix]))
    return float(roc_auc_score(y,s)), float(np.quantile(boots,.025)), float(np.quantile(boots,.975)), len(neg), len(pos)
