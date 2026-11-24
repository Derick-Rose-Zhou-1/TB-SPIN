"""Re-score the frozen blood cohorts without changing the archived inputs."""
import gzip
import hashlib
import io
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUT = Path(os.environ.get("TBSPIN_FIG2_WORK", "figure2_work")) / "validation"
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(os.environ.get("TBSPIN_SIGNATURE_ARCHIVE", "inputs/signature_archive"))

def read(name):
    return pd.read_csv(ROOT / name, sep="\t", low_memory=False)

def score(values, genes, signature):
    valid = np.isfinite(values)
    selected = valid & genes.isin(signature)
    k = selected.sum()
    assert 0 < k < valid.sum()
    ranks = rankdata(values[valid], method="min")
    return (ranks[selected[valid]].mean() - (k + 1) / 2) / (valid.sum() - k) - .5


def auc(negative, positive):
    negative = np.sort(negative)
    return (np.searchsorted(negative, positive, side="left") +
            np.searchsorted(negative, positive, side="right")).mean() / (2 * len(negative))


def main():
    new = pd.read_csv(OUT.parent / "T1_infection_up.tsv", sep="\t").gene
    assert len(new) == new.nunique() == 107
    frozen = read("data/FROZEN_SIGNATURES.tsv")
    old = frozen.loc[frozen.task == "T1", "gene"]
    history = read("tables/COHORT_SIGNATURE_AUC.tsv").query("task == 'T1'").set_index("dataset")
    summary, scores = [], []
    for cohort in sorted(read("audit/BULK_COHORT_MANIFEST.tsv").dataset.unique()):
        stem = "data/gene_level/" + cohort
        meta = read(stem + ".metadata.tsv")
        meta = meta[meta.clinical_state.isin(["HC", "LTBI"])].copy()
        assert meta.sample_id.is_unique and meta.participant_id.notna().all()
        counts = meta.groupby("clinical_state").participant_id.nunique()
        row = dict(cohort=cohort, HC=counts.get("HC", 0), infection=counts.get("LTBI", 0))
        summary.append(row)
        if min(row["HC"], row["infection"]) == 0:
            row["status"] = "Missing comparison group"
            continue
        expr = read(stem + ".expression.tsv.gz").set_index("gene")
        assert expr.index.is_unique
        missing = sorted(set(new) - set(expr.index))
        row.update(genes=107-len(missing), missing=",".join(missing),
                   id_source=",".join(sorted(meta.participant_id_source.unique())))
        for label, signature in [("old", old), ("new", new)]:
            meta[label] = [score(expr[s].values, expr.index, signature) for s in meta.sample_id]
        assert meta.groupby("participant_id").clinical_state.nunique().max() == 1
        people = meta.groupby(["participant_id", "clinical_state"])[["old", "new"]].median().reset_index()
        neg = people.loc[people.clinical_state == "HC"]
        pos = people.loc[people.clinical_state == "LTBI"]
        assert np.isclose(auc(neg.old, pos.old), history.loc[cohort, "auc"], atol=1e-10, rtol=0)
        seed = (20260906 + int(hashlib.sha256((cohort+"|T1").encode()).hexdigest()[:8],16)) % (2**32-1)
        rng = np.random.default_rng(seed)
        boot = [auc(rng.choice(neg.new, len(neg)), rng.choice(pos.new, len(pos))) for _ in range(2000)]
        low, high = np.quantile(boot, [.025,.975])
        row.update(AUC=auc(neg.new, pos.new), CI_low=low, CI_high=high,
                   old_AUC_reproduced=auc(neg.old, pos.old), status="Primary" if row["genes"]/107 >= .8 else "Low coverage")
        scores.append(people.assign(cohort=cohort))
    assert {r["cohort"] for r in summary if "AUC" in r} == set(history.index)
    pd.DataFrame(summary).to_csv(OUT/"cohort_validation.tsv",sep="\t",index=False)
    pd.concat(scores).to_csv(OUT/"unit_scores.tsv",sep="\t",index=False)


if __name__ == "__main__":
    main()
