"""Compute Supplementary Table 6 AUCs from the frozen seed-11 predictions."""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def summarize(predictions):
    if (len(predictions) != 283 or predictions.participant_id.nunique() != 283
            or predictions.outer_fold.nunique() != 5
            or not predictions.fold_source.eq("FROZEN_TBSTATE_REPEAT_SEED_11").all()):
        raise ValueError("Expected one seed-11 out-of-fold prediction per participant")
    frame = predictions.reset_index(drop=True)
    strata = list(frame.groupby(["phenotype", "source_cohort"], sort=True).indices.values())
    if sum(map(len, strata)) != len(frame):
        raise ValueError("Missing phenotype or source")
    y = frame.phenotype.to_numpy()
    scores = frame[["TB_STATE", "DEG_CENTROID", "DEG_ELASTIC_NET"]]
    rng = np.random.default_rng(20260824)
    draws = [np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
             for _ in range(2000)]
    rows = []
    for model, column in scores.items():
        values = column.to_numpy()
        low, high = np.percentile([roc_auc_score(y[i], values[i]) for i in draws], [2.5, 97.5])
        rows.append((model, roc_auc_score(y, values), low, high))
    return pd.DataFrame(rows, columns=["model", "AUROC", "AUROC_CI_low", "AUROC_CI_high"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="OOF_PREDICTIONS_ALL_MODELS.tsv.gz")
    parser.add_argument("output")
    args = parser.parse_args()
    summarize(pd.read_csv(args.predictions, sep="\t")).to_csv(args.output, sep="\t", index=False)
