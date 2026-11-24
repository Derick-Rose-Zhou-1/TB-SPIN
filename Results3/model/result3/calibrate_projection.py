"""Join fold scores to the six-lineage atlas and centre Cox scores for the final cell lock."""
import argparse
import numpy as np
import pandas as pd

SEED = 20260816
LINEAGES = {"Monocyte", "NK cell", "T cell", "B cell", "pDC", "Neutrophil"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", help="Parquet produced by project_cells.py")
    parser.add_argument("atlas", help="Formal atlas metadata (TSV or release Parquet)")
    parser.add_argument("output", help="CELL_FOLDWISE_SCORES.parquet")
    args = parser.parse_args()
    cols = ["cell_id", "participant_id", "dataset_id", "disease_group", "lineage", "subtype"]
    if args.atlas.endswith(".parquet"):
        atlas = pd.read_parquet(args.atlas, columns=cols)
    else:
        atlas = pd.read_csv(args.atlas, sep="\t", usecols=cols)
    atlas = atlas.loc[atlas.lineage.isin(LINEAGES)]
    scores = atlas.merge(pd.read_parquet(args.scores), on="cell_id", how="left", validate="1:1")
    if len(scores) != 214402 or scores.filter(regex="^E_").isna().any().any():
        raise ValueError("Incomplete formal atlas or missing fold scores")
    reference = scores.loc[scores.disease_group.eq("LTBI")]
    sizes = reference.groupby("participant_id").size()
    if (sizes >= 500).mean() < .8:
        raise ValueError("Frozen LTBI participant coverage differs from the manuscript")
    rng = np.random.default_rng(SEED)
    rows = []
    for _, group in reference.groupby("participant_id", sort=True):
        rows.extend(group.index if len(group) <= 500 else rng.choice(group.index, 500, replace=False))
    for fold in range(1, 4):
        values = scores[f"E_COX_fold{fold}"]
        scores[f"cox_RR_m{fold}"] = np.exp(values - values.loc[rows].median())
    scores.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
