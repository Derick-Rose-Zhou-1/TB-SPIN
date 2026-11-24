"""Calculate the disease-stage contrast and verify the frozen Fig. 2 gene set."""
import pandas as pd
from recompute import BASE, MASTER, OUT, differential, load_infection, ora, signature


def main():
    labels = pd.read_csv(MASTER, sep="\t", usecols=["cell_id", "task", "analysis_stage", "R_ensemble"])
    disease = labels.loc[labels.task.eq("T2") & labels.analysis_stage.eq("ATB")]
    scores = disease.set_index("cell_id", verify_integrity=True).R_ensemble.dropna()
    counts, cells, genes = load_infection(scores)
    result = differential(counts, cells, genes, scores)
    selected = signature(result)
    archived = pd.read_csv(BASE / "volcano/ATB_SOURCE_ALL_TESTABLE_DEG.tsv.gz", sep="\t")
    if set(selected.gene) != set(signature(archived).gene):
        raise ValueError("Disease signature differs from the frozen analysis")
    result.to_csv(OUT / "T2_disease_DE.tsv.gz", sep="\t", index=False)
    selected.to_csv(OUT / "T2_disease_up.tsv", sep="\t", index=False)
    ora(result).to_csv(OUT / "T2_disease_GO_BP.tsv", sep="\t", index=False)
    print(f"{len(result)} genes tested, {len(selected)} risk-up genes")


if __name__ == "__main__":
    main()
