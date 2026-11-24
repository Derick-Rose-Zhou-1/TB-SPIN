"""Validate the disease signature by leaving out each of the eight cohorts in turn."""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

import fsqn_core as core

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('archive', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
work = args.archive
core.set_archive(work)

archive = work / "NONATB_ATB_ALL_COHORTS_20260907"
original = pd.read_csv(archive / "tables/NONATB_ATB_FSQN_CROSS_PLATFORM_AUC.tsv", sep="\t")
cohorts = original.loc[original.normalization.eq("FSQN"), "dataset"].tolist()
panel = core.all_panels()["T2"]
assert len(cohorts) == 9 and len(set(panel)) == len(panel) == 18


def load_expression(dataset):
    data = pd.read_csv(core.GENE_LEVEL / f"{dataset}.expression.tsv.gz", sep="\t")
    gene = data.columns[0]
    data[gene] = data[gene].map(core.ngene)
    return data.groupby(gene, sort=False).mean(numeric_only=True).T.astype("float32")


expression = {dataset: load_expression(dataset) for dataset in cohorts}
metadata = {dataset: core.metadata(dataset).set_index("sample_id") for dataset in cohorts}


def run(included):
    shared = sorted(set.intersection(*(set(expression[d].columns) for d in included)) - {""})
    assert set(panel) <= set(shared)
    matrix = {d: expression[d][shared] for d in included}
    results = []
    for method in ("FSQN", "QN"):
        for holdout in included:
            train = [d for d in included if d != holdout]
            refs = [d for d in train if core.TECH[d][1] == "Microarray"]
            # FSQN acts independently per gene; sample-wise QN needs the whole transcriptome.
            columns = panel if method == "FSQN" else shared
            reference = pd.concat([matrix[d][columns] for d in refs]).to_numpy(dtype=np.float32)
            features, labels = {}, {}
            for d in train + [holdout]:
                x = matrix[d][columns]
                if core.TECH[d][1] != "Microarray":
                    transform = core.fsqn_map if method == "FSQN" else core.qn_map
                    x = pd.DataFrame(transform(x.to_numpy(), reference), index=x.index, columns=columns)
                ids = [s for s in metadata[d].index if s in x.index]
                features[d] = x.loc[ids, panel]
                labels[d] = metadata[d].loc[ids].copy()
                labels[d]["binary_label"] = labels[d].clinical_state.ne("ATB").map({True: 0, False: 1})
            scaler = StandardScaler()
            x_train = scaler.fit_transform(pd.concat([features[d] for d in train]).to_numpy())
            y_train = pd.concat([labels[d] for d in train]).binary_label.to_numpy()
            model = LogisticRegression(C=1., class_weight="balanced", max_iter=2000, random_state=core.SEED)
            model.fit(x_train, y_train)
            test = labels[holdout]
            test["score"] = model.predict_proba(scaler.transform(features[holdout].to_numpy()))[:, 1]
            a, lo, hi, neg, pos = core.participant_auc(test)
            if method == "FSQN":
                unit = test.groupby("participant_id").agg(y=("binary_label","first"), score=("score","mean"))
                fpr, tpr, _ = roc_curve(unit.y, unit.score)
                curves.append(pd.DataFrame({"dataset":holdout,"fpr":fpr,"tpr":tpr}))
            results.append(dict(dataset=holdout, normalization=method, auc=a, ci_low=lo, ci_high=hi,
                                n_control=neg, n_TB=pos, common_genes=len(shared),
                                training_cohorts=";".join(train), reference_microarrays=";".join(refs)))
    return pd.DataFrame(results)


output = args.output
output.mkdir(parents=True, exist_ok=True)
curves = []
reduced = run([d for d in cohorts if d != "GSE83456"])
reduced.to_csv(output / "cohort_auc.tsv", sep="\t", index=False)
pd.concat(curves, ignore_index=True).to_csv(output / "roc_coordinates.tsv", sep="\t", index=False)
