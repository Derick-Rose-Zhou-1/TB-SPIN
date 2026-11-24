"""Run locked endpoint configurations once on their untouched outer cohorts."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


TARGET_ENDPOINTS = ("E_COX", "E_6M", "E_12M")


def load_historical_training(source: Path):
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("historical_v126_training", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-source", required=True, type=Path)
    parser.add_argument("--compat-root", required=True, type=Path)
    parser.add_argument("--inner-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ["TB_STATE_PROJECT_ROOT"] = str(args.compat_root / "PROJECT_ROOT")
    os.environ["TB_STATE_V126_ROOT"] = str(args.compat_root / "V126_ROOT")
    os.environ["TB_STATE_V126_CODE_ROOT"] = str(args.historical_source.parent)
    module = load_historical_training(args.historical_source)
    locks = pd.read_csv(args.inner_run / "05_MODELS" / "FINAL_CONFIGURATION_LOCKS.tsv", sep="\t")
    bundle = module.load_bundle()
    rows: list[dict] = []
    metric_rows: list[dict] = []
    for lock in locks.itertuples(index=False):
        if lock.endpoint not in TARGET_ENDPOINTS or lock.status != "LOCKED":
            continue
        heldout = str(lock.outer_heldout)
        development = set(module.FORMAL_COHORTS) - {heldout}
        training = module.frame_for(bundle, lock.endpoint, development)
        outer = module.frame_for(bundle, lock.endpoint, {heldout})
        if outer.empty:
            metric_rows.append({"heldout_cohort": heldout, "endpoint": lock.endpoint, "status": "STRUCTURAL_NE_NO_OUTER_ENDPOINT", "outer_labels_read_after_lock": False})
            continue
        if training.empty or training["y_true"].nunique() < 2:
            metric_rows.append({"heldout_cohort": heldout, "endpoint": lock.endpoint, "status": "STRUCTURAL_NE_TRAINING_CLASSES", "outer_labels_read_after_lock": False})
            continue
        statistics = module.cox_gene_statistics(bundle, training) if lock.endpoint == "E_COX" else module.binary_gene_statistics(bundle, training, module.SEED)
        weights, effective_genes = module.gene_weights(statistics, lock.gene_condition, lock.endpoint)
        # The lock fixes the selection rule, not a mask learned on an inner
        # subset.  The final mask is intentionally recomputed using all
        # development participants, before outer labels are read.
        locked_effective_genes = int(lock.effective_genes)
        alignment = module.load_alignment(bundle, heldout)
        train_mean = bundle.bulk.loc[training["GSM"], bundle.genes].to_numpy(dtype=np.float32).mean(axis=0)
        train_latent = module.latent_features(bundle, alignment, training, weights, train_mean)
        outer_latent = module.latent_features(bundle, alignment, outer, weights, train_mean)
        if lock.endpoint == "E_COX":
            fit = module.fit_cox_latent(train_latent, training, lock.model)
            score = module.predict_cox(fit, outer_latent)
            metrics = module.cox_metrics(outer, score)
        else:
            scaler = StandardScaler().fit(train_latent)
            estimator = module.binary_estimator(lock.model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(scaler.transform(train_latent), training["y_true"], sample_weight=training["participant_metric_weight"])
            score = estimator.predict_proba(scaler.transform(outer_latent))[:, 1]
            metrics = module.binary_metrics(outer["y_true"], score, outer["participant_metric_weight"])
        metric_rows.append({"heldout_cohort": heldout, "endpoint": lock.endpoint, "gene_condition": lock.gene_condition, "model": lock.model, "locked_effective_genes": locked_effective_genes, "final_development_effective_genes": int(effective_genes), "n_development": int(len(training)), "n_outer": int(len(outer)), "outer_labels_read_after_lock": True, **metrics})
        for record, predicted in zip(outer.to_dict("records"), score):
            rows.append({"heldout_cohort": heldout, "endpoint": lock.endpoint, "GSM": record["GSM"], "participant_id": record["participant_id"], "y_true": int(record["y_true"]), "score": float(predicted), "gene_condition": lock.gene_condition, "model": lock.model})
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "OUTER_PREDICTIONS.tsv", sep="\t", index=False)
    pd.DataFrame(metric_rows).to_csv(args.output / "OUTER_METRICS.tsv", sep="\t", index=False)
    (args.output / "STATUS.json").write_text(json.dumps({"status": "PASS", "n_metrics": len(metric_rows), "outer_labels_read_after_lock": True}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
