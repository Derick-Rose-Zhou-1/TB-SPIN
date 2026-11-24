#!/usr/bin/env python3
"""Freeze final full-model Fisher and percentile risk-layer parameters."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("TBSPIN_R2_RUN", "results2_run"))
SEEDS = (20260828, 20260829, 20260830)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    layers: dict[str, object] = {}
    selection_frames = []
    for seed in SEEDS:
        folder = ROOT / f"17_FINAL_FULL_MODEL/seed_{seed}"
        if not (folder / "RISK_DONE.json").exists():
            raise RuntimeError(f"missing final risk completion for seed {seed}")
        directions = pd.read_csv(folder / "fisher_directions.tsv", sep="\t")
        models = json.loads((folder / "fitted_risk_models.json").read_text(encoding="utf-8"))
        seed_layer: dict[str, object] = {}
        for _, row in directions.iterrows():
            key = f"{row.task}|{row.celltype_raw}"
            if row.status != "PASS" or key not in models:
                seed_layer[key] = {"status": "LOW_SUPPORT"}
                continue
            seed_layer[key] = {
                "status": "PASS", "task": row.task, "celltype_raw": row.celltype_raw,
                "reference_stage": row.reference_stage, "target_stage": row.target_stage,
                "w": json.loads(row.w_json), "intercept_b": float(row.intercept_b),
                "reference_median_c": float(row.reference_median_c),
                "covariance_method": "LEDOIT_WOLF", "coordinate_method": "FISHER_LDA",
                "risk_transform": "EXP_SHIFTED_DISCRIMINANT",
                **models[key],
            }
        layers[str(seed)] = seed_layer
        selection_frames.append(pd.read_csv(folder / "final_distribution_selection.tsv", sep="\t"))
    payload = {
        "status": "FROZEN", "seeds": list(SEEDS), "annotation_column": "celltype_raw",
        "HVG_primary": 5000, "primary_pcut": 0.95, "sensitivity_pcut": [0.90, 0.975],
        "cell_level_BH": False, "formal_inference_unit": "PARTICIPANT",
        "distribution_candidates": ["GAMMA", "LOGNORMAL", "WEIBULL", "EXPONENTIAL"],
        "distribution_selection": "BOOTSTRAP_KS_THEN_BIC",
        "reference_balancing": "EQUAL_CELL_PER_PARTICIPANT_RESAMPLING",
        "layers": layers,
    }
    target = ROOT / "17_FINAL_FULL_MODEL/FINAL_RISK_LAYER.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    checksum = sha256(target)
    (ROOT / "17_FINAL_FULL_MODEL/FINAL_RISK_LAYER.sha256").write_text(f"{checksum}  FINAL_RISK_LAYER.json\n", encoding="ascii")
    pd.concat(selection_frames, ignore_index=True).to_csv(
        ROOT / "17_FINAL_FULL_MODEL/FINAL_DISTRIBUTION_SELECTION.tsv", sep="\t", index=False
    )
    (ROOT / "17_FINAL_FULL_MODEL/DONE.json").write_text(
        json.dumps({"status": "PASS", "seed_n": 3, "risk_layer_sha256": checksum}, indent=2), encoding="utf-8"
    )
    print(f"V3_FINAL_RISK_LAYER=PASS sha256={checksum}", flush=True)


if __name__ == "__main__":
    main()
