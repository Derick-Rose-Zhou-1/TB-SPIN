"""Create a read-only compatibility view for the historical inner selector.

The selector code is imported directly from FINAL_RELEASE but all inputs and
all outputs are staged below the de-novo run.  The historical tree is never
modified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


COHORTS = ("GSE79362", "GSE94438", "GSE107993_GSE107994_combined")


def link(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source:
            raise RuntimeError(f"COMPAT_LINK_TARGET_CONFLICT:{destination}")
        return
    if destination.exists():
        raise RuntimeError(f"COMPAT_PATH_ALREADY_EXISTS:{destination}")
    destination.symlink_to(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--alignment-root", required=True, type=Path)
    parser.add_argument("--compat-root", required=True, type=Path)
    parser.add_argument("--endpoint-run", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    project = args.compat_root / "PROJECT_ROOT"
    v126_root = args.compat_root / "V126_ROOT"
    args.endpoint_run.mkdir(parents=True, exist_ok=True)
    for directory in ("02_ENDPOINTS", "03_FEATURE_SELECTION", "04_INNER_CV", "05_MODELS"):
        (args.endpoint_run / directory).mkdir(exist_ok=True)
    link(Path(config["endpoint_table"]), args.endpoint_run / "02_ENDPOINTS" / "PARTICIPANT_ENDPOINTS.tsv")
    (v126_root / "CURRENT_RUN.txt").parent.mkdir(parents=True, exist_ok=True)
    (v126_root / "CURRENT_RUN.txt").write_text(str(args.endpoint_run.resolve()) + "\n", encoding="utf-8")

    link(Path(config["bulk_expression"]), project / "01_SHARED_BULK_LOG2CPM.joblib")
    link(Path(config["frozen_gene_list"]), project / "01_SHARED_GENE_LIST.tsv")
    link(Path(config["h5ad"]), project / "00_INPUTS_LOCKED" / "single_cell" / Path(config["h5ad"]).name)
    link(Path(config["formal_cell_manifest"]), project / "02_SC_CELL_LATENT.tsv")
    link(Path(config["endpoint_table"]), project / "02_SC_ENCODER_CHECKPOINT.pt")
    for endpoint in ("H1", "H2", "H3"):
        link(Path(config["endpoint_table"]), project / "02_ENDPOINTS" / endpoint / f"{endpoint}_endpoint.tsv")
    unused = (
        project / "19_SIDISH_ITERATIVE_REFINEMENT" / "28_TB_SIDISH_V1251_CLINICAL_REPAIR" / "04_FINAL_OUTER_EVALUATION_REPAIRED" / "FINAL_OUTER_METRICS_BY_COHORT_ENDPOINT.tsv",
        project / "19_SIDISH_ITERATIVE_REFINEMENT" / "29_TB_SIDISH_V1252_TB_FREE_SURVIVAL" / "V1252_PARTICIPANT_SURVIVAL_TABLE.tsv",
        project / "19_SIDISH_ITERATIVE_REFINEMENT" / "29_TB_SIDISH_V1252_TB_FREE_SURVIVAL" / "V1252_COX_HR_BY_COHORT.tsv",
        project / "19_SIDISH_ITERATIVE_REFINEMENT" / "29_TB_SIDISH_V1252_TB_FREE_SURVIVAL" / "V1252_KM_RISK_GROUPS.tsv",
        project / "19_SIDISH_ITERATIVE_REFINEMENT" / "28_TB_SIDISH_V1251_CLINICAL_REPAIR" / "V1251_NESTED_SPLIT_MANIFEST.tsv",
        project / "19_SIDISH_ITERATIVE_REFINEMENT" / "28_TB_SIDISH_V1251_CLINICAL_REPAIR" / "03_LOCKED_OUTER" / "FINAL_CONFIGURATION_LOCK.json",
        project / "20_TOP_JOURNAL_EXTENSION" / "16_RESULT3_CELLULAR_INTERPRETATION_V1253_CORRECTED" / "01_RISK_LABEL" / "MASTER_CELL_UNIVERSE.tsv.gz",
    )
    for destination in unused:
        link(Path(config["endpoint_table"]), destination)
    canonical = Path(config["source_project"]) / "FINAL_RELEASE" / "CODE" / "V126_PRODUCTION" / "modelling" / "canonical" / "tbstate_v12_models.py"
    link(canonical, project / "13_CODE" / "modelling" / "canonical" / "tbstate_v12_models.py")

    architecture = {"n_genes": 16930, "hidden_dim": config["single_cell"]["hidden_dim"], "latent_dim": config["single_cell"]["latent_dim"], "dropout": config["single_cell"]["dropout"]}
    for cohort in COHORTS:
        source = args.alignment_root / f"heldout_{cohort}" / "best_alignment.pt"
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if payload.get("heldout_cohort") != cohort or payload.get("heldout_expression_used") or payload.get("heldout_outcomes_used"):
            raise RuntimeError(f"ALIGNMENT_PROVENANCE_GATE_FAILED:{cohort}")
        name = "heldout_GSE107993_GSE107994" if cohort.endswith("combined") else f"heldout_{cohort}"
        target = project / "19_SIDISH_ITERATIVE_REFINEMENT" / "28_TB_SIDISH_V1251_CLINICAL_REPAIR" / "03_LOCKED_OUTER" / name / "LOCKED_FINAL_MODEL.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"alignment_config": architecture, "alignment_state": payload["model_state"], "compatibility_source": str(source), "heldout_cohort": cohort}, target)
        link(Path(config["endpoint_table"]), target.parent / "UNLABELED_OUTER_PREDICTIONS.tsv")
    print(json.dumps({"status": "PASS", "compat_root": str(args.compat_root), "endpoint_run": str(args.endpoint_run), "cohorts": list(COHORTS)}, indent=2))


if __name__ == "__main__":
    main()
