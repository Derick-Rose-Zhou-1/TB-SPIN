from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_PROJECT_ROOT = Path.cwd() / "data"
PROJECT_ROOT = Path(os.environ.get("TB_STATE_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).expanduser().resolve()
V126_ROOT = Path(
    os.environ.get(
        "TB_STATE_V126_ROOT",
        str(PROJECT_ROOT / "TB_STATE_RESULT3_V126_ENDPOINT_SPECIFIC"),
    )
).expanduser().resolve()
CODE_ROOT = Path(
    os.environ.get(
        "TB_STATE_V126_CODE_ROOT",
        str(PROJECT_ROOT / "13_CODE" / "tb_state_v126_endpoint_specific"),
    )
).expanduser().resolve()
IMAGE_SUFFIXES = {".pdf", ".svg", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FORMAL_COHORTS = (
    "GSE79362",
    "GSE94438",
    "GSE107993_GSE107994_combined",
)
ENDPOINT_MAP = {"E_ALL": "H1", "E_6M": "H2", "E_12M": "H3"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path, block: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            payload = handle.read(block)
            if not payload:
                break
            digest.update(payload)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_tsv(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def read_current_run() -> Path:
    path = V126_ROOT / "CURRENT_RUN.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    run = Path(path.read_text(encoding="utf-8-sig").strip())
    if not run.exists():
        raise FileNotFoundError(run)
    return run


def create_run() -> Path:
    V126_ROOT.mkdir(parents=True, exist_ok=True)
    run = V126_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    subdirectories = (
        "00_AUDIT", "01_DATA", "02_ENDPOINTS", "03_FEATURE_SELECTION",
        "04_INNER_CV", "05_MODELS", "06_OUTER_EVALUATION", "07_SURVIVAL",
        "08_CELL_PROJECTION", "09_RISK_CELLS", "10_PSEUDOBULK_DEG",
        "11_HORIZON_COMPARISON", "12_FIGURES", "13_TABLES", "14_MANUSCRIPT",
        "15_REVIEW_PACKAGE", "logs",
    )
    for name in subdirectories:
        (run / name).mkdir(parents=True, exist_ok=True)
    atomic_text(V126_ROOT / "CURRENT_RUN.txt", str(run.resolve()) + "\n")
    atomic_text(run / "RUN_DIR.txt", str(run.resolve()) + "\n")
    atomic_text(
        run / "12_FIGURES" / "FIGURE_GENERATION_DISABLED.txt",
        "FIGURE_GENERATION_DISABLED=TRUE\nPLOTS_CREATED=FALSE\nUSER_OVERRIDE_BINDING=TRUE\n",
    )
    return run


def resolve_sources() -> dict[str, Path]:
    v1251 = PROJECT_ROOT / "19_SIDISH_ITERATIVE_REFINEMENT" / "28_TB_SIDISH_V1251_CLINICAL_REPAIR"
    v1252 = PROJECT_ROOT / "19_SIDISH_ITERATIVE_REFINEMENT" / "29_TB_SIDISH_V1252_TB_FREE_SURVIVAL"
    six = PROJECT_ROOT / "20_TOP_JOURNAL_EXTENSION" / "16_RESULT3_CELLULAR_INTERPRETATION_V1253_CORRECTED"
    paths = {
        "bulk_expression": PROJECT_ROOT / "01_SHARED_BULK_LOG2CPM.joblib",
        "gene_list": PROJECT_ROOT / "01_SHARED_GENE_LIST.tsv",
        "sc_encoder": PROJECT_ROOT / "02_SC_ENCODER_CHECKPOINT.pt",
        "h5ad": PROJECT_ROOT / "00_INPUTS_LOCKED" / "single_cell" / "Tian_PBMC_16986genes_HVG3000_res2_QCclean_fullgenes.h5ad",
        "sc_latent": PROJECT_ROOT / "02_SC_CELL_LATENT.tsv",
        "endpoint_H1": PROJECT_ROOT / "02_ENDPOINTS" / "H1" / "H1_endpoint.tsv",
        "endpoint_H2": PROJECT_ROOT / "02_ENDPOINTS" / "H2" / "H2_endpoint.tsv",
        "endpoint_H3": PROJECT_ROOT / "02_ENDPOINTS" / "H3" / "H3_endpoint.tsv",
        "survival": v1252 / "V1252_PARTICIPANT_SURVIVAL_TABLE.tsv",
        "v125_metrics": v1251 / "04_FINAL_OUTER_EVALUATION_REPAIRED" / "FINAL_OUTER_METRICS_BY_COHORT_ENDPOINT.tsv",
        "v125_cox": v1252 / "V1252_COX_HR_BY_COHORT.tsv",
        "v125_km": v1252 / "V1252_KM_RISK_GROUPS.tsv",
        "v125_nested_manifest": v1251 / "V1251_NESTED_SPLIT_MANIFEST.tsv",
        "v125_configuration_lock": v1251 / "03_LOCKED_OUTER" / "FINAL_CONFIGURATION_LOCK.json",
        "six_lineage_atlas": six / "01_RISK_LABEL" / "MASTER_CELL_UNIVERSE.tsv.gz",
    }
    for cohort in FORMAL_COHORTS:
        directory = "heldout_GSE107993_GSE107994" if cohort.endswith("combined") else f"heldout_{cohort}"
        paths[f"v125_predictions_{cohort}"] = v1251 / "03_LOCKED_OUTER" / directory / "UNLABELED_OUTER_PREDICTIONS.tsv"
        paths[f"v125_model_{cohort}"] = v1251 / "03_LOCKED_OUTER" / directory / "LOCKED_FINAL_MODEL.pt"
    missing = {key: str(path) for key, path in paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError("Missing canonical sources: " + json.dumps(missing, indent=2))
    return paths


def assert_no_figures(run: Path) -> None:
    found = [str(path) for path in run.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if found:
        raise RuntimeError("NO_FIGURE_RULE_VIOLATION:" + "|".join(found))
