"""Apply each available reconstructed endpoint head to the saved single-cell latent coordinates."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import torch
import v126_training as t

def load_model(folder: Path, model_name: str, endpoint: str):
    meta=json.loads((folder/"MODEL_ARTIFACT.json").read_text())
    kind=meta["artifact"]["type"]
    if kind == "sklearn": return {"kind":"sklearn","estimator":joblib.load(folder/"estimator.joblib"),"scaler":joblib.load(folder/"scaler.joblib"),"name":model_name}
    if kind == "cox":
        scaler=joblib.load(folder/"scaler.joblib"); p=torch.load(folder/"head.pt",map_location="cpu",weights_only=False); head=t.TorchHead(32,p["architecture"]); head.load_state_dict(p["state_dict"]); head.eval(); return {"kind":"cox","scaler":scaler,"head":head,"name":model_name}
    p=torch.load(folder/"model.pt",map_location="cpu",weights_only=False)
    if kind == "last_block_binary":
        head=torch.nn.Sequential(torch.nn.Linear(32,16),torch.nn.GELU(),torch.nn.Linear(16,1)); head.load_state_dict(p["head"]); head.eval(); return {"kind":"binary_head","head":head,"name":model_name}
    head=t.TorchHead(32,"B_LAST_BLOCK_COX16"); head.load_state_dict(p["head"]); head.eval(); return {"kind":"cox_head","head":head,"name":model_name}


def predict(model, z):
    if model["kind"]=="sklearn": return model["estimator"].predict_proba(model["scaler"].transform(z))[:,1]
    with torch.no_grad(): raw=model["head"](torch.from_numpy(z.astype(np.float32))).numpy()
    return 1/(1+np.exp(-raw)) if model["kind"]=="binary_head" else raw

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    archive = args.release / "PROVENANCE/RECONSTRUCTED_PROVENANCE_ARTIFACT"
    fits = pd.read_csv(archive / "03_REPRODUCTION/MODEL_REPRODUCTION.tsv", sep="\t")
    cells = pd.read_csv(args.release / "CELL_PROJECTION/02_SC_CELL_LATENT.tsv", sep="\t")
    assert cells.cell_id.is_unique
    latent = cells[[f"latent_{i:02d}" for i in range(1, 33)]].to_numpy(dtype=np.float32)
    output = cells[["cell_id"]].copy()
    for endpoint in ("E_6M", "E_12M", "E_COX"):
        rows = fits[(fits.endpoint == endpoint) & (fits.status == "PASS")]
        for fold, row in enumerate(rows.itertuples(index=False), 1):
            folder = archive / "02_RECONSTRUCTED_MODELS" / endpoint / row.outer_cohort
            model = load_model(folder, row.model, endpoint)
            output[f"{endpoint}_fold{fold}"] = predict(model, latent).reshape(-1)
    output.to_parquet(args.output, index=False)
