"""Native binomial scPAS-compatible API and checkpointable model container."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .apml0 import apml0
from .genes import common_genes
from .network import build_network_from_correlation
from .normalize import quantile_normalize
from .permutation import permutation_statistics
from .sparse_cor import sparse_cor


DEFAULT_ALPHA = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


@dataclass
class ScPASModel:
    common_genes: np.ndarray
    alpha: np.ndarray
    lambda_: np.ndarray
    family: str
    coefficients: np.ndarray
    bulk: np.ndarray
    phenotype: np.ndarray
    network: np.ndarray

    def save_checkpoint(self, directory: str | Path) -> None:
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "MODEL_CHECKPOINT.npz", common_genes=self.common_genes,
                            alpha=self.alpha, lambda_=self.lambda_, family=self.family,
                            coefficients=self.coefficients, bulk=self.bulk,
                            phenotype=self.phenotype, network=self.network)
        pd.DataFrame({"gene": self.common_genes, "coefficient": self.coefficients,
                      "sign": np.sign(self.coefficients).astype(int)}).to_csv(
                          directory / "COEFFICIENTS.tsv", sep="\t", index=False)
        pd.DataFrame({"alpha": self.alpha, "lambda": self.lambda_}).to_csv(
            directory / "ALPHA_LAMBDA_TRAJECTORY.tsv", sep="\t", index=False)
        digest = hashlib.sha256(np.asarray(self.network, dtype=np.float64).tobytes(order="C")).hexdigest()
        (directory / "NETWORK_CHECKSUM").write_text(digest + "\n", encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, file: str | Path) -> "ScPASModel":
        value = np.load(file, allow_pickle=True)
        family = value["family"].item() if np.asarray(value["family"]).ndim == 0 else str(value["family"])
        return cls(value["common_genes"], value["alpha"], value["lambda_"],
                   family, value["coefficients"], value["bulk"],
                   value["phenotype"], value["network"])


@dataclass
class ScPASResult:
    table: pd.DataFrame
    model: ScPASModel


def fit_binomial(bulk: np.ndarray, bulk_genes, cells: sp.spmatrix, cell_genes,
                 phenotype: np.ndarray, alpha=None, network_class: str = "SC",
                 checkpoint_dir: str | Path | None = None) -> ScPASModel:
    common = np.asarray(common_genes(list(bulk_genes), list(cell_genes)), dtype=object)
    if not len(common): raise ValueError("There is no common genes between the given single-cell and bulk samples.")
    b_index = {g:i for i,g in enumerate(bulk_genes)}; c_index = {g:i for i,g in enumerate(cell_genes)}
    bulk_qn = quantile_normalize(np.asarray(bulk, dtype=float))
    expression_bulk = bulk_qn[[b_index[g] for g in common]]
    expression_cell = sp.csc_matrix(cells)[[c_index[g] for g in common]]
    x = expression_bulk.T
    cor = np.corrcoef(x, rowvar=False) if network_class == "bulk" else sparse_cor(expression_cell.T)
    network, _, _, _ = build_network_from_correlation(cor)
    alpha_values = DEFAULT_ALPHA.copy() if alpha is None else np.atleast_1d(np.asarray(alpha, dtype=float))
    lambdas=[]; coefficients=None
    for value in alpha_values:
        fit0=apml0(x,phenotype,Omega=network,alpha=float(value),nlambda=100,
                   nfolds=min(10,len(x)),iL0=True,icutB=False,cv_seed=123)
        fit1=apml0(x,phenotype,Omega=network,alpha=float(value),
                   lambda_=np.array([fit0.lambda_min]),iL0=True,icutB=False)
        lambdas.append(fit0.lambda_min); coefficients=np.asarray(fit1.Beta)[1:,0]
    model=ScPASModel(common,alpha_values,np.asarray(lambdas),"binomial",coefficients,x,
                     np.asarray(phenotype,dtype=float),network)
    if checkpoint_dir is not None: model.save_checkpoint(checkpoint_dir)
    return model


def score(model: ScPASModel, cells: sp.spmatrix, cell_genes, cell_ids=None,
          permutation_times: int = 2000, independent: bool = True,
          fdr_threshold: float = 0.05, checkpoint_dir: str | Path | None = None) -> ScPASResult:
    index={g:i for i,g in enumerate(cell_genes)}
    missing=[g for g in model.common_genes if g not in index]
    if missing: raise ValueError(f"missing model genes: {len(missing)}")
    expression=sp.csc_matrix(cells)[[index[g] for g in model.common_genes]]
    output=permutation_statistics(expression,model.coefficients,permutation_times,independent,fdr_threshold)
    ids=np.asarray(cell_ids if cell_ids is not None else [str(i) for i in range(expression.shape[1])])
    table=pd.DataFrame({"sample":ids,"scPAS_RS":output["scPAS_RS"],
                        "scPAS_NRS":output["scPAS_NRS"],"scPAS_Pvalue":output["scPAS_Pvalue"],
                        "scPAS_FDR":output["scPAS_FDR"],"scPAS":output["scPAS"]})
    if checkpoint_dir is not None:
        d=Path(checkpoint_dir);d.mkdir(parents=True,exist_ok=True)
        means=np.asarray(output["background_mean"]);sd=np.asarray(output["background_sd"])
        np.savez_compressed(d/"SCALING_STATS.npz",background_mean=means,background_sd=sd)
        (d/"PERMUTATION_PROGRESS").write_text(json.dumps({"completed":permutation_times})+"\n")
    return ScPASResult(table,model)
