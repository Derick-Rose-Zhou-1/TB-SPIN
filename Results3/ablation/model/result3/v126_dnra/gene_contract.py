"""Strict case-only gene matching for the V126 de novo contract."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class GeneContractError(ValueError):
    """Raised when the H5AD and frozen gene universes are not case-equivalent."""


def _canonical(name: str) -> str:
    value = str(name)
    if value != value.strip() or not value:
        raise GeneContractError(f"INVALID_GENE_NAME:{value!r}")
    return value.upper()


@dataclass(frozen=True)
class GeneContract:
    h5ad_column_index: np.ndarray
    canonical_names: list[str]


def build_gene_contract(h5ad_names: list[str], frozen_names: list[str]) -> GeneContract:
    h5ad_canonical = [_canonical(name) for name in h5ad_names]
    frozen_canonical = [_canonical(name) for name in frozen_names]
    if len(set(h5ad_canonical)) != len(h5ad_canonical):
        raise GeneContractError("DUPLICATE_H5AD_CANONICAL")
    if len(set(frozen_canonical)) != len(frozen_canonical):
        raise GeneContractError("DUPLICATE_FROZEN_CANONICAL")
    h5ad_lookup = {name: index for index, name in enumerate(h5ad_canonical)}
    missing = [name for name in frozen_canonical if name not in h5ad_lookup]
    if missing:
        raise GeneContractError(f"NON_CASE_MATCH:{','.join(missing[:10])}")
    return GeneContract(
        h5ad_column_index=np.asarray([h5ad_lookup[name] for name in frozen_canonical], dtype=np.int64),
        canonical_names=frozen_canonical,
    )


def build_contract_audit(h5ad_names: list[str], frozen_names: list[str]) -> pd.DataFrame:
    contract = build_gene_contract(h5ad_names, frozen_names)
    return pd.DataFrame(
        {
            "frozen_gene_symbol": [str(name) for name in frozen_names],
            "canonical_gene_symbol": contract.canonical_names,
            "h5ad_gene_symbol": [str(h5ad_names[index]) for index in contract.h5ad_column_index],
            "h5ad_column_index": contract.h5ad_column_index,
        }
    )
