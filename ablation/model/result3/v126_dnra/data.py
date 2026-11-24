"""Sparse count-matrix operations for the immutable 16,930-gene contract."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def reorder_to_contract(matrix: csr_matrix, h5ad_column_index: np.ndarray) -> csr_matrix:
    index = np.asarray(h5ad_column_index, dtype=np.int64)
    if index.ndim != 1 or len(index) == 0:
        raise ValueError("INVALID_CONTRACT_INDEX")
    if index.min() < 0 or index.max() >= matrix.shape[1]:
        raise ValueError("CONTRACT_INDEX_OUT_OF_BOUNDS")
    return matrix[:, index].tocsr()
