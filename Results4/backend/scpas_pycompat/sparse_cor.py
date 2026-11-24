from __future__ import annotations
import numpy as np
import scipy.sparse as sp

def sparse_cor(x):
    """Literal semantic port of uploaded R/scPAS.R sparse.cor(x)."""
    x = sp.csc_matrix(x, dtype=np.float64)
    n, m = x.shape
    ii = np.unique(x.indices)
    ex = np.asarray(x.mean(axis=0)).ravel()
    dense_rows = x[ii, :].toarray()
    nozero = dense_rows - ex[None, :]
    covmat = (nozero.T @ nozero + np.outer(ex, ex) * (n-len(ii))) / (n-1)
    sdvec = np.sqrt(np.diag(covmat))
    with np.errstate(invalid="ignore", divide="ignore"):
        return covmat / np.outer(sdvec, sdvec)
