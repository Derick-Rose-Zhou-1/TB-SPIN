from __future__ import annotations
import numpy as np

def _rank_average_r(values: np.ndarray) -> np.ndarray:
    """R rank(ties.method='average') in double precision, 1-based ranks."""
    order = np.argsort(values, kind="mergesort")
    sv = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sv[j] == sv[i]: j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks

def quantile_normalize(x):
    """Match preprocessCore::normalize.quantiles including averaged-rank tie handling."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2: raise ValueError("x must be 2D")
    target = np.mean(np.sort(a, axis=0), axis=1)
    out = np.empty_like(a)
    n = a.shape[0]
    for j in range(a.shape[1]):
        r = _rank_average_r(a[:, j])
        lo = np.floor(r).astype(np.int64)
        frac = r - lo
        vals = np.empty(n, dtype=np.float64)
        integer = frac == 0
        vals[integer] = target[lo[integer]-1]
        ni = ~integer
        # preprocessCore maps the average rank: integer ranks select one target
        # value and half-integer ranks interpolate the two adjacent target values.
        if np.any(ni):
            col=a[:,j]
            for v in np.unique(col[ni]):
                ix=np.flatnonzero(col==v); rr=r[ix[0]]
                flo=int(np.floor(rr)); cei=int(np.ceil(rr))
                vals[ix]=(target[flo-1]+target[cei-1])/2.0
        out[:,j]=vals
    return out
