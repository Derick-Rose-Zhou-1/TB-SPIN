from __future__ import annotations
import numpy as np
import scipy.sparse as sp

def scaling_stats(expression):
    x=sp.csc_matrix(expression,dtype=np.float64);n=x.shape[1]
    means=np.asarray(x.mean(axis=1)).ravel();sumsq=np.asarray(x.multiply(x).sum(axis=1)).ravel()
    var=(sumsq-n*means*means)/(n-1);var=np.maximum(var,0);sd=np.sqrt(var)
    return means,sd

def score_cells(expression, coefficients, means=None, sd=None, chunk_cells=8192, scale_max=10.0):
    """Bounded-memory algebraic equivalent of FastSparseRowScale + crossprod."""
    x=sp.csc_matrix(expression,dtype=np.float64);b=np.asarray(coefficients,dtype=np.float64)
    if means is None or sd is None: means,sd=scaling_stats(x)
    valid=np.isfinite(sd)&(sd!=0);out=np.empty(x.shape[1],dtype=np.float64)
    zero_z=np.zeros_like(means)
    zero_z[valid]=np.minimum(-means[valid]/sd[valid],scale_max)
    baseline=float(np.dot(b,zero_z))
    for lo in range(0,x.shape[1],chunk_cells):
        hi=min(x.shape[1],lo+chunk_cells)
        # Exact FastSparseRowScale semantics include its default upper cap of 10.
        # Start from the standardized zero baseline and apply only sparse nonzero
        # deltas, avoiding a dense genes x cells or genes x chunk matrix.
        block=x[:,lo:hi].tocoo()
        keep=valid[block.row]
        z=np.zeros_like(block.data)
        z[keep]=np.minimum((block.data[keep]-means[block.row[keep]])/sd[block.row[keep]],scale_max)
        delta=b[block.row]*(z-zero_z[block.row])
        out[lo:hi]=baseline+np.bincount(block.col,weights=delta,minlength=hi-lo)
    return out,means,sd
