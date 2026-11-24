from __future__ import annotations
import numpy as np
import scipy.sparse as sp

def _r_order(v):
    # R order is stable for ties in the reference fixtures; NaN sorts last.
    return np.argsort(v, kind="stable")

def compute_snn(nn_ranked, prune=1/15):
    """Seurat ComputeSNN semantics: shared-neighbor Jaccard, prune <= threshold."""
    nn=np.asarray(nn_ranked,dtype=np.int64);n,k=nn.shape
    rows=np.repeat(np.arange(n),k);cols=nn.ravel();knn=sp.csr_matrix((np.ones(n*k),(rows,cols)),shape=(n,n))
    shared=(knn@knn.T).tocoo()
    denom=(2*k-shared.data)
    vals=shared.data/denom
    keep=vals>prune
    return sp.csr_matrix((vals[keep],(shared.row[keep],shared.col[keep])),shape=(n,n))

def build_network_from_correlation(cor, k=20, prune=1/15, threshold=0.2):
    c=np.asarray(cor,dtype=np.float64).copy();c[c<0]=0
    d=1-c;n=d.shape[0];kk=min(k,n-1) if n<k else k
    nn=np.vstack([_r_order(d[i])[:kk] for i in range(n)])
    snn=compute_snn(nn,prune).toarray();np.fill_diagonal(snn,0)
    return (snn>threshold).astype(np.float64), nn, snn, d
