from __future__ import annotations
import numpy as np
from scipy.special import ndtr

def pnorm_upper_abs(z): return ndtr(-np.abs(np.asarray(z,dtype=np.float64)))

def bh_adjust(p):
    p=np.asarray(p,dtype=np.float64);out=np.full_like(p,np.nan);ok=np.isfinite(p);x=p[ok];n=len(x)
    order=np.argsort(x,kind="stable")[::-1];rank=np.arange(n,0,-1,dtype=np.float64);adj=np.minimum.accumulate((n/rank)*x[order]);adj=np.minimum(adj,1.0);tmp=np.empty(n);tmp[order]=adj;out[ok]=tmp;return out

def labels(z,fdr,threshold=.05):
    z=np.asarray(z);q=np.asarray(fdr);return np.where((z>0)&(q<=threshold),'scPAS+',np.where((z<0)&(q<=threshold),'scPAS-','0'))
