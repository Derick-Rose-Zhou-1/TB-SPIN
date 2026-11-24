from __future__ import annotations
import numpy as np
from .r_rng import permutation_indices
from .scoring import score_cells,scaling_stats
from .stats import pnorm_upper_abs,bh_adjust,labels

def permutation_statistics(expression,coefficients,n_permutations=2000,independent=True,fdr_threshold=.05,cell_chunk=2048,permutation_chunk=32,progress=None):
    b=np.asarray(coefficients,dtype=np.float64);rs,means,sd=score_cells(expression,b,chunk_cells=cell_chunk)
    n=len(rs);count=np.zeros(n,dtype=np.int64);mean=np.zeros(n);m2=np.zeros(n)
    gmean=0.0;gm2=0.0;gcount=0
    for start in range(1,n_permutations+1,permutation_chunk):
        end=min(n_permutations,start+permutation_chunk-1)
        for pi in range(start,end+1):
            bp=b[permutation_indices(len(b),pi)];v,_,_=score_cells(expression,bp,means,sd,cell_chunk)
            if independent:
                count+=1;delta=v-mean;mean+=delta/count;m2+=delta*(v-mean)
            else:
                for val in v:
                    gcount+=1;delta=val-gmean;gmean+=delta/gcount;gm2+=delta*(val-gmean)
        if progress: progress(end,n_permutations)
    if independent: bgmean=mean;bgsd=np.sqrt(m2/(count-1))
    else: bgmean=gmean;bgsd=np.sqrt(gm2/(gcount-1))
    z=(rs-bgmean)/bgsd;p=pnorm_upper_abs(z);q=bh_adjust(p);lab=labels(z,q,fdr_threshold)
    return dict(scPAS_RS=rs,background_mean=bgmean,background_sd=bgsd,scPAS_NRS=z,scPAS_Pvalue=p,scPAS_FDR=q,scPAS=lab)
