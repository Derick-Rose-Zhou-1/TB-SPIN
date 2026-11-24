from pathlib import Path
import sys
import numpy as np
import pandas as pd
import h5py, anndata as ad
from scipy.sparse import csr_matrix
from scipy.stats import false_discovery_control
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
P=Path(__file__).resolve().parent
B=P.parent
archive=B/'inputs'
from ora_original import calculate,read_gmt
labels=pd.read_csv(B/'scoring/ensemble_labels.tsv.gz',sep='\t',dtype={'risk':'boolean'})
labels=labels[labels.analysis_stage.ne('ATB')].set_index('cell_id',verify_integrity=True)
sizes=labels.groupby(['analysis_stage','sample_id_std','celltype_raw']).risk.agg(total='size',eligible='count',positive='sum')
sizes['background']=sizes.eligible-sizes.positive
sizes.to_csv(P/'participant_celltype.tsv',sep='\t')
with h5py.File(B/'FIG5_PREP/input/TBSPIN_Fig1_QC_ATBTBETBM_248054.h5ad') as h:
    obs=ad.io.read_elem(h['obs']).index
    genes=ad.io.read_elem(h['var']).index.str.upper()
    assert genes.is_unique and obs.is_unique and labels.index.isin(obs).all()
    node=h['layers/counts']
    raw=csr_matrix((node['data'][:],node['indices'][:],node['indptr'][:]),shape=(len(obs),len(genes)))
meta=labels.reindex(obs)
allterms=[];numbers=[]
for target,minimum in [('DTB',20),('TPE',10),('TBM',20)]:
    mask=(meta.analysis_stage.eq(target)&meta.risk.notna()).to_numpy()
    y=meta[mask].rename_axis('cell_id').reset_index()
    y['p95_status']=np.where(y.risk,'P95_POSITIVE','P95_BACKGROUND')
    x=raw[mask]
    groups=y.groupby(['sample_id_std','p95_status']).indices
    sizes=y.groupby(['sample_id_std','p95_status']).size().unstack(fill_value=0)
    ids=sizes.index[sizes.min(axis=1)>=minimum]
    assert len(ids)=={'DTB':4,'TPE':3,'TBM':3}[target]
    keys=[k for k in groups if k[0] in ids]
    design=pd.DataFrame(keys,columns=['participant','status'])
    design.index=design.participant+'_'+design.status
    counts=pd.DataFrame([np.asarray(x[groups[k]].sum(axis=0,dtype=np.int64)).ravel() for k in keys],index=design.index,columns=genes)
    design.status=pd.Categorical(design.status,categories=['P95_BACKGROUND','P95_POSITIVE'])
    print(target,'paired',list(ids),flush=True)
    dd=DeseqDataSet(counts=counts.loc[:,counts.sum()>0],metadata=design,design='~participant + status',n_cpus=4,quiet=True)
    dd.deseq2();ds=DeseqStats(dd,contrast=['status','P95_POSITIVE','P95_BACKGROUND'],n_cpus=4,quiet=True);ds.summary()
    paired=ds.results_df.rename_axis('gene').reset_index().rename(columns={'padj':'FDR'})
    for method,res in [('paired',paired)]:
        res.to_csv(P/f'{target}_{method}.tsv.gz',sep='\t',index=False)
        universe=set(res.loc[np.isfinite(res.pvalue)&np.isfinite(res.log2FoldChange),'gene'])
        for direction,sign in [('up',1),('down',-1)]:
            query=set(res.loc[res.FDR.lt(.05)&(sign*res.log2FoldChange).ge(1.5),'gene'])
            for db,stem in [('GO_BP','c5.go.bp'),('KEGG','c2.cp.kegg_legacy')]:
                sets=read_gmt(archive/'resources'/f'{stem}.v2025.1.Hs.symbols.gmt')
                tab=calculate(query,universe,sets,'annotated')
                n=sum(bool(v&universe) for v in sets.values())
                tab['full_BH']=false_discovery_control(np.r_[tab.raw_P,np.ones(n-len(tab))])[:len(tab)] if len(tab) else []
                tab['overlap_genes']=[','.join(sorted(sets[v]&query)) for v in tab.term]
                tab=tab.assign(method=method,group=target,direction=direction,library=db)
                allterms.append(tab)
                numbers.append([method,target,len(y),y.sample_id_std.nunique() if method=='cell' else len(ids),direction,len(query),db,int(tab.BH_FDR.lt(.05).sum()),int(tab.full_BH.lt(.05).sum())])
pd.concat(allterms,ignore_index=True).to_csv(P/'ORA_all.tsv.gz',sep='\t',index=False)
pd.DataFrame(numbers,columns=['method','target','eligible_cells_before_pair_filter','patients','direction','DE','library','original_scope_FDR05','full_library_FDR05']).to_csv(P/'summary.tsv',sep='\t',index=False)
