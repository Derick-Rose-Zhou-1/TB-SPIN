"""Rebuild the Fig. 2 molecular analysis from frozen labels (no training)."""
from pathlib import Path
import os
import hashlib
import h5py
import numpy as np
import pandas as pd
from scipy import sparse, stats

OUT = Path(os.environ.get("TBSPIN_FIG2_WORK", "figure2_work"))
OUT.mkdir(exist_ok=True, parents=True)
BASE = Path(os.environ.get("TBSPIN_FIG2_SOURCE", "inputs/source_data"))
MASTER = Path(os.environ.get("TBSPIN_R2_LABELS", "inputs/RESULT2_FINAL_RISK_CELL_MASTER.tsv.gz"))
H5 = Path(os.environ.get("TBSPIN_R2_COUNTS", "inputs/counts.h5ad"))
GMT = Path(os.environ.get("TBSPIN_GO_BP", "inputs/c5.go.bp.v2025.1.Hs.symbols.gmt"))

def load_infection(labels):
    with h5py.File(H5) as h:
        ids = pd.Index(h["obs"][h["obs"].attrs["_index"]].asstr()[:])
        genes = pd.Index(h["var"][h["var"].attrs["_index"]].asstr()[:]).str.upper()
        assert ids.is_unique and genes.is_unique and set(labels.index) <= set(ids)
        mask = ids.isin(labels.index)
        g = h["layers/counts"]
        ptr = g["indptr"][:]
        blocks = []
        for start in range(0,len(ids),4000):
            end = min(len(ids),start+4000)
            sel = mask[start:end]
            if not sel.any():
                continue
            lo,hi = ptr[start],ptr[end]
            chunk = sparse.csr_matrix((g["data"][lo:hi],g["indices"][lo:hi],ptr[start:end+1]-lo),shape=(end-start,len(genes)))[sel].astype("float32")
            total = np.asarray(chunk.sum(axis=1)).ravel()
            assert min(total)>0
            blocks.append(chunk)
        return sparse.vstack(blocks).tocsc(),ids[mask],genes

def differential(mat,ids,genes,labels):
    valid = labels.reindex(ids).notna().to_numpy()
    x=mat[valid].tocsc()
    pos=labels.reindex(ids[valid]).ge(.95).to_numpy()
    n=len(pos); a=int(pos.sum()); b=n-a
    testable=np.maximum(x[pos].getnnz(axis=0)/a,x[~pos].getnnz(axis=0)/b)>=.01
    genes=genes[testable];x=x[:,testable].tocsr()
    total=np.asarray(x.sum(axis=1)).ravel()
    x.data/=np.repeat(np.where(total>0,total,1)/np.float32(10000),np.diff(x.indptr))
    np.log1p(x.data,out=x.data);x=x.tocsc()
    rows=[]
    for j,gene in enumerate(genes):
        lo,hi=x.indptr[j:j+2]
        values=x.data[lo:hi]; active=pos[x.indices[lo:hi]]
        va=values[active]; vb=values[~active]
        pa=len(va)/a; pb=len(vb)/b
        if max(pa,pb)<.01:
            continue
        zeros=n-len(values); zero_rank=(zeros+1)/2
        ranks=stats.rankdata(values)+zeros
        rank_sum=(a-len(va))*zero_rank+ranks[active].sum()
        ties=np.unique(values,return_counts=True)[1].astype(float)
        correction=1-((ties**3-ties).sum()+float(zeros)**3-zeros)/(float(n)**3-n)
        z=(rank_sum-a*(n+1)/2)/np.sqrt(a*b*(n+1)*correction/12) if correction>0 else 0.
        ma=va.sum(dtype=float)/a; mb=vb.sum(dtype=float)/b
        lfc=np.log2((np.expm1(ma)+1e-9)/(np.expm1(mb)+1e-9))
        rows.append((gene,lfc,2*stats.norm.sf(abs(z)),pa,pb,ma,mb,z))
    table=pd.DataFrame(rows,columns=['gene','log2FC','raw_pvalue','pct_risk','pct_background','mean_risk','mean_background','wilcoxon_z'])
    table['FDR']=stats.false_discovery_control(table.raw_pvalue)
    return table

def signature(table):
    return table[(table.FDR<.05)&(table.log2FC>1.5)].sort_values(['FDR','raw_pvalue','log2FC','gene'],ascending=[True,True,False,True])

def ora(table):
    # clusterProfiler-style hypergeometric test against GO-annotated testable genes.
    sets={}
    for line in GMT.read_text(encoding='utf-8').splitlines():
        term,_,*symbols=line.split('\t')
        sets.setdefault(term,set()).update(s.upper() for s in symbols)
    universe=set(table.gene)&set.union(*sets.values())
    query=set(signature(table).gene)&universe
    rows=[]
    for term,gs in sets.items():
        members=gs&universe; overlap=members&query
        if not overlap or not 10<=len(members)<=500:continue
        p=stats.hypergeom.sf(len(overlap)-1,len(universe),len(members),len(query))
        rows.append((term,len(universe),len(query),len(members),len(overlap),p,'/'.join(sorted(overlap))))
    result=pd.DataFrame(rows,columns=['term','universe','query','term_n','overlap_n','pvalue','genes'])
    if len(result):result['FDR']=stats.false_discovery_control(result.pvalue)
    return result.sort_values(['FDR','pvalue','term'])

def main():
    assert hashlib.sha256(MASTER.read_bytes()).hexdigest()=='e5490505b522ac7168896bf3ce6b56c0bd3b9102e44d21a4ffae3cc1fcabd6f0'
    m=pd.read_csv(MASTER,sep='\t',usecols=['cell_id','task','analysis_stage','R_ensemble','risk_layer_status'])
    def scores(task,stage):
        s=m[(m.task==task)&(m.analysis_stage==stage)].set_index('cell_id',verify_integrity=True)
        assert s.R_ensemble.notna().equals(s.risk_layer_status.eq('PASS'))
        return s.R_ensemble
    infection=scores('T1','TBINF')
    assert (infection.count(),infection.ge(.95).sum())==(72891,2602)
    mat,ids,genes=load_infection(infection)
    # Reproduce the old T2 infection signature before changing task.
    old=differential(mat,ids,genes,scores('T2','TBINF'))
    old_sig=pd.read_csv(BASE/'volcano/LTBI_PRIMARY_RISK_UP_FDR05_LFC15_N50.tsv',sep='\t')
    assert set(signature(old).gene)==set(old_sig.gene)
    frozen=pd.read_csv(BASE/'volcano/LTBI_SOURCE_ALL_TESTABLE_DEG.tsv.gz',sep='\t').set_index('gene')
    assert set(frozen.index)==set(old.gene)
    for col in ['pct_risk','pct_background','log2FC']:
        np.testing.assert_allclose(old[col],frozen.loc[old.gene,col],atol=2e-4)
    lock=pd.read_csv(BASE/'go/FIG2D_GO_SELECTED_20.tsv',sep='\t')
    lock=lock[lock.stage.str.startswith('LTBI')]
    np.testing.assert_allclose(ora(old).set_index('term').loc[lock.term,'FDR'],lock['p.adjust'],rtol=1e-9)
    result=differential(mat,ids,genes,infection)
    sig=signature(result)
    result.to_csv(OUT/'T1_infection_DE.tsv.gz',sep='\t',index=False)
    sig.to_csv(OUT/'T1_infection_up.tsv',sep='\t',index=False)
    ora(result).to_csv(OUT/'T1_infection_GO_BP.tsv',sep='\t',index=False)
    # Rank positive genes by z to break P-value underflow ties.
    bars=[]
    pos=infection.reindex(ids).ge(.95).to_numpy()
    valid=infection.reindex(ids).notna().to_numpy()
    def bar(stage,gene,status,x,p):
        bars.append((stage,gene,status,len(x),np.mean(x,dtype=float),stats.t.ppf(.975,len(x)-1)*np.std(x,dtype=float,ddof=1)/np.sqrt(len(x)),p))
    for row in sig.nlargest(3,'wilcoxon_z').itertuples():
        total=np.asarray(mat.sum(axis=1)).ravel()
        x=np.log1p(mat[:,genes.get_loc(row.gene)].toarray().ravel()/(total/np.float32(10000)))
        for name,mask in [('Background',valid&~pos),('Risk',pos)]:bar('TB infection',row.gene,name,x[mask],row.raw_pvalue)
    data=pd.read_csv(BASE/'six_genes/SELECTED_SIX_GENES_CELL_LEVEL_EXPRESSION.tsv.gz',sep='\t')
    s=scores('T2','ATB').dropna()
    d=data[(data.clinical_state=='ATB')&data.cell_id.isin(s.index)].copy()
    audit=pd.read_csv(BASE/'six_genes/SELECTED_SIX_GENES_CELL_LEVEL_DE_AUDIT.tsv',sep='\t').set_index('gene')
    for gene,sub in d.groupby('gene',sort=False):
        assert set(sub.cell_id)==set(s.index) and sub.cell_id.is_unique
        for name,mask in [('Background',sub.cell_id.map(s)<.95),('Risk',sub.cell_id.map(s)>=.95)]:
            bar('TB disease',gene,name,sub.loc[mask,'log1p_normalized_expression'].to_numpy(),audit.loc[gene,'cell_level_raw_P'])
    pd.DataFrame(bars,columns=['stage','gene','group','n','mean','ci95','raw_pvalue']).to_csv(OUT/'k_expression.tsv',sep='\t',index=False)
    print('T1:',len(result),'tested,',len(sig),'up;',sig.nlargest(3,'wilcoxon_z').gene.tolist(),flush=True)

if __name__=='__main__':
    main()
