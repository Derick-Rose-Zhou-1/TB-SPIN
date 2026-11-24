from pathlib import Path
import json,hashlib
import h5py,anndata as ad,pandas as pd,numpy as np
root=Path(__file__).resolve().parents[1]
out=root/'FIG5_PREP'; path=out/'input/TBSPIN_Fig1_QC_ATBTBETBM_248054.h5ad'
h=hashlib.sha256()
with path.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()=='e649e723d9cab502e13a1933661974d201c37898a0d1b6960e771ae875476ee0'
with h5py.File(path) as f:
 obs=ad.io.read_elem(f['obs']);var=ad.io.read_elem(f['var']);shape=tuple(f['layers/counts'].attrs['shape'])
assert shape==(248054,16986) and obs.index.is_unique and var.index.is_unique
meta=pd.DataFrame({'cell_id':obs.index.astype(str),'sample_id_std':obs.sample_id.astype(str).to_numpy(),'analysis_stage':obs.group.astype(str).to_numpy(),'celltype_raw':obs.celltype_major.astype(str).to_numpy()})
assert meta.groupby('sample_id_std').analysis_stage.nunique().max()==1
assert meta.sample_id_std.nunique()==30
expected={'ATB':141802,'DTB':52450,'TPE':37945,'TBM':15857}
assert meta.analysis_stage.value_counts().to_dict()==expected
gene=pd.read_csv(root/'05_FEATURES/FOLD_FINAL_HVG5000.tsv',sep='\t').sort_values('feature_rank')
positions=var.index.get_indexer(gene.gene_id.astype(str)); assert (positions>=0).all()
meta.to_csv(out/'MODEL_READY_METADATA.tsv.gz',sep='\t',index=False)
pd.DataFrame({'gene_id':gene.gene_id,'h5ad_position':positions,'feature_rank':gene.feature_rank}).to_csv(out/'HVG5000_ORDERED_POSITIONS.tsv',sep='\t',index=False)
outer=pd.read_csv(out/'PARTICIPANT_OUTER_FOLDS.tsv',sep='\t');inner=pd.read_csv(out/'PARTICIPANT_INNER_FOLDS.tsv',sep='\t')
assert outer.sample_id.is_unique and set(outer.sample_id)==set(meta.sample_id_std)
split_rows=[]
for fold in sorted(outer.outer_fold.unique()):
 tr=set(outer[outer.outer_fold.ne(fold)].sample_id);te=set(outer[outer.outer_fold.eq(fold)].sample_id)
 assert not tr&te
 sub=inner[inner.outer_fold.eq(fold)];assert set(sub.sample_id)==tr and sub.sample_id.is_unique
 for k in sorted(sub.inner_fold.unique()):
  a=set(sub[sub.inner_fold.ne(k)].sample_id);b=set(sub[sub.inner_fold.eq(k)].sample_id)
  assert not a&b and not (a|b)&te
  for target in ['DTB','TPE','TBM']:
   for part,ids in [('train',a),('validation',b)]:
    stages=outer[outer.sample_id.isin(ids)].analysis_stage
    n0=int(stages.eq('ATB').sum());n1=int(stages.eq(target).sum());assert n0>0 and n1>0
    split_rows.append(dict(outer_fold=int(fold),inner_fold=int(k),part=part,target=target,reference_participants=n0,target_participants=n1))
pd.DataFrame(split_rows).to_csv(out/'SPLIT_CLASS_SUPPORT.csv',index=False)
up=pd.read_csv(root/'02_INPUT_AUDIT/MODEL_READY_METADATA.tsv.gz',sep='\t',usecols=['cell_id','sample_id_std','analysis_stage'])
common=set(up.cell_id)&set(meta.cell_id)
audit={'status':'INPUT_AND_SPLIT_PREFLIGHT_PASS','path':str(path),'sha256':h.hexdigest(),'cells':len(meta),'participants':30,'groups':expected,'celltype_counts':pd.crosstab(meta.analysis_stage,meta.celltype_raw).to_dict(),'hvg_count':len(gene),'missing_genes':0,'gene_order_sha256':hashlib.sha256('\n'.join(gene.gene_id).encode()).hexdigest(),'upstream_exact_cell_id_overlap':len(common),'scope':'Fig5-stage participant disjointness only; not fully independent of upstream pretraining','training_launched':False,'risk_method_status':'UNRESOLVED_HISTORICAL_CODE_VS_MANUSCRIPT','downstream_enabled':False}
(out/'PREFLIGHT.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
