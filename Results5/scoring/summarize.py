from pathlib import Path
import numpy as np, pandas as pd
P=Path(__file__).resolve().parent
B=P.parent
meta_path=B/'inputs/FIG1_QC_CELL_METADATA_UMAP.tsv.gz'
meta=pd.read_csv(meta_path,sep='\t').set_index('cell_id',verify_integrity=True)
assert len(meta)==248054
keys=['cell_id','task','fold','analysis_stage','sample_id_std','celltype_raw']
s=[]
for run in [20260828,20260829,20260830]:
    t=pd.read_csv(P/f'heldout_{run}.tsv.gz',sep='\t',usecols=keys+['R']).set_index(keys,verify_integrity=True).sort_index()
    if s: assert s[0].index.equals(t.index)
    s.append(t.R.rename('R_'+str(run)))
d=pd.concat(s,axis=1)
assert d.count(axis=1).isin([0,3]).all()
d['R_median']=d.median(axis=1)
d['risk']=d.R_median.ge(.95).astype('boolean').mask(d.R_median.isna())
d=d.reset_index()
assert not d.duplicated(['cell_id','task']).any()
d.to_csv(P/'ensemble_labels.tsv.gz',sep='\t',index=False)
targets=d[d.analysis_stage.ne('ATB')]
assert len(targets)==106252 and targets.cell_id.is_unique
lookup=meta.loc[d.cell_id]
if 'celltype_raw' not in lookup:
    check=pd.read_csv(B/'FIG5_PREP/MODEL_READY_METADATA.tsv.gz',sep='\t').set_index('cell_id',verify_integrity=True)
    assert np.array_equal(lookup.analysis_stage.to_numpy(),d.analysis_stage.to_numpy())
    lookup=check.loc[d.cell_id]
for k in ['analysis_stage','celltype_raw','sample_id_std']:
    assert np.array_equal(d[k].to_numpy(),lookup[k].to_numpy())
patient=d.groupby(['task','analysis_stage','sample_id_std']).agg(total=('risk','size'),eligible=('risk','count'),positive=('risk','sum')).astype(int)
patient['background']=patient.eligible-patient.positive
patient['unsupported']=patient.total-patient.eligible
patient['paired']=pd.array([pd.NA if g=='ATB' else min(row.background,row.positive)>=(10 if g=='TPE' else 20) for (_,g,_),row in patient.iterrows()],dtype='boolean')
patient.to_csv(P/'participants.tsv',sep='\t')
summary=patient.groupby(level=[0,1]).sum(min_count=1)
summary['percent']=100*summary.positive/summary.eligible
summary.to_csv(P/'summary.tsv',sep='\t')
print(summary.to_string());print(patient[patient.index.get_level_values(1)!='ATB'].to_string())

