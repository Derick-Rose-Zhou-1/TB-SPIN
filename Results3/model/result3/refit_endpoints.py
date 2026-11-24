from pathlib import Path
import sys,json,traceback,gc
import numpy as np,pandas as pd,torch,joblib
from sklearn.preprocessing import StandardScaler
import argparse
parser=argparse.ArgumentParser(description="Refit locked clinical heads using frozen outer alignments.")
parser.add_argument("release",type=Path)
parser.add_argument("output",type=Path)
args=parser.parse_args()
R=args.release
O=args.output
O.mkdir(parents=True,exist_ok=True)
import v126_training as t
import endpoint_heads as reconstruct
run=R/'RESULTS/V126_RUN';locks=pd.read_csv(run/'05_MODELS/FINAL_CONFIGURATION_LOCKS.tsv',sep='\t');end=pd.read_csv(run/'02_ENDPOINTS/PARTICIPANT_ENDPOINTS.tsv',sep='\t',low_memory=False);end['participant_id']=end.participant_id.astype(str);end['GSM']=end.GSM.astype(str)
bulk=joblib.load(R/'RESULTS/CORE/01_SHARED_BULK_LOG2CPM.joblib');bulk.index=bulk.index.astype(str);genes=pd.read_csv(R/'RESULTS/CORE/01_SHARED_GENE_LIST.tsv',sep='\t').gene_symbol.astype(str).tolist();assert list(bulk.columns.astype(str))==genes and not(set(end.GSM)-set(bulk.index));assert torch.cuda.is_available()
sources={}
for cohort in t.FORMAL_COHORTS:
 name='heldout_GSE107993_GSE107994' if cohort.endswith('combined') else f'heldout_{cohort}';sources['v125_model_'+cohort]=R/'CHECKPOINTS/FROZEN_OUTER_SOURCE_ARTIFACTS'/name/'LOCKED_FINAL_MODEL.pt'
bundle=t.DataBundle(run=run,sources=sources,endpoints=end,bulk=bulk,genes=genes,device=torch.device('cuda:0'));archive=pd.read_csv(run/'06_OUTER_EVALUATION/V126_OUTER_PREDICTIONS.tsv',sep='\t');archive['participant_id']=archive.participant_id.astype(str);archive['GSM']=archive.GSM.astype(str);t.seed_all();rows=[]
for r in locks.itertuples(index=False):
 try:
  ep=r.endpoint;cohort=r.outer_heldout;print('START',cohort,ep,r.model,flush=True);train=t.frame_for(bundle,ep,set(t.FORMAL_COHORTS)-{cohort});held=t.frame_for(bundle,ep,{cohort});assert set(train.participant_id).isdisjoint(held.participant_id)
  if held.empty:
   rows.append({'cohort':cohort,'endpoint':ep,'model':r.model,'status':'STRUCTURAL_NE_NO_ELIGIBLE_HELDOUT'});print('STRUCTURAL_NE',cohort,ep,flush=True);continue
  alignment=t.load_alignment(bundle,cohort);statistics=t.cox_gene_statistics(bundle,train) if ep=='E_COX' else t.binary_gene_statistics(bundle,train,t.SEED);weights,n=t.gene_weights(statistics,r.gene_condition,ep);mean=bundle.bulk.loc[train.GSM,genes].to_numpy(dtype=np.float32).mean(axis=0);x=t.latent_features(bundle,alignment,train,weights,mean);xh=t.latent_features(bundle,alignment,held,weights,mean)
  if r.model=='B_LAST_BLOCK_MLP16':artifact=reconstruct.fit_last_binary_artifact(bundle,alignment,train,weights,mean)
  elif r.model=='B_LAST_BLOCK_COX16':artifact=reconstruct.fit_last_cox_artifact(bundle,alignment,train,weights,mean)
  elif ep=='E_COX':artifact={'kind':'cox',**t.fit_cox_latent(x,train,r.model)}
  else:
   scaler=StandardScaler().fit(x);est=t.binary_estimator(r.model);est.fit(scaler.transform(x),train.y_true,sample_weight=train.participant_metric_weight);artifact={'kind':'sklearn','scaler':scaler,'estimator':est}
  if artifact['kind'].startswith('last_block'):
   predx=bundle.bulk.loc[held.GSM,genes].to_numpy(dtype=np.float32);predx=mean+(predx-mean)*weights
  else:predx=xh
  score=np.asarray(reconstruct.predict_artifact(artifact,predx,ep)).reshape(-1);ref=archive[(archive.outer_heldout==cohort)&(archive.endpoint==ep)];new=held[['participant_id','GSM']].copy();new['new_score']=score;j=new.merge(ref[['participant_id','GSM','score']],on=['participant_id','GSM'],validate='one_to_one');assert len(j)==len(new)==len(ref)
  j.to_csv(O/f"{cohort}_{ep}_predictions.tsv",sep="\t",index=False)
  diff=np.abs(j.new_score-j.score);maxdiff=float(diff.max());rho=float(j.new_score.corr(j.score,method='spearman'));record={'cohort':cohort,'endpoint':ep,'model':r.model,'gene_condition':r.gene_condition,'n':len(j),'n_participants':int(held.participant_id.nunique()),'max_abs_difference':maxdiff,'mean_abs_difference':float(diff.mean()),'spearman':rho,'status':'NUMERICAL_PASS_1E-6' if maxdiff<=1e-6 else 'MISMATCH'};rows.append(record);print('DONE',json.dumps(record),flush=True);del alignment,artifact,x,xh;torch.cuda.empty_cache();gc.collect()
 except Exception as e:
  record={'cohort':r.outer_heldout,'endpoint':r.endpoint,'model':r.model,'status':'ERROR','error':repr(e)};rows.append(record);print('ERROR',json.dumps(record),flush=True);traceback.print_exc()
 finally:pd.DataFrame(rows).to_csv(O/'FROZEN_ALIGNMENT_ENDPOINT_REFIT_COMPARISON.tsv',sep='\t',index=False)
print('ALL_ENDPOINTS_PROCESSED',len(rows),flush=True)

if any(row['status']=='ERROR' for row in rows): raise RuntimeError('See endpoint refit summary')
