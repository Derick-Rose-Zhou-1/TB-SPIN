from pathlib import Path
import json, argparse
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);args=p.parse_args();O,P=args.output,args.pilot;rows=[]
for seed in (2026091801,2026091802,2026091803):
 a=pd.read_csv(P/f'R3_FULL_seed{seed}/OUTER_METRICS.tsv',sep='\t')
 for var in ('NO_ORDINAL_SINGLE_CELL_SUPERVISION','NO_ENDPOINT_GENE_WEIGHTING','NO_CORAL','NO_MMD'):
  if var=='NO_ORDINAL_SINGLE_CELL_SUPERVISION':path=P/f'R3_{var}_seed{seed}/OUTER_METRICS.tsv'
  elif var=='NO_ENDPOINT_GENE_WEIGHTING':path=O/f'r3/{var}/seed_{seed}/OUTER_METRICS.tsv'
  else:path=O/f'r3/{var}/seed_{seed}/results/OUTER_METRICS.tsv'
  b=pd.read_csv(path,sep='\t');x=a.merge(b,on=['heldout_cohort','endpoint'],suffixes=('_full','_abl'),validate='one_to_one');assert len(x)==9
  assert (x.model_full.fillna('NA')==x.model_abl.fillna('NA')).all();assert (x.gene_condition_full.fillna('NA')==x.gene_condition_abl.fillna('NA')).all()
  for _,r in x.iterrows():
   metric='Harrell_C' if r.endpoint=='E_COX' else 'AUROC';full=r[metric+'_full'];abl=r[metric+'_abl'];rows.append({'variant':var,'seed':seed,'cohort':r.heldout_cohort,'endpoint':r.endpoint,'metric':metric,'full':full,'ablation':abl,'delta':abl-full,'n_outer':r.n_outer_full,'status_full':r.status_full,'status_ablation':r.status_abl})
z=pd.DataFrame(rows);z.to_csv(O/'R3_PAIRED_SEED_METRICS.tsv',sep='\t',index=False);sm=z.groupby(['variant','cohort','endpoint','metric'],as_index=False,dropna=False).agg(n_outer=('n_outer','first'),n_estimable=('delta','count'),full_mean=('full','mean'),ablation_mean=('ablation','mean'),delta_mean=('delta','mean'),delta_sd=('delta','std'));sm.to_csv(O/'R3_PAIRED_SUMMARY.tsv',sep='\t',index=False);print('R3_SUMMARY_COMPLETE',len(z),len(sm),flush=True)
