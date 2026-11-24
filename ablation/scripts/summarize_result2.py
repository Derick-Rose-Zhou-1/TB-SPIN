from pathlib import Path
import json, argparse
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);args=p.parse_args();O,R=args.output,args.reference;seeds=[20260828,20260829,20260830];variants=['NO_ORDINAL','NO_CROSSCOV','NO_IDENTITY','NO_KL','MEAN_ONLY','NO_PARTICIPANT_BALANCING'];metrics={'AUROC':roc_auc_score,'AUPRC':average_precision_score,'Brier':brier_score_loss}
rows=[];boot=[];rng=np.random.default_rng(20261124)
for var in variants:
 aligned={}
 for seed in seeds:
  aa=[];bb=[]
  for fold in range(5):
   d=O/'runs'/var/f'fold_{fold}'/f'seed_{seed}';assert json.loads((d/'STATUS.json').read_text())['status']=='PASS'
   aa.append(pd.read_csv(d/'PREDICTIONS.tsv',sep='\t'));bb.append(pd.read_csv(R/f'10_OOF_PARTICIPANT/fold_{fold}_seed_{seed}.tsv',sep='\t'))
  a=pd.concat(aa,ignore_index=True);b=pd.concat(bb,ignore_index=True);x=b.merge(a,on=['task','participant'],suffixes=('_full','_ablation'),validate='one_to_one');assert len(x)==len(a)==len(b);assert (x.y_true_full==x.y_true_ablation).all()
  for task,g in x.groupby('task'):
   g=g.sort_values('participant');assert not g.participant.duplicated().any();aligned[(seed,task)]=g
   for metric,f in metrics.items():
    full=float(f(g.y_true_full,g.probability_full));ab=float(f(g.y_true_full,g.probability_ablation));rows.append({'variant':var,'seed':seed,'task':task,'metric':metric,'n':len(g),'positive':int(g.y_true_full.sum()),'full':full,'ablation':ab,'delta':ab-full})
 for task in ('T1','T2'):
  gs=[aligned[(s,task)] for s in seeds];ids=gs[0].participant.tolist();assert all(g.participant.tolist()==ids and np.array_equal(g.y_true_full,gs[0].y_true_full) for g in gs)
  y=gs[0].y_true_full.to_numpy();draws={m:[] for m in metrics}
  for i in range(2000):
   idx=rng.integers(0,len(y),len(y)); yy=y[idx]
   if len(np.unique(yy))<2:continue
   for metric,f in metrics.items():draws[metric].append(float(np.mean([f(yy,g.probability_ablation.to_numpy()[idx])-f(yy,g.probability_full.to_numpy()[idx]) for g in gs])))
  for metric,vals in draws.items():boot.append({'variant':var,'task':task,'metric':metric,'n_boot_estimable':len(vals),'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975)),'bootstrap_unit':'participant_paired_same_draw_across_seeds'})
res=pd.DataFrame(rows);res.to_csv(O/'R2_POOLED_SEED_METRICS.tsv',sep='\t',index=False);summary=res.groupby(['variant','task','metric'],as_index=False).agg(n=('n','first'),positive=('positive','first'),full_mean=('full','mean'),ablation_mean=('ablation','mean'),delta_mean=('delta','mean'),delta_sd=('delta','std'));summary.merge(pd.DataFrame(boot),on=['variant','task','metric'],validate='one_to_one').to_csv(O/'R2_POOLED_SUMMARY_BOOTSTRAP.tsv',sep='\t',index=False)
print('R2_SUMMARY_COMPLETE',len(res),len(summary),flush=True)
