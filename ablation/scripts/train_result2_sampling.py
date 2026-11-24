from pathlib import Path
import sys,json,gc,argparse,traceback
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
p=argparse.ArgumentParser();p.add_argument('--folds',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--training-code',type=Path,required=True);p.add_argument('--input-h5ad',type=Path,required=True);a=p.parse_args();O,R=a.output,a.reference;sys.path.insert(0,str(a.training_code))
import tb_state_sc_v3_training as t
t.R2_H5AD=a.input_h5ad
original_batches=t.participant_batches;original_model=t.TBStateSCV3

def unbalanced(data,participants,config,rng):
 available=np.concatenate([data.participant_to_cells[p] for p in participants]);n=min(len(available),len(participants)*config.cells_per_participant);selected=rng.choice(available,n,replace=False);rng.shuffle(selected);names_all=data.metadata[data.participant_col].astype(str).to_numpy();bs=config.participants_per_batch*config.cells_per_participant;out=[]
 for start in range(0,n,bs):
  idx=np.asarray(selected[start:start+bs],dtype=np.int64);names,local=np.unique(names_all[idx],return_inverse=True);stages=np.asarray([t.STAGE_ORDINAL[data.participant_to_stage[x]] for x in names],dtype=np.int64);out.append((idx,local.astype(np.int64),stages))
 return out

def mean_only(*args,**kwargs):
 model=original_model(*args,**kwargs)
 for h in model.transition_heads.values():
  h.raw_tail_weight.data.fill_(-30.0);h.raw_tail_weight.requires_grad_(False)
 return model

def selftest():
 m=mean_only(n_genes=8,n_major_celltypes=2,n_technical_batches=1,transition_names=t.TASKS,state_dim=8,tau=.5,identity_dim=16,hidden_dim=16,likelihood='zinb')
 assert all((not h.raw_tail_weight.requires_grad) and h.raw_tail_weight.item()==-30.0 and torch.nn.functional.softplus(h.raw_tail_weight).item()<1e-12 for h in m.transition_heads.values())
 d=SimpleNamespace(participant_to_cells={'a':np.arange(5),'b':np.arange(5,8)},metadata=pd.DataFrame({'sample':['a']*5+['b']*3}),participant_col='sample',participant_to_stage={'a':'HC','b':'ATB'});c=SimpleNamespace(cells_per_participant=3,participants_per_batch=2);b=unbalanced(d,['a','b'],c,np.random.default_rng(42));idx=np.concatenate([x[0] for x in b]);assert len(idx)==6 and len(set(idx))==6 and all(set(x[2]).issubset({0,2}) for x in b)
 print('EXTRA_ABLATION_SELFTEST_PASS',flush=True)

selftest()
torch.set_num_threads(3);device=torch.device('cuda:0');assert torch.cuda.is_available();outer=pd.read_csv(R/'04_SPLITS/PARTICIPANT_OUTER_FOLDS.tsv',sep='\t')
for fold in map(int,a.folds.split(',')):
 print('LOAD_FOLD',fold,flush=True);data=t.load_result2_fold(R,fold);train=outer.loc[outer.outer_fold.ne(fold),'sample_id_std'].astype(str).tolist();test=outer.loc[outer.outer_fold.eq(fold),'sample_id_std'].astype(str).tolist();assert set(train).isdisjoint(test)
 for seed in (20260828,20260829,20260830):
  source=R/f'08_OOF_MODELS/fold_{fold}/seed_{seed}';done=json.loads((source/'DONE.json').read_text());ck=torch.load(source/'model.pt',map_location='cpu',weights_only=False);assert ck['gene_ids']==data.gene_ids and ck['celltype_levels']==data.celltype_levels;cfg=t.TrainConfig(**ck['train_config']);del ck
  for var in ('MEAN_ONLY','NO_PARTICIPANT_BALANCING'):
   dest=O/'runs'/var/f'fold_{fold}'/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=False)
   try:
    desc='freeze tail contribution at softplus(-30)' if var=='MEAN_ONLY' else 'sample cells uniformly rather than equally per participant';(dest/'protocol.json').write_text(json.dumps({'fold':fold,'seed':seed,'variant':var,'change':desc,'pretrain_epochs':done['pretrain_epochs'],'finetune_epochs':done['finetune_epochs'],'historical_full_model_sha256':done['model_sha256'],'config':vars(cfg),'weights_saved':False},indent=2));print('START',var,fold,seed,flush=True)
    t.TBStateSCV3=mean_only if var=='MEAN_ONLY' else original_model;t.participant_batches=unbalanced if var=='NO_PARTICIPANT_BALANCING' else original_batches;save=torch.save
    try:
     torch.save=lambda *args,**kwargs:None
     model,hist=t.fit_model_fixed_epochs(data,train,cfg,seed,device,done['pretrain_epochs'],done['finetune_epochs'],dest)
    finally:torch.save=save;t.TBStateSCV3=original_model;t.participant_batches=original_batches
    assert np.isfinite(hist.select_dtypes(include='number').to_numpy()).all();pred,_,metrics,quality=t.evaluate_model(model,data,test,cfg,device,False,chunk_size=4096);assert np.isfinite(pred[['logit','probability']].to_numpy()).all() and set(pred.participant).issubset(test)
    pred.to_csv(dest/'PREDICTIONS.tsv',sep='\t',index=False);metrics.to_csv(dest/'METRICS.tsv',sep='\t',index=False);(dest/'STATUS.json').write_text(json.dumps({'status':'PASS','quality':quality},indent=2));print('DONE',var,fold,seed,flush=True);del model,hist,pred,metrics;torch.cuda.empty_cache();gc.collect()
   except Exception as e:
    (dest/'STATUS.json').write_text(json.dumps({'status':'ERROR','error':repr(e)},indent=2));traceback.print_exc();raise
 del data;gc.collect()
print('EXTRA_WORKER_COMPLETE',a.folds,flush=True)
