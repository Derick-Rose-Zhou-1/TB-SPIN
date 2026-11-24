from pathlib import Path
import sys,json,gc,hashlib,traceback,argparse
from dataclasses import replace,asdict
import numpy as np,pandas as pd,torch
p=argparse.ArgumentParser(); p.add_argument('--folds',required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--reference',type=Path,required=True); p.add_argument('--training-code',type=Path,required=True); p.add_argument('--input-h5ad',type=Path,required=True); a=p.parse_args(); out,root,src=a.output,a.reference,a.training_code
sys.path.insert(0,str(src));import tb_state_sc_v3_training as t
t.R2_H5AD=a.input_h5ad
folds=list(map(int,a.folds.split(',')))
torch.set_num_threads(4);device=torch.device('cuda:0');assert torch.cuda.is_available()
variants={'NO_ORDINAL':'lambda_ordinal','NO_CROSSCOV':'lambda_cross','NO_IDENTITY':'lambda_id','NO_KL':'beta_kl'}
src_text=Path(t.__file__).read_text();assert all(v in src_text for v in ['config.lambda_ordinal * ordinal_loss','config.lambda_cross * crosscov','config.lambda_id * identity','config.beta_kl * min(1.0, epoch / 30.0)'])
outer=pd.read_csv(root/'04_SPLITS/PARTICIPANT_OUTER_FOLDS.tsv',sep='\t')
for fold in folds:
 print('LOAD_FOLD',fold,flush=True);data=t.load_result2_fold(root,fold);train=outer.loc[outer.outer_fold.ne(fold),'sample_id_std'].astype(str).tolist();test=outer.loc[outer.outer_fold.eq(fold),'sample_id_std'].astype(str).tolist();assert set(train).isdisjoint(test)
 for seed in (20260828,20260829,20260830):
  original=root/f'08_OOF_MODELS/fold_{fold}/seed_{seed}';done=json.loads((original/'DONE.json').read_text());ck=torch.load(original/'model.pt',map_location='cpu',weights_only=False);assert ck['gene_ids']==data.gene_ids and ck['celltype_levels']==data.celltype_levels
  full=t.TrainConfig(**ck['train_config']);del ck
  for variant,field in variants.items():
   dest=out/'runs'/variant/f'fold_{fold}'/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=False)
   try:
    cfg=replace(full,**{field:0.0}); changed=[k for k in vars(cfg) if getattr(cfg,k)!=getattr(full,k)];assert changed==[field]
    meta={'fold':fold,'seed':seed,'variant':variant,'changed_field':field,'original_value':getattr(full,field),'pretrain_epochs':done['pretrain_epochs'],'finetune_epochs':done['finetune_epochs'],'n_train_participants':len(train),'n_test_participants':len(test),'historical_full_model_sha256':done['model_sha256'],'config':asdict(cfg),'fixed_epochs_from_full':True,'model_weights_saved':False}
    (dest/'protocol.json').write_text(json.dumps(meta,indent=2));print('START',variant,fold,seed,flush=True)
    save=torch.save
    try:
     torch.save=lambda *args,**kwargs:None
     model,hist=t.fit_model_fixed_epochs(data,train,cfg,seed,device,done['pretrain_epochs'],done['finetune_epochs'],dest)
    finally:torch.save=save
    assert np.isfinite(hist.select_dtypes(include='number').to_numpy()).all(), 'NONFINITE_TRAINING_HISTORY'
    pred,_,metrics,quality=t.evaluate_model(model,data,test,cfg,device,False,chunk_size=4096)
    assert np.isfinite(pred[['logit','probability']].to_numpy()).all(); assert set(pred.participant).issubset(test)
    pred.to_csv(dest/'PREDICTIONS.tsv',sep='\t',index=False);metrics.to_csv(dest/'METRICS.tsv',sep='\t',index=False)
    (dest/'STATUS.json').write_text(json.dumps({'status':'PASS','quality':quality},indent=2));print('DONE',variant,fold,seed,flush=True)
    del model,hist,pred,metrics;torch.cuda.empty_cache();gc.collect()
   except Exception as e:
    (dest/'STATUS.json').write_text(json.dumps({'status':'ERROR','error':repr(e)},indent=2));traceback.print_exc();raise
 del data;gc.collect()
print('WORKER_COMPLETE',folds,flush=True)
