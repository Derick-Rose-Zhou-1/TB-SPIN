import sys,json,gc,hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Results2/model/result2'))
import diagnostic_data as d
import locked_core as core
import numpy as np,pandas as pd,torch
root=d.root/'FIG5_LOCKED100_20261116';fold=d.args.fold
out=root/f'fold_{fold}';out.mkdir(parents=True,exist_ok=True)
assert not (out/'DONE.json').exists()
outer=pd.read_csv(d.prep/'PARTICIPANT_OUTER_FOLDS.tsv',sep='\t')
inner=pd.read_csv(d.prep/'PARTICIPANT_INNER_FOLDS.tsv',sep='\t');inner=inner[inner.outer_fold==fold]
train=outer.loc[outer.outer_fold!=fold,'sample_id'].tolist();test=outer.loc[outer.outer_fold==fold,'sample_id'].tolist()
assert not set(train)&set(test)
selection=[]
for k in range(3):
    dev=inner.loc[inner.inner_fold!=k,'sample_id'].tolist();valid=inner.loc[inner.inner_fold==k,'sample_id'].tolist()
    assert not set(dev)&set(valid) and not set(valid)&set(test)
    print(f'INNER_START {fold} {k}',flush=True)
    m,best,h=core.train_result6_model(d.data,dev,valid,d.config,20260829+fold*10+k,d.device,d.root/'17_FINAL_FULL_MODEL/seed_20260829/model.pt',False,5,100,True,out/f'inner_{k}')
    assert np.isfinite(h.loss).all()
    old=pd.read_csv(d.root/f'CALIBRATION_EARLYSTOP_20261116/fold_{fold}/extended_inner/inner_{k}/training_history.tsv',sep='\t')
    j=h.merge(old,on=['phase','epoch'],suffixes=('_new','_old'))
    delta=float((j.loss_new-j.loss_old).abs().max());assert delta<1e-6
    selection.append({'inner_fold':k,'selected_epoch':best,'first40_loss_max_difference':delta})
    del m;gc.collect();torch.cuda.empty_cache()
epochs=max(1,int(round(np.median([s['selected_epoch'] for s in selection]))))
(out/'SELECTION.json').write_text(json.dumps({'epochs':epochs,'inner':selection,'train_participants':train,'test_participants':test},indent=2))
for seed in d.seeds:
    folder=out/f'seed_{seed}';print(f'OUTER_START {fold} {seed} epoch={epochs}',flush=True)
    m,_,h=core.train_result6_model(d.data,train,None,d.config,seed,d.device,d.root/f'17_FINAL_FULL_MODEL/seed_{seed}/model.pt',False,5,epochs,False,folder)
    assert np.isfinite(h.loss).all() and all(torch.isfinite(v).all() for v in m.state_dict().values())
    p,_,metrics,q=d.training.evaluate_model(m,d.data,test,d.config,d.device,False)
    p.to_csv(folder/'heldout_predictions.tsv',sep='\t',index=False)
    metrics.to_csv(folder/'heldout_metrics.tsv',sep='\t',index=False)
    for name,ids in [('training_latent',train),('heldout_latent',test)]:
        z=d.training.encode_cell_states(m,d.data,ids,d.device)
        z.to_csv(folder/f'{name}.tsv.gz',sep='\t',index=False)
    (folder/'QUALITY.json').write_text(json.dumps(q,indent=2))
    del m;gc.collect();torch.cuda.empty_cache()
(out/'DONE.json').write_text(json.dumps({'status':'TRAINING_COMPLETE','fold':fold,'epochs':epochs,'risk_complete':False},indent=2))
print(f'FOLD_DONE {fold}',flush=True)
