"""Predict held-out participants using the 15 original fold/seed checkpoints."""
import argparse
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import torch
import tb_state_sc_v3_training as t
from tbstate_sc_v3_model import TBStateSCV3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
parser.add_argument("expression", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--fold", type=int, choices=range(5), action="append")
args = parser.parse_args()
t.R2_H5AD = args.expression
root = args.run
args.output.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
for fold in (args.fold or range(5)):
    print('LOAD_FOLD',fold,flush=True)
    data=t.load_result2_fold(root,fold)
    outer=pd.read_csv(root/'04_SPLITS/PARTICIPANT_OUTER_FOLDS.tsv',sep='\t')
    held=outer.loc[outer.outer_fold.eq(fold),'sample_id_std'].astype(str).tolist()
    for seed in (20260828,20260829,20260830):
        ck=torch.load(root/f'08_OOF_MODELS/fold_{fold}/seed_{seed}/model.pt',map_location='cpu',weights_only=False)
        config = t.TrainConfig(**ck['train_config'])
        assert ck['gene_ids'] == data.gene_ids and ck['celltype_levels'] == data.celltype_levels
        model=TBStateSCV3(n_genes=5000,n_major_celltypes=len(data.celltype_levels),n_technical_batches=1,transition_names=t.TASKS,state_dim=config.state_dim,tau=config.tau,identity_dim=config.identity_dim,hidden_dim=config.hidden_dim,likelihood=config.likelihood).to(device)
        model.load_state_dict(ck['state_dict'],strict=True)
        pred,_,metrics,quality=t.evaluate_model(model,data,held,config,device,False,chunk_size=4096)
        pred.to_csv(args.output / f"fold_{fold}_seed_{seed}.tsv", sep="\t", index=False)
        del model, ck, pred
    del data
    gc.collect()
