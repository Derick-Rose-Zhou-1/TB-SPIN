"""QC Fig.5 training only; V3 backbone, original R6 inner-AUPRC selection."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,hashlib,gc
import numpy as np,pandas as pd,h5py,torch
import tb_state_sc_v3_training as training
import locked_core as core

root=Path(__file__).resolve().parents[1]
prep=root/'FIG5_PREP';outroot=root/'FIG5_TRAINING'
seeds=[20260828,20260829,20260830]
parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
torch.set_num_threads(4)
training.TASKS=core.TASKS;training.STAGE_ORDINAL={'ATB':0,'DTB':1,'TPE':2,'TBM':3}
lock=json.loads((prep/'PREFLIGHT.json').read_text());assert lock['status']=='INPUT_AND_SPLIT_PREFLIGHT_PASS'
for seed in seeds:assert json.loads((root/f'17_FINAL_FULL_MODEL/seed_{seed}/DONE.json').read_text())['status']=='TRAINING_COMPLETE_CALIBRATION_NOT_RUN'
meta=pd.read_csv(prep/'MODEL_READY_METADATA.tsv.gz',sep='\t',dtype=str)
genes=pd.read_csv(prep/'HVG5000_ORDERED_POSITIONS.tsv',sep='\t')
with h5py.File(lock['path']) as f:
 assert np.array_equal(training.dataframe_index(f['obs']).to_numpy(str),meta.cell_id.to_numpy(str))
 counts=training.load_csr(f['layers/counts'])[:,genes.h5ad_position.to_numpy(int)].tocsr()
celltype=pd.Categorical(meta.celltype_raw);freq=np.bincount(celltype.codes,minlength=len(celltype.categories)).astype(float)
iw=freq.sum()/np.maximum(freq,1);iw/=iw.mean();iw=np.minimum(iw,5);iw/=iw.mean()
data=training.PreparedData(counts=counts,metadata=meta,gene_ids=genes.gene_id.tolist(),participant_col='sample_id_std',participant_to_cells={str(k):np.asarray(v) for k,v in meta.groupby('sample_id_std').indices.items()},participant_to_stage=meta.groupby('sample_id_std').analysis_stage.first().to_dict(),celltype_codes=celltype.codes.astype(np.int64),celltype_levels=celltype.categories.astype(str).tolist(),identity_weights=iw.astype(np.float32))
baseck=torch.load(root/'17_FINAL_FULL_MODEL/seed_20260829/model.pt',map_location='cpu',weights_only=True)
config=training.TrainConfig(**{**baseck['train_config'],'lambda_ordinal':0.0})
assert data.gene_ids==baseck['gene_ids']
device=torch.device('cuda');assert torch.cuda.is_available()
