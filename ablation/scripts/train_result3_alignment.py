from pathlib import Path
import os,sys,subprocess,json,hashlib,argparse
p=argparse.ArgumentParser();p.add_argument('--retraining',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--historical-source',type=Path,required=True);a=p.parse_args();D,O,P=a.retraining,a.output,sys.executable
SRC=a.historical_source
os.environ['PYTHONPATH']=str(Path(__file__).resolve().parents[1] / 'model' / 'result3');os.environ['OMP_NUM_THREADS']='3';os.environ['OPENBLAS_NUM_THREADS']='3'
def run(*args):
 print('EXEC',*map(str,args),flush=True);subprocess.run([P,'-u',*map(str,args)],check=True,env=os.environ)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for variant in ('NO_CORAL','NO_MMD'):
 for seed in (2026091801,2026091802,2026091803):
  base=O/'r3'/variant/f'seed_{seed}';base.mkdir(parents=True,exist_ok=False);temp=base/'temporary';align=temp/'alignment';record=[]
  for cohort in ('GSE79362','GSE94438','GSE107993_GSE107994_combined'):
   dst=align/f'heldout_{cohort}'
   run('-m','v126_dnra.train_alignment','--config',D/'config/contract.json','--anchors',D/f'03_ANCHORS/FORMAL_4FAMILY/FULL/seed_{seed}','--sc-checkpoint',D/f'01_SC_VAE/FORMAL_214402/FULL/seed_{seed}/best_sc_vae.pt','--heldout-cohort',cohort,'--variant',variant,'--seed',seed,'--output',dst)
   st=json.loads((dst/'STATUS.json').read_text());assert st['status']=='PASS' and not st['heldout_expression_used'] and not st['heldout_outcomes_used'];st.update({'checkpoint_sha256':sha(dst/'best_alignment.pt'),'variant':variant,'seed':seed});record.append(st)
  (base/'ALIGNMENT_AUDIT.json').write_text(json.dumps(record,indent=2))
  compat=temp/'compat';endpoint=temp/'endpoint_run'
  run('-m','v126_dnra.prepare_endpoint_engine','--config',D/'config/contract.json','--alignment-root',align,'--compat-root',compat,'--endpoint-run',endpoint)
  run('-m','v126_dnra.run_outer_endpoint','--historical-source',SRC,'--compat-root',compat,'--inner-run',D/f'05_ENDPOINT_INNER/FULL/seed_{seed}','--output',base/'results')
  status=json.loads((base/'results/STATUS.json').read_text());assert status['status']=='PASS'
  print('PAIR_FINISHED',variant,seed,flush=True)
print('ALIGNMENT_ABLATIONS_COMPLETE',flush=True)
