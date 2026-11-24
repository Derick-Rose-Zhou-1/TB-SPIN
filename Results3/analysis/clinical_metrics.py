"""Recompute discrimination and Cox associations from frozen outer predictions.

The archived verification used 5,000 participant bootstraps. The manuscript
specifies 2,000; --resamples selects the count. The figure renderer reads its
locked interval table separately.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

def endpoint_name(x: str) -> str:
    return {"E_ALL": "H1", "E_6M": "H2", "E_12M": "H3", "E_COX": "H1_COX"}.get(str(x), str(x))


def weighted_metrics(y, score, w):
    y = np.asarray(y, int); score = np.asarray(score, float); w = np.asarray(w, float)
    ok = np.isfinite(y) & np.isfinite(score) & np.isfinite(w) & (w > 0)
    y, score, w = y[ok], score[ok], w[ok]
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    return (roc_auc_score(y, score, sample_weight=w),
            average_precision_score(y, score, sample_weight=w),
            float(np.sum(w * y) / np.sum(w)))


def cluster_bootstrap(df: pd.DataFrame, b):
    ids = pd.unique(df["participant_id"].astype(str))
    rng = np.random.default_rng(20260828)
    aucs, aprs = [], []
    # Resampling a participant k times is exactly equivalent to multiplying every
    # row weight for that participant by k. This avoids repeatedly concatenating
    # thousands of small data frames and preserves the participant-cluster design.
    id_to_i = {x:i for i,x in enumerate(ids)}
    row_i = df["participant_id"].astype(str).map(id_to_i).to_numpy()
    y = pd.to_numeric(df.y_true, errors="coerce").to_numpy()
    score = pd.to_numeric(df.score, errors="coerce").to_numpy()
    base_w = pd.to_numeric(df.participant_metric_weight, errors="coerce").fillna(1).to_numpy()
    for _ in range(b):
        counts = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)))
        a, p, _ = weighted_metrics(y, score, base_w * counts[row_i])
        if np.isfinite(a): aucs.append(a)
        if np.isfinite(p): aprs.append(p)
    ci = lambda x: (float(np.quantile(x, .025)), float(np.quantile(x, .975))) if x else (np.nan, np.nan)
    return ci(aucs), ci(aprs), len(aucs)


def boolish(s):
    return str(s).strip().lower() in {"1", "true", "yes", "y"}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path, help="RESULTS/V126_RUN from the release")
parser.add_argument("output", type=Path)
parser.add_argument("--resamples",type=int,default=2000)
args=parser.parse_args()
RUN=args.run
args.output.mkdir(parents=True,exist_ok=True)
pred=pd.read_csv(RUN/'06_OUTER_EVALUATION/V126_OUTER_PREDICTIONS.tsv',sep='\t')
rows=[]
for (ep,co), d in pred[pred.endpoint.ne('E_COX')].groupby(['endpoint','outer_heldout']):
    auc,_,_=weighted_metrics(d.y_true,d.score,d.participant_metric_weight)
    bounds,_,valid=cluster_bootstrap(d,args.resamples)
    rows.append(dict(endpoint=endpoint_name(ep),cohort=co,AUROC=auc,ci_low=bounds[0],
                     ci_high=bounds[1],bootstrap_valid=valid,n=d.participant_id.nunique()))
pd.DataFrame(rows).to_csv(args.output/'auc.tsv',sep='\t',index=False)
man=pd.read_csv(RUN/'02_ENDPOINTS/PARTICIPANT_ENDPOINTS.tsv',sep='\t')
p=pred[pred.endpoint.eq('E_COX')].copy(); m=man[man.endpoint.eq('E_COX')].copy()
rows=[]
for co,d in p.groupby('outer_heldout'):
    mm=m[m.cohort.eq(co)].copy()
    if 'baseline' in mm and mm.baseline.map(boolish).any(): mm=mm[mm.baseline.map(boolish)]
    else: mm=mm.sort_values(['participant_id','sample_date','visit_number'],na_position='last').groupby('participant_id').head(1)
    z=d.merge(mm[['participant_id','GSM','time_to_event_or_censor_days','y_true']],on=['participant_id','GSM'],suffixes=('','_m'))
    z=z.sort_values('participant_id').groupby('participant_id').head(1)
    z['duration']=pd.to_numeric(z.time_to_event_or_censor_days,errors='coerce'); z['event']=pd.to_numeric(z.y_true_m,errors='coerce')
    z['score_z']=(z.score-z.score.mean())/z.score.std(ddof=0)
    q=z[['duration','event','score_z']].replace([np.inf,-np.inf],np.nan).dropna()
    rec=dict(endpoint='H1',cohort=co,N=len(q),events=int(q.event.sum()),source_file="V126_OUTER_PREDICTIONS.tsv")
    try:
        c=CoxPHFitter().fit(q,duration_col='duration',event_col='event')
        sm=c.summary.loc['score_z']; rec.update(HR_per_SD=float(sm['exp(coef)']),CI_low=float(sm['exp(coef) lower 95%']),CI_high=float(sm['exp(coef) upper 95%']),P=float(sm['p']),Harrell_C=float(concordance_index(q.duration,-q.score_z,q.event)),status='OK')
    except Exception as e: rec.update(HR_per_SD=np.nan,CI_low=np.nan,CI_high=np.nan,P=np.nan,Harrell_C=np.nan,status='NE:'+str(e))
    rec.update(Uno_C=np.nan,tdAUC_180=np.nan,tdAUC_365=np.nan,Uno_status='NOT_ESTIMABLE_WITHOUT_VALID_TRAIN_CENSORING_REFERENCE')
    rows.append(rec)
pd.DataFrame(rows).to_csv(args.output/'survival.tsv',sep='\t',index=False)
