"""Score frozen Fig. 5 encoders using Fig. 2 Fisher coordinates and reference fits."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import sys, os, json, hashlib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import fig2_reference as ref

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
TASKS = {'B1':'DTB','B2':'TPE','B3':'TBM'}
def coordinates(run):
    cols=['cell_id','sample_id_std','analysis_stage','celltype_raw']+[f'z_state_{i:02d}' for i in range(1,9)]
    records={p:h for h,p in (line.split() for line in (HERE/'candidate_weights.sha256').read_text().splitlines())}
    jobs=[]
    for fold in range(4):
        rel=f'FIG5_LOCKED100_20261116/fold_{fold}/seed_{run}'
        path=BASE/rel
        assert hashlib.sha256((path/'model.pt').read_bytes()).hexdigest()==records[rel+'/model.pt']
        split=json.loads((path.parent/'SELECTION.json').read_text())
        dev,held=[pd.read_csv(path/(key+'_latent.tsv.gz'),sep='\t',usecols=cols) for key in ['training','heldout']]
        assert set(dev.sample_id_std)==set(split['train_participants'])
        assert set(held.sample_id_std)==set(split['test_participants'])
        assert set(dev.sample_id_std).isdisjoint(held.sample_id_std)
        assert dev.cell_id.is_unique and held.cell_id.is_unique and len(dev)+len(held)==248054
        z=[c for c in dev if c.startswith('z_state_')]
        assert len(z)==8
        for task,target in TASKS.items():
            for lineage in sorted(dev.celltype_raw.unique()):
                g=dev[dev.analysis_stage.isin(['ATB',target]) & dev.celltype_raw.eq(lineage)]
                g=g[g.groupby('sample_id_std').cell_id.transform('size').ge(20)]
                m=g.groupby(['sample_id_std','analysis_stage'],observed=True)[z].median().reset_index()
                a,b=[m.loc[m.analysis_stage.eq(k),z].to_numpy(float) for k in ['ATB',target]]
                h=held[held.analysis_stage.isin(['ATB',target]) & held.celltype_raw.eq(lineage)].copy()
                h=h[['cell_id','sample_id_std','analysis_stage','celltype_raw']+z]
                h['fold']=fold;h['task']=task
                refs=g.loc[g.analysis_stage.eq('ATB'),['sample_id_std']+z].copy()
                if min(len(a),len(b))>=3:
                    u,v=a.mean(0),b.mean(0)
                    cov=LedoitWolf(assume_centered=True).fit(np.vstack([a-u,b-v])).covariance_
                    w=np.linalg.solve(cov,v-u); intercept=-.5*float((u+v)@w)
                    d=refs[z].to_numpy(float)@w+intercept
                    shift=np.median(d)
                    refs['Y']=np.exp(np.clip(d-shift,-80,80))
                    h['Y']=np.exp(np.clip(h[z].to_numpy(float)@w+intercept-shift,-80,80))
                jobs.append((run,fold,task,lineage,refs.drop(columns=z),h.drop(columns=z),len(a),len(b)))
        print('encoded fold',fold,flush=True)
    return jobs

def bootstrap_reference(family,values,observed,rs):
    import warnings
    from scipy.optimize import brentq
    from scipy.special import logsumexp
    from scipy.stats import kstest
    dist=ref.DISTROS[family]
    pars=dist.fit(values,floc=0.)
    rng=np.random.default_rng(rs)
    exceed=0
    fallback_count=0
    for i in range(500):
        sample=np.maximum(dist.rvs(*pars,size=len(values),random_state=rng),np.finfo(float).tiny)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                b=dist.fit(sample,floc=0.)
            stat=kstest(sample,dist.cdf,args=b).statistic
            if not np.isfinite(stat):raise FloatingPointError('invalid KS')
        except (ValueError, RuntimeError, FloatingPointError, OverflowError):
            if family!='WEIBULL' or not np.isfinite(sample).all():raise
            # Fixed-location Weibull maximum likelihood, solved in log space.
            v=np.log(sample)
            u=v-v.max()
            def profile(k):
                w=np.exp(k*u)
                return 1/k+v.mean()-v.max()-np.dot(w,u)/w.sum()
            upper=1.
            while profile(upper)>0:upper*=2
            k=brentq(profile,1e-9,upper)
            logscale=v.max()+(logsumexp(k*u)-np.log(len(u)))/k
            p=-np.expm1(-np.exp(np.clip(k*(np.sort(v)-logscale),-745,700)))
            j=np.arange(len(p))/len(p)
            stat=max(np.max(j+1/len(p)-p),np.max(p-j))
            fallback_count+=1
        exceed+=int(stat>=observed)
    return (1+exceed)/501,fallback_count

def classify(job):
    run,fold,task,lineage,r,h,n0,n1=job
    info=dict(run=run,fold=fold,task=task,lineage=lineage,reference_patients=n0,target_patients=n1)
    h['R']=np.nan
    if 'Y' not in r: return info|{'status':'LOW_SUPPORT'},h
    n=int(min(2000,r.groupby('sample_id_std').size().min()))
    fits={k:[] for k in ref.DISTROS}
    for i in range(20):
        y=ref.balanced_sample(r,n,ref.stable_seed('balance',fold,run,task,lineage,i))
        if i==0: primary=y
        for family in fits:
            fits[family].append(ref.fit_dist(family,y))
    order=sorted(fits,key=lambda k:np.median([x[1]['BIC'] for x in fits[k]]))
    # Equivalent winner to the original pass-first, minimum-median-BIC rule.
    pvals={};fallbacks=0
    for family in order:
        pars,diag=fits[family][0]
        p,n_fallback=bootstrap_reference(family,primary,diag['KS'],ref.stable_seed('bootstrap',fold,run,task,lineage,family));fallbacks+=n_fallback
        pvals[family]=p
        if p>=.05:break
    else:family=order[0]
    pars=fits[family][0][0]
    h['R']=ref.DISTROS[family].cdf(h.Y.to_numpy(float),*pars)
    assert np.isfinite(h.R).all()
    return info|dict(status='PASS',bootstrap_logspace_refits=fallbacks,winner=family,parametric_fit_warning=max(pvals.values())<.05,parameters=json.dumps(pars),ks_p_checked=json.dumps(pvals),median_bic_order=json.dumps(order),cells_per_reference_participant=n),h

if __name__=='__main__':
    run=int(sys.argv[1])
    assert run in (20260828,20260829,20260830)
    jobs=coordinates(run)
    # Children use CPU only; no subsequent CUDA calls in the children.
    tables=[]; diagnostics=[]
    with ProcessPoolExecutor(max_workers=min(6,len(os.sched_getaffinity(0))),mp_context=mp.get_context('fork')) as pool:
        for future in as_completed([pool.submit(classify,j) for j in jobs]):
            info,table=future.result();diagnostics.append(info);tables.append(table)
            print('completed',len(tables),'/',len(jobs),info,flush=True)
    allcells=pd.concat(tables,ignore_index=True)
    assert not allcells.duplicated(['cell_id','task']).any()
    allcells['risk']=allcells.R.ge(.95).astype('boolean').mask(allcells.R.isna())
    allcells.to_csv(HERE/f'heldout_{run}.tsv.gz',sep='\t',index=False)
    pd.DataFrame(diagnostics).sort_values(['fold','task','lineage']).to_csv(HERE/f'fits_{run}.tsv',sep='\t',index=False)

