"""Read-only distribution and participant-held-out tail calibration audit."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('inputs/TB_STATE_DISTRIBUTION_SELECTION')
SEED = 20260904
ALPHAS = (0.10, 0.05, 0.025, 0.01)
TAIL_ALPHAS = (0.05, 0.025, 0.01)


def _finite(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)[np.isfinite(x)]


def distributions() -> dict[str, object]:
    return {
        'GAUSSIAN': stats.norm,
        'WEIBULL': stats.weibull_min,
        'GAMMA': stats.gamma,
        'GENERALIZED_GAMMA': stats.gengamma,
        'LOGNORMAL': stats.lognorm,
        'EXPONENTIAL': stats.expon,
    }


def fit_distribution(name: str, values: np.ndarray) -> tuple[object, tuple[float, ...], str]:
    x = _finite(values)
    if x.size < 20 or np.unique(x).size < 3:
        raise ValueError('INSUFFICIENT_NONDEGENERATE_REFERENCE_VALUES')
    dist = distributions()[name]
    with np.errstate(all='ignore'):
        params = tuple(float(v) for v in dist.fit(x))
    if not all(np.isfinite(params)):
        raise ValueError('NONFINITE_MLE_PARAMETERS')
    return dist, params, 'SCIPY_MLE_FREE_LOCATION'


def qq_r2(dist: object, params: tuple[float, ...], values: np.ndarray) -> float:
    x = np.sort(_finite(values)); n = x.size
    theoretical = np.asarray(dist.ppf((np.arange(n) + .5) / n, *params), dtype=float)
    valid = np.isfinite(theoretical) & np.isfinite(x)
    if valid.sum() < 3 or np.nanstd(theoretical[valid]) == 0 or np.nanstd(x[valid]) == 0:
        return float('nan')
    return float(np.corrcoef(theoretical[valid], x[valid])[0, 1] ** 2)


def global_fit(axis: str, values: np.ndarray, result: str, score_source: str) -> pd.DataFrame:
    rows = [{'result': result, 'score_axis': axis, 'distribution': 'EMPIRICAL_ECDF',
             'fit_status': 'REFERENCE_ECDF', 'fit_method': 'EMPIRICAL', 'n_reference_cells': len(values),
             'log_likelihood': np.nan, 'number_parameters': 0, 'AIC': np.nan, 'BIC': np.nan,
             'KS_D': 0.0, 'KS_P': 1.0, 'QQ_R2': 1.0, 'CVM_STATISTIC': 0.0, 'CVM_P': np.nan,
             'score_source': score_source}]
    x = _finite(values)
    for name in distributions():
        base = {'result': result, 'score_axis': axis, 'distribution': name, 'n_reference_cells': len(x),
                'score_source': score_source}
        try:
            dist, params, method = fit_distribution(name, x)
            lp = np.asarray(dist.logpdf(x, *params), dtype=float)
            ll = float(np.log(np.clip(np.exp(lp), 1e-300, None)).sum())
            k = len(params); n = len(x)
            ks = stats.kstest(x, dist.cdf, args=params)
            cvm = stats.cramervonmises(x, lambda z: dist.cdf(z, *params))
            rows.append({**base, 'fit_status': 'PASS', 'fit_method': method, 'parameters_json': json.dumps(params),
                         'log_likelihood': ll, 'number_parameters': k, 'AIC': 2*k - 2*ll,
                         'BIC': math.log(n)*k - 2*ll, 'KS_D': float(ks.statistic), 'KS_P': float(ks.pvalue),
                         'QQ_R2': qq_r2(dist, params, x), 'CVM_STATISTIC': float(cvm.statistic),
                         'CVM_P': float(cvm.pvalue)})
        except Exception as exc:
            rows.append({**base, 'fit_status': 'FAIL', 'fit_method': 'SCIPY_MLE_FREE_LOCATION',
                         'parameters_json': '', 'log_likelihood': np.nan, 'number_parameters': np.nan,
                         'AIC': np.nan, 'BIC': np.nan, 'KS_D': np.nan, 'KS_P': np.nan, 'QQ_R2': np.nan,
                         'CVM_STATISTIC': np.nan, 'CVM_P': np.nan, 'error': f'{type(exc).__name__}:{exc}'})
    return pd.DataFrame(rows)


def _threshold(name: str, train: np.ndarray, alpha: float) -> float:
    if name == 'EMPIRICAL_ECDF':
        return float(np.quantile(train, 1-alpha, method='higher'))
    dist, params, _ = fit_distribution(name, train)
    return float(dist.ppf(1-alpha, *params))


def heldout_tail(axis: str, frame: pd.DataFrame, score_col: str, participant_col: str,
                 result: str, score_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = frame[[participant_col, score_col]].dropna().copy()
    base[participant_col] = base[participant_col].astype(str)
    base = base[np.isfinite(base[score_col].to_numpy(float))]
    participants = sorted(base[participant_col].unique())
    detail: list[dict[str, object]] = []
    for heldout in participants:
        test = base.loc[base[participant_col] == heldout, score_col].to_numpy(float)
        train = base.loc[base[participant_col] != heldout, score_col].to_numpy(float)
        for name in ('EMPIRICAL_ECDF', *distributions().keys()):
            for alpha in ALPHAS:
                row = {'result': result, 'score_axis': axis, 'distribution': name,
                       'heldout_participant': heldout, 'nominal_alpha': alpha,
                       'expected_fraction': alpha, 'n_cells': len(test), 'score_source': score_source}
                try:
                    threshold = _threshold(name, train, alpha)
                    observed = float(np.mean(test >= threshold))
                    detail.append({**row, 'threshold': threshold, 'observed_fraction': observed,
                                   'absolute_error': abs(observed-alpha), 'status': 'PASS'})
                except Exception as exc:
                    detail.append({**row, 'threshold': np.nan, 'observed_fraction': np.nan,
                                   'absolute_error': np.nan, 'status': 'FAIL',
                                   'error': f'{type(exc).__name__}:{exc}'})
    det = pd.DataFrame(detail)
    summary: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED)
    for (name, alpha), sub in det.loc[det.status == 'PASS'].groupby(['distribution','nominal_alpha'], sort=False):
        vals = sub.observed_fraction.to_numpy(float)
        boot = np.mean(rng.choice(vals, size=(2000, len(vals)), replace=True), axis=1)
        summary.append({'result': result, 'score_axis': axis, 'distribution': name, 'nominal_alpha': alpha,
                        'n_heldout_participants': len(vals), 'mean_observed_FPR': float(vals.mean()),
                        'median_observed_FPR': float(np.median(vals)), 'participant_weighted_FPR': float(vals.mean()),
                        'FPR_CI_low': float(np.quantile(boot,.025)), 'FPR_CI_high': float(np.quantile(boot,.975)),
                        'absolute_calibration_error': float(np.mean(np.abs(vals-alpha))),
                        'status': 'PASS', 'bootstrap_B': 2000, 'seed': SEED})
    return det, pd.DataFrame(summary)


def input_for(mode: str) -> tuple[str, pd.DataFrame, str, str, str, str]:
    inp = ROOT / 'inputs'
    if mode == 'R2':
        df = pd.read_csv(inp/'result2_oof_fold0.tsv.gz', sep='\t')
        # Two real Result2 disease-state axes use distinct reference populations.
        parts = []
        for axis, score, stage in [('T1_HC_to_TBINF','eta_T1_ensemble','HC'),
                                   ('T2_TBINF_to_ATB','eta_T2_ensemble','TBINF')]:
            z = df.loc[df.analysis_stage.astype(str).eq(stage), ['sample_id_std',score]].copy()
            z.columns = ['participant', 'score']
            z['axis'] = axis
            parts.append(z)
        return 'Result2', pd.concat(parts,ignore_index=True), 'axis', 'score', 'participant', 'OOF_CELL_SCORES_ENSEMBLE.tsv.gz'
    ref = pd.read_parquet(inp/'result3_ltbi_reference.parquet')
    mapping = {'U6':'consensus_6m','U12':'consensus_12m','UCOX':'consensus_cox'}
    return 'Result3', ref[['participant_id', mapping[mode]]].rename(columns={'participant_id':'participant', mapping[mode]:'score'}), mode, 'score', 'participant', 'LTBI_BALANCED_K500_PRIMARY.parquet'


def main(mode: str) -> None:
    result, frame, axis_key, score_col, participant_col, source = input_for(mode)
    out = (ROOT / 'results' / '02_RESULT2_DISTRIBUTION' if mode == 'R2'
           else ROOT / 'results' / '03_RESULT3_DISTRIBUTION' / mode)
    out.mkdir(parents=True, exist_ok=True)
    axes = sorted(frame[axis_key].unique()) if mode == 'R2' else [axis_key]
    globals_, details, summaries, rankings = [], [], [], []
    for axis in axes:
        sub = frame.loc[frame[axis_key] == axis] if mode == 'R2' else frame
        vals = sub[score_col].to_numpy(float)
        gf = global_fit(axis, vals, result, source)
        det, summ = heldout_tail(axis, sub, score_col, participant_col, result, source)
        param = gf.loc[(gf.distribution != 'EMPIRICAL_ECDF') & (gf.fit_status == 'PASS')].copy()
        if not param.empty:
            param['global_fit_rank'] = param.BIC.rank(method='min').astype(int)
        tail = summ.loc[summ.nominal_alpha.isin(TAIL_ALPHAS)].groupby('distribution',as_index=False).agg(
            tail_calibration_error=('absolute_calibration_error','mean'), n_tail_levels=('nominal_alpha','nunique'))
        tail['tail_calibration_rank'] = tail.tail_calibration_error.rank(method='min').astype(int)
        rank = tail.merge(param[['distribution','global_fit_rank','BIC']],how='left',on='distribution')
        rank['final_rank'] = rank.tail_calibration_rank.rank(method='min').astype(int)
        rank['result']=result; rank['score_axis']=axis
        rank['selected_distribution'] = rank.final_rank.eq(1)
        rank['selection_rule'] = 'PARTICIPANT_HELDOUT_UPPER_TAIL_CALIBRATION_P95_P975_P99_THEN_GLOBAL_BIC'
        globals_.append(gf); details.append(det); summaries.append(summ); rankings.append(rank)
    g=pd.concat(globals_,ignore_index=True); d=pd.concat(details,ignore_index=True); s=pd.concat(summaries,ignore_index=True); r=pd.concat(rankings,ignore_index=True)
    g.to_csv(out/'GLOBAL_DISTRIBUTION_FIT.tsv',sep='\t',index=False)
    d.to_csv(out/'PARTICIPANT_HELDOUT_TAIL_CALIBRATION.tsv',sep='\t',index=False)
    s.to_csv(out/'TAIL_CALIBRATION_SUMMARY.tsv',sep='\t',index=False)
    r.to_csv(out/'DISTRIBUTION_SELECTION_RANKING.tsv',sep='\t',index=False)
    tc=s.loc[s.nominal_alpha.isin((.10,.05,.025,.01))].copy()
    tc['threshold_name']=tc.nominal_alpha.map({.10:'P90',.05:'P95',.025:'P97.5',.01:'P99'})
    tc.to_csv(out/'THRESHOLD_CALIBRATION.tsv',sep='\t',index=False)
    winner=r.sort_values(['score_axis','final_rank','tail_calibration_error','global_fit_rank'],na_position='last').groupby('score_axis',as_index=False).first()
    winner.to_csv(out/'FINAL_DISTRIBUTION_SELECTION.tsv',sep='\t',index=False)
    (out/'STATUS.json').write_text(json.dumps({'status':'PASS','result':result,'mode':mode,
        'n_score_rows':int(len(frame)),'n_axes':len(axes),'seed':SEED,'no_outcome_used':True},indent=2)+'\n')
    print(json.dumps({'status':'PASS','result':result,'mode':mode,'axes':axes,'rows':len(frame)},sort_keys=True))


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in {'R2','U6','U12','UCOX'}:
        raise SystemExit('usage: distribution_runner.py R2|U6|U12|UCOX')
    main(sys.argv[1])
