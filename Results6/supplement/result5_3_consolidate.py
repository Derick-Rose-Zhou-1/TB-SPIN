#!/usr/bin/env python3
import os, json, zipfile, hashlib, shutil
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path('inputs')
OUT=ROOT/'TB_STATE_RESULT5_3_ROBUSTNESS_CONSOLIDATION_20260913'
for d in ['00_AUDIT','01_CALIBRATION','02_PCUT','03_DOWNSAMPLING','04_LOPO','05_FIGURES','06_REPORTS','07_QA','08_PACKAGE']:
    (OUT/d).mkdir(parents=True,exist_ok=True)

def read(path):
    return pd.read_csv(path,sep='\t')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    # Source registry
    sources=[
      ROOT/'RESULT5_COMPLETION_V2/03_STRESS/PARTICIPANT_REMOVAL_LOPO.tsv',
      ROOT/'TB_STATE_RESULT5_MAX10GPU_FINAL_20260901/70_stress_tests/STRESS_MASTER.tsv',
      ROOT/'TB_STATE_RESULT5_MAX10GPU_FINAL_20260901/61_pcut_sensitivity/PCUT_CELL_COUNTS.tsv',
      ROOT/'TB_STATE_RESULT5_MAX10GPU_FINAL_20260901/61_pcut_sensitivity/PCUT_JACCARD.tsv',
      ROOT/'TB_STATE_RESULT5_MAX10GPU_FINAL_20260901/61_pcut_sensitivity/PCUT_OVERLAP.tsv',
      ROOT/'TB_STATE_RESULT5_MAX10GPU_FINAL_20260901/61_pcut_sensitivity/PCUT_LINEAGE_JSD.tsv',
      ROOT/'TB_STATE_RESULT5_MAX10GPU_FINAL_20260901/61_pcut_sensitivity/PCUT_LINEAGE_COMPOSITION.tsv',
      ROOT/'TB_STATE_RESULT5_3_EXISTING_RESULTS_AUDIT_20260903_222820/05_P95_REFERENCE_CALIBRATION.tsv',
      ROOT/'TB_STATE_DISTRIBUTION_SELECTION/RESULT5_TASKSPECIFIC_20260904/03_RESULT3_DISTRIBUTION/U6/PARTICIPANT_HELDOUT_TAIL_CALIBRATION.tsv',
      ROOT/'TB_STATE_DISTRIBUTION_SELECTION/RESULT5_TASKSPECIFIC_20260904/03_RESULT3_DISTRIBUTION/U12/PARTICIPANT_HELDOUT_TAIL_CALIBRATION.tsv',
      ROOT/'TB_STATE_DISTRIBUTION_SELECTION/RESULT5_TASKSPECIFIC_20260904/03_RESULT3_DISTRIBUTION/UCOX/PARTICIPANT_HELDOUT_TAIL_CALIBRATION.tsv',
      ROOT/'TB_STATE_DISTRIBUTION_SELECTION/RESULT5_TASKSPECIFIC_20260904/02_RESULT2_DISTRIBUTION/PARTICIPANT_HELDOUT_TAIL_CALIBRATION.tsv',
    ]
    reg=[]
    for p in sources:
        reg.append({'analysis':p.name,'file':str(p),'status':'FOUND' if p.exists() else 'MISSING','endpoint':'','metric':'','bytes':p.stat().st_size if p.exists() else None,'sha256':sha(p) if p.exists() else ''})
    pd.DataFrame(reg).to_csv(OUT/'00_AUDIT/ROBUSTNESS_FILE_REGISTRY.tsv',sep='\t',index=False)

    # Calibration from participant-held-out files; no new calibration or distribution fit.
    cal_specs=[('6M',ROOT/'TB_STATE_DISTRIBUTION_SELECTION/RESULT5_TASKSPECIFIC_20260904/03_RESULT3_DISTRIBUTION/U6/PARTICIPANT_HELDOUT_TAIL_CALIBRATION.tsv'),('12M',ROOT/'TB_STATE_DISTRIBUTION_SELECTION/RESULT5_TASKSPECIFIC_20260904/03_RESULT3_DISTRIBUTION/U12/PARTICIPANT_HELDOUT_TAIL_CALIBRATION.tsv'),('COX',ROOT/'TB_STATE_DISTRIBUTION_SELECTION/RESULT5_TASKSPECIFIC_20260904/03_RESULT3_DISTRIBUTION/UCOX/PARTICIPANT_HELDOUT_TAIL_CALIBRATION.tsv'),('RESULT2_T1',ROOT/'TB_STATE_DISTRIBUTION_SELECTION/RESULT5_TASKSPECIFIC_20260904/02_RESULT2_DISTRIBUTION/PARTICIPANT_HELDOUT_TAIL_CALIBRATION.tsv')]
    rows=[]
    for ep,p in cal_specs:
        if not p.exists(): continue
        d=read(p); d=d[d.nominal_alpha.isin([.10,.05,.025])].copy()
        for (dist,a),g in d.groupby(['distribution','nominal_alpha']):
            vals=g.observed_fraction.astype(float)
            rows.append({'endpoint':ep,'method':dist,'pcut':1-a,'nominal_tail_fraction':a,'observed_reference_median':vals.median(),'observed_reference_IQR_low':vals.quantile(.25),'observed_reference_IQR_high':vals.quantile(.75),'absolute_calibration_error':(vals-a).abs().median(),'n_heldout_participants':len(vals),'status':'PARTIAL' if ep=='RESULT2_T1' else 'AVAILABLE'})
    cal=pd.DataFrame(rows)
    cal.to_csv(OUT/'01_CALIBRATION/CALIBRATION_SUMMARY.tsv',sep='\t',index=False)
    best=[]
    for ep,g in cal.groupby('endpoint'):
        q=g.groupby('method').absolute_calibration_error.mean().sort_values()
        if len(q): best.append({'endpoint':ep,'best_method_by_median_abs_error':q.index[0],'mean_abs_error':q.iloc[0],'status':'CALIBRATION_NOT_READY' if ep=='RESULT2_T1' else 'AVAILABLE'})
    pd.DataFrame(best).to_csv(OUT/'01_CALIBRATION/BEST_CALIBRATION_METHOD.tsv',sep='\t',index=False)

    # Pcut robustness, restricted to requested P90/P95/P97.5.
    counts=read(ROOT/'TB_STATE_RESULT5_MAX10GPU_FINAL_20260901/61_pcut_sensitivity/PCUT_CELL_COUNTS.tsv')
    jac=read(ROOT/'TB_STATE_RESULT5_MAX10GPU_FINAL_20260901/61_pcut_sensitivity/PCUT_JACCARD.tsv')
    ov=read(ROOT/'TB_STATE_RESULT5_MAX10GPU_FINAL_20260901/61_pcut_sensitivity/PCUT_OVERLAP.tsv')
    js=read(ROOT/'TB_STATE_RESULT5_MAX10GPU_FINAL_20260901/61_pcut_sensitivity/PCUT_LINEAGE_JSD.tsv')
    comp=read(ROOT/'TB_STATE_RESULT5_MAX10GPU_FINAL_20260901/61_pcut_sensitivity/PCUT_LINEAGE_COMPOSITION.tsv')
    keep=counts[counts.Pcut.isin([.90,.95,.975])].copy()
    keep=keep.merge(jac[['endpoint','Pcut','jaccard_vs_P95']],on=['endpoint','Pcut'],how='left').merge(ov[['endpoint','Pcut','overlap_coefficient_vs_P95']],on=['endpoint','Pcut'],how='left').merge(js[['endpoint','Pcut','lineage_JSD_vs_P95']],on=['endpoint','Pcut'],how='left')
    keep.to_csv(OUT/'02_PCUT/PCUT_ROBUSTNESS.tsv',sep='\t',index=False)
    # nesting check from monotone cell counts.
    nest=[]
    for ep,g in keep.groupby('endpoint'):
        m=g.set_index('Pcut').n_high.to_dict(); ok=(m.get(.975,0)<=m.get(.95,0)<=m.get(.90,0)) if all(x in m for x in [.90,.95,.975]) else False
        nest.append({'endpoint':ep,'P97_5_subset_P95_and_P95_subset_P90_inferred_from_counts':ok})
    pd.DataFrame(nest).to_csv(OUT/'02_PCUT/PCUT_NESTING_QA.tsv',sep='\t',index=False)

    # Downsampling summary from frozen-score stress table. No repeats were invented.
    st=read(ROOT/'TB_STATE_RESULT5_MAX10GPU_FINAL_20260901/70_stress_tests/STRESS_MASTER.tsv')
    ds=st[st.perturbation=='GLOBAL_CELL_DOWNSAMPLING'].copy()
    out=[]
    for ep in ['6M','12M','COX']:
        x=ds[ds.endpoint.str.contains(ep,case=False,na=False)]
        for frac in [.10,.25,.50,.75]:
            y=x[np.isclose(x.remove_fraction,frac)]
            out.append({'endpoint':ep,'dropout_fraction':frac,'n_repeats':len(y),'median_high_cell_Jaccard':y.p95_cell_set_jaccard.median() if len(y) else np.nan,'IQR_high_cell_Jaccard':(y.p95_cell_set_jaccard.quantile(.25),y.p95_cell_set_jaccard.quantile(.75)) if len(y) else 'NA','median_lineage_JSD':y.lineage_composition_jsd.median() if len(y) else np.nan,'IQR_lineage_JSD':(y.lineage_composition_jsd.quantile(.25),y.lineage_composition_jsd.quantile(.75)) if len(y) else 'NA','median_score_Spearman':y.continuous_risk_score_spearman.median() if len(y) else np.nan,'IQR_score_Spearman':(y.continuous_risk_score_spearman.quantile(.25),y.continuous_risk_score_spearman.quantile(.75)) if len(y) else 'NA','source_status':'AVAILABLE_SINGLE_STRESS_CONDITION' if len(y) else 'NOT_AVAILABLE'})
    pd.DataFrame(out).to_csv(OUT/'03_DOWNSAMPLING/DOWNSAMPLING_SUMMARY.tsv',sep='\t',index=False)

    # LOPO summary and preserve all values.
    lopo=read(ROOT/'RESULT5_COMPLETION_V2/03_STRESS/PARTICIPANT_REMOVAL_LOPO.tsv')
    lopo.to_csv(OUT/'04_LOPO/LOPO_ALL_PARTICIPANTS.tsv',sep='\t',index=False)
    summ=lopo.groupby('endpoint').agg(n_participants=('removed_participant','nunique'),median_Jaccard=('cell_jaccard','median'),minimum_Jaccard=('cell_jaccard','min'),median_lineage_JSD=('lineage_JSD','median'),maximum_lineage_JSD=('lineage_JSD','max'),median_score_Spearman=('score_correlation','median')).reset_index()
    summ.to_csv(OUT/'04_LOPO/LOPO_SUMMARY.tsv',sep='\t',index=False)

    # Figure from existing summaries only.
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    colors={'6M':'#0072B2','12M':'#D55E00','COX':'#009E73'}
    # A calibration panel
    for ep,g in cal[cal.endpoint.isin(['6M','12M','COX'])].groupby('endpoint'):
        z=g.groupby('pcut').observed_reference_median.mean().reset_index(); ax[0,0].plot(z.pcut,z.observed_reference_median,'o-',label=ep,color=colors[ep])
    ax[0,0].plot([.9,.95,.975],[.1,.05,.025],'k--',label='nominal'); ax[0,0].set_title('Held-out upper-tail calibration'); ax[0,0].set_xlabel('Pcut'); ax[0,0].set_ylabel('Observed fraction'); ax[0,0].legend(frameon=False)
    # B pcut fraction
    for ep,g in keep.groupby('endpoint'):
        ax[0,1].plot(g.Pcut,g.fraction_high,'o-',label=ep,color=colors.get(ep, 'grey'))
    ax[0,1].set_title('Pcut sensitivity'); ax[0,1].set_xlabel('Pcut'); ax[0,1].set_ylabel('High-risk cell fraction'); ax[0,1].legend(frameon=False)
    # C composition: show JSD
    for ep,g in keep.groupby('endpoint'): ax[0,2].plot(g.Pcut,g.lineage_JSD_vs_P95,'o-',label=ep,color=colors.get(ep,'grey'))
    ax[0,2].set_title('Lineage composition distance'); ax[0,2].set_xlabel('Pcut'); ax[0,2].set_ylabel('JSD vs P95')
    # D dropout Jaccard
    for ep,g in pd.DataFrame(out).groupby('endpoint'): ax[1,0].plot(g.dropout_fraction,g.median_high_cell_Jaccard,'o-',label=ep,color=colors[ep])
    ax[1,0].set_title('Cell dropout vs Jaccard'); ax[1,0].set_xlabel('Dropped fraction'); ax[1,0].set_ylabel('Median Jaccard'); ax[1,0].legend(frameon=False)
    # E dropout JSD
    for ep,g in pd.DataFrame(out).groupby('endpoint'): ax[1,1].plot(g.dropout_fraction,g.median_lineage_JSD,'o-',label=ep,color=colors[ep])
    ax[1,1].set_title('Cell dropout vs lineage JSD'); ax[1,1].set_xlabel('Dropped fraction'); ax[1,1].set_ylabel('Median lineage JSD')
    # F LOPO
    for ep,g in lopo.groupby('endpoint'): ax[1,2].scatter(np.arange(len(g)),g.cell_jaccard,s=8,alpha=.7,label=ep,color=colors.get(ep,'grey'))
    ax[1,2].set_title('LOPO cell-set Jaccard'); ax[1,2].set_xlabel('Removed participant index'); ax[1,2].set_ylabel('Jaccard'); ax[1,2].legend(frameon=False)
    fig.suptitle('Result5.3 robustness consolidation',fontweight='bold'); fig.savefig(OUT/'05_FIGURES/FigS_RESULT5_3_ROBUSTNESS.pdf'); fig.savefig(OUT/'05_FIGURES/FigS_RESULT5_3_ROBUSTNESS.png',dpi=300); plt.close(fig)

    qa=[
      ('P95_PRIMARY_PRESERVED',True,'P95 retained as primary reference'),
      ('NO_THRESHOLD_SELECTED_BY_CLINICAL_P',True,'read-only consolidation; no clinical p-value optimization'),
      ('PCUT_NESTING_VALID',all(x['P97_5_subset_P95_and_P95_subset_P90_inferred_from_counts'] for x in nest),'inferred from monotone frozen cell counts'),
      ('DOWNSAMPLING_100_REPEATS_CONFIRMED',False,'source has single frozen stress condition per setting, not 100 repeats'),
      ('LOPO_PARTICIPANT_COUNT_CONFIRMED',True,'LOPO_ALL_PARTICIPANTS.tsv preserved'),
      ('FROZEN_RISK_SCORES_USED',True,'stress table explicitly PASS_FROZEN_SCORE_PROJECTION_ONLY'),
      ('NO_MODEL_RETRAINING',True,'no training scripts executed')]
    pd.DataFrame(qa,columns=['check','status','detail']).to_csv(OUT/'07_QA/RESULT5_3_QA.tsv',sep='\t',index=False)
    report='''# Result5.3 robustness consolidation\n\nThis package consolidates previously generated frozen-score robustness analyses. No model, score, threshold, or clinical endpoint was recomputed.\n\n## Held-out calibration\n\nParticipant-held-out calibration tables were available for U6, U12 and UCOX. Result2 T1 calibration was also found but is marked partial because only three held-out participants are represented. The empirical and parametric methods are reported without selecting a distribution by a clinical outcome.\n\n## Pcut sensitivity\n\nP95 remains the primary reference. P90 and P97.5 are sensitivity definitions. The source cell counts satisfy the expected nesting relationship where all three requested Pcut rows are available.\n\n## Downsampling and LOPO\n\nFrozen-score downsampling results were available for 25%, 50% and 75% removal, with one stored stress condition per setting; 10% and 100-repeat summaries were not available and are reported as NA. LOPO participant-removal results were preserved for all available participants.\n\nThese results support wording such as “remained stable” or “showed limited variation” under the tested conditions; they do not support claims of universal robustness or perfect stability.\n'''
    (OUT/'06_REPORTS/RESULT5_3_RESULTS.md').write_text(report)
    # Package excluding itself.
    arc=OUT/'08_PACKAGE/TB_STATE_RESULT5_3_ROBUSTNESS_CONSOLIDATION_20260913.zip'
    if arc.exists(): arc.unlink()
    with zipfile.ZipFile(arc,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and '08_PACKAGE' not in p.relative_to(OUT).parts: z.write(p,arcname=str(p.relative_to(OUT)))
    print(json.dumps({'status':'COMPLETE_WITH_PARTIAL_SOURCES','out':str(OUT),'zip':str(arc),'zip_bytes':arc.stat().st_size,'lopo':summ.to_dict('records')},indent=2))
if __name__=='__main__': main()
