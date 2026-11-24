"""Assemble Fig. 2 using the unchanged source panels and verified molecular tables."""
from io import BytesIO
import textwrap
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model/result2"))
from burden_permutation import stable_seed, permutation_p
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from recompute import BASE, OUT, signature

plt.rcParams.update({'font.family':'Arial','font.size':7,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.linewidth':.6,'xtick.major.width':.5,'ytick.major.width':.5})

def put(page,fig,rect):
    buf=BytesIO()
    fig.savefig(buf,format='pdf')
    plt.close(fig)
    with fitz.open(stream=buf.getvalue(),filetype='pdf') as sub:
        page.show_pdf_page(fitz.Rect(rect),sub,0,keep_proportion=False)

def volcanos(page,inf,dis,bar):
    fig,axes=plt.subplots(1,2,figsize=(4.0,2.65))
    fig.subplots_adjust(left=.17,right=.86,bottom=.19,top=.80,wspace=.33)
    for ax,t,stage in zip(axes,[inf,dis],['TB infection','TB disease']):
        x=100*(t.pct_risk-t.pct_background)
        y=np.sign(t.log2FC)*(-np.log10(t.FDR.clip(lower=1e-310)))
        lim=np.abs(y)>300
        norm=TwoSlopeNorm(vmin=-4,vcenter=0,vmax=4)
        for mask,mk in [(~lim,'o'),(lim & (y>0),'^'),(lim & (y<0),'v')]:
            dots=ax.scatter(x[mask],y[mask].clip(-300,300),c=t.log2FC[mask].clip(-4,4),
                            cmap='RdBu_r',norm=norm,marker=mk,s=3,alpha=.8,linewidth=0,rasterized=True)
        ax.axvline(0,lw=.5,ls='--',color='.55')
        for q in [-2,2]:ax.axhline(q,lw=.5,ls='--',color='.55')
        # Draw selected expression-panel genes with labels at fixed, spaced rows.
        selected=bar.loc[bar.stage==stage,'gene'].unique()
        for k,g in enumerate(selected):
            ix=t.index[t.gene.eq(g)][0]
            ax.annotate(g,(x[ix],np.clip(y[ix],-300,300)),xytext=(.04,.72-.12*k),
                        textcoords='axes fraction',fontsize=5.8,fontstyle='italic',
                        arrowprops={'arrowstyle':'-','color':'.4','lw':.35})
        ax.set_ylim(-328,340)
        ax.set_title(stage+' risk vs background',fontsize=6.8)
        ax.text(.02,.96,'Risk up: '+str(len(signature(t))),transform=ax.transAxes,va='top',fontsize=6.6)
        ax.tick_params(labelsize=6.4,length=2)
    axes[0].set_ylabel(r'Signed $-\log_{10}$(FDR)')
    axes[1].tick_params(labelleft=False)
    fig.text(.5,.025,'Difference in cells expressing gene (percentage points)',ha='center',fontsize=6.8)
    cb=fig.colorbar(dots,cax=fig.add_axes([.91,.29,.018,.34]));cb.ax.set_title(r'$\log_2$FC',fontsize=6)
    fig.text(.5,.935,'Dashed lines: FDR = 0.01     Triangles: |value| > 300',ha='center',fontsize=6.2)
    put(page,fig,(114,111,333,242))

def bars(page,bar):
    fig,axes=plt.subplots(2,3,figsize=(3.2,2.8))
    fig.subplots_adjust(left=.15,right=.99,bottom=.21,top=.84,wspace=.70,hspace=1.35)
    for i,(stage,color) in enumerate([('TB infection','#557c9f'),('TB disease','#d86a63')]):
        x=bar[bar.stage==stage]
        for ax,(gene,v) in zip(axes[i],x.groupby('gene',sort=False)):
            v=v.set_index('group').loc[['Background','Risk']]
            ax.bar([0,1],v['mean'],yerr=v.ci95,color=['#b5b5b5',color],width=.63,
                   error_kw={'elinewidth':.65,'capsize':1.8},edgecolor='#333',linewidth=.4)
            ax.set_title(gene,fontstyle='italic',fontsize=7,pad=1)
            ax.set_ylim(0,max(v['mean']+v.ci95)*1.25)
            ax.set_xticks([0,1],['Background','Risk'] if i==1 else [],rotation=40,ha='right')
            ax.tick_params(labelsize=6.5,length=2)
            if v.raw_pvalue.iloc[0]<.001:ax.text(.5,.76,'***',ha='center',transform=ax.transAxes,fontsize=7)
        axes[i,0].set_ylabel('Expression',fontsize=7)
    for y,label in [(.967,'TB infection'),(.51,'TB disease')]:
        fig.text(.52,y,label+' risk cells vs background',fontsize=7.5,ha='center',weight='bold')
    put(page,fig,(335,111,504,242))

def go(page):
    t=pd.read_csv(OUT/'T1_infection_GO_BP.tsv',sep='\t')
    a=t[t.FDR<.05].head(6).copy()
    a['label']=a.term.str.removeprefix('GOBP_').str.replace('_',' ').str.lower()
    old=pd.read_csv(BASE/'go/FIG2D_GO_SELECTED_20.tsv',sep='\t')
    b=old[old.stage.str.contains('ATB')].copy().rename(columns={'display_label':'label','p.adjust':'FDR'})
    disease = pd.read_csv(OUT/'T2_disease_GO_BP.tsv', sep='\t').set_index('term')
    b['FDR'] = disease.loc[b.term, 'FDR'].to_numpy()
    fig,axes=plt.subplots(1,2,figsize=(5.1,3.1))
    fig.subplots_adjust(left=.31,right=.98,bottom=.15,top=.87,wspace=2.0)
    for ax,df,col,title in zip(axes,[a,b],['#567e9e','#539988'],['TB infection','TB disease']):
        y=np.arange(len(df));x=-np.log10(df.FDR)
        ax.hlines(y,0,x,color=col,lw=1);ax.scatter(x,y,color=col,s=14)
        ax.set_yticks(y,['\n'.join(textwrap.wrap(str(z),30)) for z in df.label],fontsize=6.1)
        ax.tick_params(length=2,pad=2);ax.invert_yaxis();ax.set_xlim(left=0)
        ax.set_xlabel(r'$-\log_{10}$(FDR)',fontsize=7);ax.set_title(title,fontsize=8,weight='bold')
    put(page,fig,(0,259,245,381))

def burdens(page):
    t=pd.read_csv(BASE/'burden/PANEL_C_PARTICIPANT_P95_BURDEN.tsv',sep='\t')
    fig,axes=plt.subplots(1,2,figsize=(2.2,2.55))
    fig.subplots_adjust(left=.24,right=.985,bottom=.19,top=.78,wspace=.55)
    rng=np.random.default_rng(20260831)
    for ax,(task,stages,title,col) in zip(axes,[
        ('T1',['HC','TBINF'],'HC vs infection',['#aaaaaa','#6487a6']),
        ('T2',['TBINF','ATB'],'Infection vs disease',['#6487a6','#d86a63'])]):
        ordered = t[t.task.eq(task)].sort_values('sample_id_std')
        groups = [ordered.loc[ordered.analysis_stage.eq(s), 'burden_pct'].to_numpy() for s in stages]
        p = permutation_p(*groups, stable_seed(task, 'OVERALL', 'P95'))
        for j,(stage,c) in enumerate(zip(stages,col)):
            sub=t[(t.task==task)&(t.analysis_stage==stage)]
            v=sub.burden_pct.to_numpy();assert len(v)
            ax.boxplot([v],positions=[j],widths=.55,patch_artist=True,showfliers=False,
                       boxprops={'facecolor':c,'linewidth':.6},medianprops={'color':'black'})
            ax.scatter(j+rng.uniform(-.09,.09,len(v)),v,s=4.5,zorder=3,
                       facecolor='white',edgecolor='.25',linewidth=.4)
            ax.text(j,.98,f'{100*sub.risk_n.sum()/sub.eligible_n.sum():.2f}%',ha='center',
                    va='top',transform=ax.get_xaxis_transform(),fontsize=6)
        ax.set_ylim(0,10 if task=='T1' else 50)
        ax.set_xticks([0,1],[{'TBINF':'TB\ninfection','HC':'HC','ATB':'TB\ndisease'}[s] for s in stages],rotation=30,ha='right')
        ax.tick_params(labelsize=6,length=2);ax.set_title(title+'\n'+f'$P$ = {p:.4f}',fontsize=6.2)
    axes[0].set_ylabel('Risk cells (%)',fontsize=7)
    put(page,fig,(0,111,114,242))

def roc_panels(page):
    curves = pd.read_csv(OUT / "roc_coordinates.tsv", sep="\t")
    metrics = pd.read_csv(OUT / "cohort_auc.tsv", sep="\t")
    auc = metrics.loc[metrics.normalization.eq("FSQN")].set_index("dataset").auc
    panels = [("RNA-seq", ["GSE101705", "GSE193777", "GSE107994", "GSE89403"]),
              ("Microarray", ["GSE19435", "GSE19439", "GSE29536"])]
    fig, axes = plt.subplots(1, 2, figsize=(4.3, 2.3))
    fig.subplots_adjust(left=.12, right=.98, bottom=.2, top=.87, wspace=.4)
    for ax, (title, cohorts) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], "--", color=".7", lw=.6)
        for cohort in cohorts:
            curve = curves.loc[curves.dataset.eq(cohort)]
            if curve.empty:
                raise ValueError(f"Missing ROC points for {cohort}")
            ax.step(curve.fpr, curve.tpr, where="post", lw=1,
                    label=f"{cohort} ({auc.at[cohort]:.2f})")
        ax.set(xlim=(0, 1), ylim=(0, 1.02), title=title, xlabel="False positive rate")
        ax.legend(loc="lower right", fontsize=5.8, frameon=False)
    axes[0].set_ylabel("True positive rate")
    put(page, fig, (263, 259, 504, 383))


def main():
    out = fitz.open()
    page = out.new_page(width=504.566, height=385.51)
    for stem, left, right in [("INTERNAL", 8, 168),
                              ("CD_RISK_FRACTION_COMPOSITION_LARGE_FONT", 172, 332),
                              ("EXTERNAL", 336, 496)]:
        with fitz.open(OUT / "cell_panels" / f"{stem}.pdf") as panel:
            page.show_pdf_page(fitz.Rect(left, 13, right, 97), panel, 0)
    for x, letter in zip([9, 88, 173, 251, 337, 416], "abcdef"):
        page.insert_text((x, 11), letter, fontname="hebo", fontsize=10)
    roc_panels(page)
    for x,y,letter in [(9,108,'g'),(64,108,'h'),(121,108,'i'),(229,108,'j'),(335,108,'k'),(9,255,'l'),(268,255,'m'),(391,255,'n')]:
        page.insert_text((x,y),letter,fontname='hebo',fontsize=10)
    i=pd.read_csv(OUT/'T1_infection_DE.tsv.gz',sep='\t')
    j=pd.read_csv(OUT/'T2_disease_DE.tsv.gz',sep='\t')
    bar=pd.read_csv(OUT/'k_expression.tsv',sep='\t')
    burdens(page);volcanos(page,i,j,bar);bars(page,bar);go(page)
    out.save(OUT/'Figure2.pdf',garbage=4,deflate=True)
    page.get_pixmap(dpi=450,alpha=False).save(OUT/'Figure2.png')
    out.close()

if __name__=='__main__':main()
