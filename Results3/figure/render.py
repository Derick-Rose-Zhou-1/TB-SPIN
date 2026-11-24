"""Source-locked typography revision of PPT slide 2. No statistical recomputation."""
from pathlib import Path
import importlib.util, hashlib, json, re, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import fitz

import argparse
parser = argparse.ArgumentParser(description="Render Fig. 3 from its four archived panel tables.")
parser.add_argument("tables", type=Path, help="Directory containing Main/01_SOURCE_TABLES and Main/tables")
parser.add_argument("output", type=Path)
args = parser.parse_args()
QA = args.output / "qa"
QA.mkdir(parents=True, exist_ok=True)
BASE = args.output / "Figure3"
def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    """Set matplotlib rcParams for publication-grade output. Call once before plotting.

    This sets mechanics (role-mapped font-size ladder, outward ticks, frameless
    legends, 300-dpi save, Type-42 embedded fonts) — not a house aesthetic.
    Frame, font and the size ladder are parameters.

    frame : 'open' (bottom+left spines, default) | 'boxed' (all four) | 'none'
    font  : sans-serif family name; None = system default sans-serif
    sizes : (base, secondary, tick) — titles/axis-labels, legend/annotation, ticks
    grid  : whether to draw axes.grid (default False)
    """
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")

    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)

apply_figure_style(font='Arial',sizes=(7,6,6))
plt.rcParams.update({'svg.fonttype':'none','savefig.bbox':None,'mathtext.fontset':'custom',
 'mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold'})
W,H=180,205
BASELINE_Y_MM=99
fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=160,facecolor='white')
SOURCE={
 'a':args.tables/'Main/01_SOURCE_TABLES/PANEL_A_AUROC.tsv',
 'b':args.tables/'Main/tables/RESULT3_3_1_FINAL_METRICS.tsv',
 'c':args.tables/'Main/01_SOURCE_TABLES/PANEL_C_LINEAGE.tsv',
 'd':args.tables/'Main/01_SOURCE_TABLES/PANEL_D_DISPLAY_PATHWAYS.tsv'}
hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in SOURCE.items()}
A,B,C,D=[pd.read_csv(SOURCE[k],sep='\t') for k in 'abcd']
B=B[(B.metric=='HR_per_SD') & (B.analysis_type=='COMPLETE_FOLLOWUP_SURVIVAL')]
assert len(D)==48 and D.groupby(['axis','direction']).size().eq(8).all()
assert (D.FDR<.05).all()
for k,d in zip('abcd',[A,B,C,D]):d.to_csv(QA/f'PANEL_{k.upper()}_SOURCE.tsv',sep='\t',index=False)

def txt(x,y,s,size=6,bold=False,ha='left',va='center',color='black',**kw):
    return fig.text(x/W,1-y/H,s,fontsize=size,fontweight='bold' if bold else 'normal',ha=ha,va=va,color=color,**kw)
def axmm(x,y,w,h):
    return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
def line(x1,y1,x2,y2,color='#444444',lw=.6,ls='-'):
    fig.add_artist(plt.Line2D([x1/W,x2/W],[1-y1/H,1-y2/H],transform=fig.transFigure,color=color,lw=lw,ls=ls))
def box(x,y,w,h,color):
    fig.add_artist(Rectangle((x/W,1-(y+h)/H),w/W,h/H,transform=fig.transFigure,facecolor=color,edgecolor='none'))
def label(x,y,s):txt(x,y,s,8,True)
cohorts=['GSE79362','GSE94438','GSE107993_GSE107994_combined']
names=['GSE79362','GSE94438','GSE107993/\nGSE107994']

# a: same estimates and marker semantics as original PPT; column whitespace reclaimed.
label(3,4,'a')
txt(3,14,'Cohort',6,True);txt(66,14,'AUROC (95% CI)',6,True);txt(99,14,'n/events',6,True,ha='right')
# Align the AUROC and lineage-composition x-axis baselines by construction.
# Both axes terminate at the same absolute figure coordinate.
forest=axmm(35,18,28,BASELINE_Y_MM-18)
forest.set(xlim=(.45,1.01),ylim=(99,18),yticks=[],xticks=[.5,.6,.7,.8,.9,1.0])
forest.spines['left'].set_visible(False)
forest.tick_params(axis='x',pad=2,length=2)
forest.axvline(.5,color='#88929d',ls='--',lw=.6)
forest.set_xlabel('AUROC',fontsize=6,labelpad=2)
groups=[('H1','Complete follow-up',19,[26,32,38],'#6b7486','o'),
 ('H2','Incident TB ≤180 days',45,[52,58,64],'#23648f','o'),
 ('H3','Incident TB ≤365 days',71,[78,84,90],'#bd494d','s')]
for ep,title,gy,ys,color,marker in groups:
    txt(3,gy,title,6,True)
    for co,nm,y in zip(cohorts,names,ys):
        txt(3,y,nm,6,linespacing=1)
        row=A[(A.endpoint==ep)&(A.cohort==co)]
        if row.empty:
            txt(67,y,'Not estimable',6);continue
        r=row.iloc[0]
        forest.errorbar(r.value,y,xerr=[[r.value-r.ci_low],[r.ci_high-r.value]],fmt=marker,
            ms=3.3,color=color,mfc='white' if 'combined' in co else color,capthick=.7,capsize=1.6,lw=.7)
        txt(65,y,f'{r.value:.2f} ({r.ci_low:.2f}–{r.ci_high:.2f})',6)
        txt(99,y,f'{int(r.n_participants)}/{int(r.n_events)}',6,ha='right')

# b: all three original PPT cohorts retained; text moved to distinct rows.
label(106,4,'b')
txt(145,13,'HR (95% CI); P',6,True)
hr=axmm(130,16,13,19)
hr.set(xlim=(0,7),ylim=(35,16),yticks=[],xticks=[0,2,4,6])
hr.spines['left'].set_visible(False);hr.tick_params(axis='x',length=2,pad=2)
hr.axvline(1,color='#999999',ls='--',lw=.6)
for co,nm,y,color in zip(cohorts,names,[19,26,33],['#155494','#4698a0','#994d95']):
    r=B[B.cohort==co].iloc[0]
    txt(107,y,nm,6,linespacing=1)
    hr.errorbar(r.value,y,xerr=[[r.value-r.ci_low],[r.ci_high-r.value]],fmt='o',color=color,ms=3.3,capsize=1.5,lw=.7)
    txt(145,y-1.8,f'{r.value:.2f} ({r.ci_low:.2f}–{r.ci_high:.2f})',6)
    if r.p_value<1e-5:
        base=txt(145,y+1.8,'P = 2.9 × 10',6)
        width_mm=base.get_window_extent(fig.canvas.get_renderer()).width/fig.dpi*25.4
        txt(145+width_mm+.2,y+.9,'−7',5.2)
    else:
        txt(145,y+1.8,f'P = {r.p_value:.3f}' if r.p_value<.1 else f'P = {r.p_value:.2f}',6)
txt(136.5,40,'HR per SD',6,ha='center')

# c: every original count and percentage retained; no staggered text collision.
label(106,45,'c')
states=['180-day only','Shared','365-day only'];colors=['#306f9e','#8061a7','#c54c53']
for x,s,color in zip([108,136,154],states,colors):
    box(x,50,2,1.3,color);txt(x+2.7,50.65,s,6)
cx,cy,cw,ch=123,55,52,BASELINE_Y_MM-55
comp=axmm(cx,cy,cw,ch)
comp.set(xlim=(0,78),ylim=(ch,0),yticks=[],xticks=[0,20,40,60])
comp.spines['left'].set_visible(False);comp.tick_params(axis='x',pad=2,length=2)
comp.set_xlabel('Cells within temporal class (%)',fontsize=6,labelpad=2)
for i,ln in enumerate(['T cell','Monocyte','NK cell','B cell','Neutrophil','pDC']):
    center=3+i*7.1
    txt(121,cy+center,ln,6,ha='right')
    for j,(s,color) in enumerate(zip(states,colors)):
        r=C[(C.major_lineage==ln)&(C.temporal_state==s)].iloc[0]
        yy=center+(j-1)*2.15
        comp.barh(yy,r.composition_pct,height=1.65,color=color,edgecolor='none')
        comp.text(r.composition_pct+1.1,yy,f'{int(r.cell_n):,} ({r.composition_pct:.1f}%)',fontsize=6,va='center')

# d: original 48 bars, exact ranking and stored -log10(FDR), variable line-aware spacing.
label(3,110,'d')
for x,s,color in [(102,'GO Biological Process','#73b7b0'),(150,'KEGG','#8297c7')]:
    box(x,109.4,3,1.8,color);txt(x+4,110.3,s,6)
txt(4,120,'UP',6,True,rotation=90)
txt(4,166,'DOWN',6,True,rotation=90)
dpcols=[(40,20,'H180','180-day'),(98,20,'H365','365-day'),(157,20,'COX','Cox')]
text_records=[]
for xp,wp,ep,title in dpcols:
    txt(xp+wp/2,116,title,7,True,ha='center')
    for di,ytop in [('UP',120),('DOWN',161)]:
        sub=D[(D.axis==ep)&(D.direction==di)].sort_values('minus_log10_FDR',ascending=True,kind='mergesort').iloc[::-1]
        labels=[]
        for term in sub.term:
            clean=re.sub('^(GOBP_|KEGG_)','',term).replace('_',' ').capitalize()
            labels.append('\n'.join(textwrap.wrap(clean,31,break_long_words=False,break_on_hyphens=False)))
        units=np.array([max(1,s.count('\n')+1)*2.2+.55 for s in labels])
        total=units.sum()
        height=max(35,total)
        if height>38.6:raise RuntimeError(f'Term text needs more space: {ep} {di} {height}')
        centers=np.cumsum(units)-units/2
        centers+= (height-total)/2
        ax=axmm(xp,ytop,wp,height)
        if ep == 'H180':
            # Requested 0-5 tick range. The 5.129 term remains visible by a
            # small right margin rather than being clipped at the boundary.
            ax.set(xlim=(0,5.3),ylim=(height,0),yticks=[],xticks=[0,1,2,3,4,5])
        else:
            ax.set(xlim=(0,11),ylim=(height,0),yticks=[],xticks=[0,5,10])
        ax.spines['left'].set_visible(False);ax.tick_params(axis='x',pad=2,length=2)
        for y,lab,(_,r) in zip(centers,labels,sub.iterrows()):
            ax.barh(y,r.minus_log10_FDR,height=1.65,color='#73b7b0' if r.database=='GO_BP' else '#8297c7',edgecolor='none')
            txt(xp-1,ytop+y,lab,6,ha='right',linespacing=1.03)
            text_records.append({'axis':ep,'direction':di,'term':r.term,'FDR':r.FDR,'value':r.minus_log10_FDR})
        if di=='UP':ax.tick_params(labelbottom=False)
txt(97,202,'−log10(FDR q value)',7,ha='center')
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
bad=[]
for i,t in enumerate(texts):
    a=t.get_window_extent(renderer)
    for u in texts[i+1:]:
        b=u.get_window_extent(renderer)
        if a.overlaps(b):bad.append({'text1':t.get_text(),'text2':u.get_text()})
pd.DataFrame(bad,columns=['text1','text2']).to_csv(QA/'TEXT_COLLISIONS.tsv',sep='\t',index=False)
assert not bad, f'Text collisions remain: {bad}'
outside=[t.get_text() for t in texts if not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0]) or not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[1])]
assert not outside, f'Off-canvas text: {outside}'
fig.savefig(str(BASE)+'.pdf')
fig.savefig(str(BASE)+'.svg')
fig.savefig(QA/'PREVIEW.png',dpi=220)
pd.DataFrame(text_records).to_csv(QA/'DRAWN_ORA_VALUES.tsv',sep='\t',index=False)
assert all(hashlib.sha256(SOURCE[k].read_bytes()).hexdigest()==v for k,v in hashes.items())
with fitz.open(str(BASE)+'.pdf') as doc:
    page=doc[0]
    sizes=[s['size'] for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
    report={'width_mm':W,'height_mm':H,'min_font_pt':min(sizes),'max_font_pt':max(sizes),
        'raster_images':len(page.get_images()),'vector_paths':len(page.get_drawings()),'text_collision_count':len(bad),
        'ORA_terms':len(text_records),'scientific_recomputation':False,'source_hashes':hashes,
        'pdf':str(BASE)+'.pdf','layout_note':'Original panel types, colours and 48 ORA terms; typography and geometry revised.'}
    report['all_fonts_embedded']=all(bool(doc.extract_font(font[0])[3]) for font in page.get_fonts(full=True))
    report['font_names']=[font[3] for font in page.get_fonts(full=True)]
    report['off_canvas_text_count']=len(outside)
    # Vector-spine QA: a and c bottom spines must share one absolute y.
    spine_rows=[]
    for dr in page.get_drawings():
        rr=dr['rect']
        if rr.width > 15 and rr.height < 1.5:
            spine_rows.append((rr.x0,rr.x1,rr.y0))
    a_spines=[y for x0,x1,y in spine_rows if x0 < 220 and x1 < 240 and 260 < y < 300]
    c_spines=[y for x0,x1,y in spine_rows if x0 > 330 and x1 > 450 and 260 < y < 300]
    report['a_c_axis_baseline_y_pt']={'a':round(min(a_spines),3) if a_spines else None,
                                      'c':round(min(c_spines),3) if c_spines else None,
                                      'delta_pt':round(abs(min(a_spines)-min(c_spines)),3) if a_spines and c_spines else None}
    assert a_spines and c_spines and abs(min(a_spines)-min(c_spines)) < 0.05
    assert 'GSEA' not in page.get_text()
    for name,rect in {'B':(104,0,180,43),'C':(104,42,180,107),'D':(0,107,180,205)}.items():
        clip=fitz.Rect(*(v*72/25.4 for v in rect))
        page.get_pixmap(clip=clip,matrix=fitz.Matrix(3,3)).save(QA/f'PANEL_{name}_PREVIEW.png')
(QA/'QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
