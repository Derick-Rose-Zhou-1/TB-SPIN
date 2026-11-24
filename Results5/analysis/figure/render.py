from pathlib import Path
import textwrap
import fitz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
P=Path(__file__).resolve().parent; B=P.parent
groups=['DTB','TPE','TBM']
colors={'up':'#E68054','down':'#41649E'}
orig=pd.read_csv(B/'bidirectional_functions/panel_source.tsv',sep='\t')
ora=pd.read_csv(B/'ORA_all.tsv.gz',sep='\t').query("method=='paired'")
extra={'DTB':('KEGG_CELL_ADHESION_MOLECULES_CAMS','Cell adhesion molecules'),'TPE':('KEGG_CYTOKINE_CYTOKINE_RECEPTOR_INTERACTION','Cytokine–receptor interaction'),'TBM':('KEGG_TOLL_LIKE_RECEPTOR_SIGNALING_PATHWAY','Toll-like receptor signalling')}
rows=[]
for g in groups:
    sel=orig[orig.group.eq(g)][['term','display_name','direction']].copy()
    entries=[('GOBP_LEUKOCYTE_PROLIFERATION','Leukocyte proliferation','up')]+list(sel[sel.direction.eq('up')].itertuples(index=False,name=None))+[(*extra[g],'up')]+list(sel[sel.direction.eq('down')].itertuples(index=False,name=None))
    for i,(t,n,dr) in enumerate(entries):
        r=ora[ora.group.eq(g)&ora.direction.eq(dr)&ora.term.eq(t)].iloc[0].to_dict()
        assert r['full_BH']<.05
        rows.append(dict(r,display_name=n,display_order=i+1))
source=pd.DataFrame(rows);assert source.groupby(['group','library']).size().eq(5).all()
source.to_csv(P/'GO_KEGG_source.tsv',sep='\t',index=False)
heat=pd.read_csv(B/'discrete_NES/Figure5j_source.tsv',sep='\t')
heat.to_csv(P/'GSEA_source.tsv',sep='\t',index=False)
names=['B cell receptor\nsignalling','Cell killing','NK cell-mediated\ncytotoxicity','MHC I peptide\nantigen presentation','Cytosolic DNA\nsensing','Tryptophan\nmetabolism','Vacuole\norganization','Classical complement\nactivation','Lysosome']
plt.rcParams.update({'font.family':'Arial','font.size':5.5,'pdf.fonttype':42,'axes.linewidth':.45,'xtick.major.width':.45,'xtick.major.size':2,'xtick.major.pad':1.4})
fig=plt.figure(figsize=(180/25.4,83/25.4))
fig.legend(handles=[Patch(color=colors[k],label=t) for k,t in [('down','Downregulated genes'),('up','Upregulated genes')]],loc='upper center',bbox_to_anchor=(.5,1.004),ncol=2,frameon=False,fontsize=5.4,handlelength=1,borderaxespad=0)
for i,g in enumerate(groups):
    left=i/3;ax=fig.add_axes([left+.178,.475,.139,.455]);sub=source[source.group.eq(g)]
    ticks=[]
    for j,r in enumerate(sub.itertuples()):
        lab=r.display_name+' ['+('GO BP' if r.library=='GO_BP' else 'KEGG')+']'
        lines=textwrap.wrap(lab.replace('GO BP','GO~BP'),width=34,break_long_words=False,break_on_hyphens=False)
        ticks.append('\n'.join(lines).replace('GO~BP','GO BP'))
        ax.barh(j,-np.log10(r.full_BH),height=.57,color=colors[r.direction])
    ax.set_yticks(range(10),ticks,fontsize=5.05);ax.tick_params(axis='y',length=0,pad=1.3);ax.set_ylim(9.6,-.65);limit=15 if g in ('DTB','TPE') else 25;ax.set_xlim(0,limit);ax.set_xticks([0,5,10,15] if limit==15 else [0,10,20])
    for j,r in enumerate(sub.itertuples()):
        value=-np.log10(r.full_BH)
        if value>limit:
            ax.plot(limit-.18,j,marker='>',markersize=3,color='white',markeredgewidth=0)
            if g=='DTB' and r.display_name=='T cell activation': continue
            ax.text(limit-.65,j,f'{value:.2f}',ha='right',va='center',fontsize=5.3,weight='bold',color='#222',bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':.25})
    ax.spines[['top','right']].set_visible(False);ax.set_xlabel('−log$_{10}$(FDR)',labelpad=1)
    fig.text(left+.01,.958,'ghi'[i],weight='bold',size=7.5)
ax=fig.add_axes([.053,.155,.937,.185])
v=heat.pivot(index='branch',columns='column',values='NES').loc[groups].values
q=heat.pivot(index='branch',columns='column',values='FDR').loc[groups].values
bounds=[-3,-2,-1.75,-1.5,-1,-.5,.5,1,1.5,1.75,2,3]
c=ListedColormap(['#322958','#554786','#41649E','#6597BA','#B9D9E4','#F6F4EF','#F5DB90','#F0B05B','#E68054','#C74C53','#87284C'])
im=ax.imshow(v,aspect='auto',cmap=c,norm=BoundaryNorm(bounds,c.N),interpolation='nearest')
ax.set_yticks(range(3),groups,weight='bold');ax.set_xticks(range(9),names,fontsize=5.2);ax.tick_params(which='both',length=0,pad=2)
for j in range(8):ax.axvline(j+.5,color='white',lw=1.8 if j in (2,5) else .45)
for i in [.5,1.5]:ax.axhline(i,color='white',lw=.6)
for s in ax.spines.values():s.set_visible(False)
for i,j in np.ndindex(v.shape):
    f=q[i,j];stars='***' if f<.001 else '**' if f<.01 else '*' if f<.05 else ''
    rgb=np.array(c(im.norm(v[i,j]))[:3])
    luminance=np.dot(np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4),[.2126,.7152,.0722])
    ax.text(j,i,f'{v[i,j]:+.2f}{stars}',ha='center',va='center',fontsize=5.8,color='white' if luminance<.179 else '#282526')
for i,g in enumerate(groups):ax.text(3*i+1,1.075,g,transform=ax.get_xaxis_transform(),size=5.3,ha='center',weight='bold')
fig.text(.009,.351,'j',weight='bold',size=7.5)
cb=fig.colorbar(im,cax=fig.add_axes([.115,.052,.39,.025]),orientation='horizontal',ticks=[-3,-2,-1,0,1,2,3],spacing='proportional')
cb.outline.set_visible(False);cb.ax.tick_params(length=1,labelsize=7.2)
fig.text(.068,.054,'NES',size=7);fig.text(.56,.054,'* q < 0.05     ** q < 0.01     *** q < 0.001',size=5.6)
fig.savefig(P/'Figure5_g-j.pdf');plt.close(fig)
with fitz.open(P/'upper.pdf') as art, fitz.open(P/'Figure5_g-j.pdf') as lower:
    r=art[0].rect;cut=r.height*.723
    doc=fitz.open();page=doc.new_page(width=r.width,height=cut+lower[0].rect.height)
    clip=fitz.Rect(0,0,r.width,cut);page.show_pdf_page(clip,art,0,clip=clip)
    for x in [.01,.343,.676]:page.draw_rect(fitz.Rect(x*r.width,r.height*.711,(x+.018)*r.width,cut),color=None,fill=(1,1,1))
    page.show_pdf_page(fitz.Rect(0,cut,r.width,page.rect.height),lower,0);doc.save(P/'Figure5.pdf',garbage=4,deflate=True)
    page.get_pixmap(dpi=350).save(P/'Figure5.png')
    page.get_pixmap(dpi=135).save(P/'preview.png')


