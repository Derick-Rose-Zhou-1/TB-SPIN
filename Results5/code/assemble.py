from pathlib import Path
from io import BytesIO
from textwrap import fill
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import fitz
import os
root=Path(os.environ["FIG5_WORK"])
base=Path(__file__).resolve().parents[1]
old=fitz.open(base/'inputs/Fig5abc_frozen.pdf')
heat=fitz.open(root/'heat.pdf')
ora=pd.read_csv(base/'results/Fig5ghi_source.tsv',sep='\t')
colors={'up':'#f48563','down':'#55548a'}
names=['DTB','TPE','TBM']
mpl.rcParams.update({'pdf.fonttype':42,'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
def vector(fig):
    buf=BytesIO();fig.savefig(buf,format='pdf',dpi=300);plt.close(fig)
    return fitz.open(stream=buf.getvalue(),filetype='pdf')
scale=mpl.colors.Normalize(-8,8,clip=True)
cmap=mpl.colors.LinearSegmentedColormap.from_list("fold_change",[(0,"#55548a"),(.40625,"#a4a2c6"),(.5,"#faf8f5"),(.59375,"#f8baa6"),(1,"#f48563")],N=257)
fig,axs=plt.subplots(1,3,figsize=(15,3.5))
fig.subplots_adjust(left=.062,right=.984,bottom=.28,top=.98,wspace=.21)
labels=['PRSS23 MS4A1 PKIG CD79A VPREB3 ENHO MPP3 CLEC9A NEGR1 HIC1','MS4A1 FCGR3A TTC38 SPON2 CD79A CLIC3 TCL1A NIBAN3 GZMH GNLY CLDN1 CDH20 MDH1B','NKG7 GZMH CCL5 PTGDS MS4A1 TEX101 SLC52A3 CLDN5']
for i,(g,ax) in enumerate(zip(names,axs)):
    f=base/'results'/f'{g}_paired.tsv.gz'
    d=pd.read_csv(f,sep='\t').dropna(subset=['FDR','log2FoldChange']).set_index('gene')
    x=d.log2FoldChange;y=-np.log10(d.FDR.clip(lower=1e-300));sig=(d.FDR<.05)&(x.abs()>=1.5)
    ax.scatter(x[~sig],y[~sig],s=2,color='#a4a4a8',alpha=.55,lw=0,rasterized=True)
    for k,mask in [('up',sig&(x>0)),('down',sig&(x<0))]:
        ax.scatter(x[mask],y[mask],s=9,c=x[mask],cmap=cmap,norm=scale,lw=0,rasterized=True)
    for v in [-1.5,1.5]:ax.axvline(v,color='#aaa',lw=.6,ls='--',zorder=0)
    ax.axhline(-np.log10(.05),color='#aaa',ls='--',lw=.6,zorder=0)
    labs=labels[i].split();assert sig.loc[labs].all()
    ax.margins(x=.075);ax.set_ylim(-.15,1.12*y.max());ax.set_xlabel('log₂ fold change')
    if i==0:ax.set_ylabel('−log₁₀(FDR)')
    fig.canvas.draw();placed=[]
    for gene in sorted(labs,key=lambda t:-y[t]):
        px,py=ax.transData.transform((x[gene],y[gene]))
        opts=[]
        for dy in [8,14,-9,19,26,33,40,47,54,61,68]:
            for dx in [6,-6,15,-15,25,-25,38,-38]:
                pos=np.array([px,py])+np.array([dx,dy])*fig.dpi/72
                w,h,_=fig.canvas.get_renderer().get_text_width_height_descent(gene,mpl.font_manager.FontProperties(size=9,style='italic'),False)
                box=mpl.transforms.Bbox.from_bounds(pos[0]-(w if dx<0 else 0),pos[1]-h/2,w,h+2).expanded(1.1,1.24)
                bad=not(ax.bbox.contains(*box.min) and ax.bbox.contains(*box.max))
                opts.append((bad*10000+sum(box.overlaps(z) for z in placed)*1000+np.hypot(dx,dy),dx,dy,box))
        _,dx,dy,box=min(opts,key=lambda a:a[0]);placed.append(box)
        ax.annotate(gene,(x[gene],y[gene]),xytext=(dx,dy),textcoords='offset points',fontsize=9,fontstyle='italic',ha='right' if dx<0 else 'left',va='center',bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.25),arrowprops=dict(arrowstyle='-',color='#777',lw=.45))
    fig.text(.012+i/3,.925,'def'[i],weight='bold',fontsize=16)
cax=fig.add_axes([.46,.045,.13,.028])
cb=fig.colorbar(mpl.cm.ScalarMappable(norm=scale,cmap=cmap),cax=cax,orientation='horizontal',ticks=[-8,0,8])
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=10,length=1.5,pad=1)
fig.text(.445,.038,'Significant genes: log₂ fold change',ha='right',fontsize=10)
volcano=vector(fig)
def oraplot(kind):
    fig=plt.figure(figsize=(15,3.65))
    for i,g in enumerate(names):
        left=i/3
        ax=fig.add_axes([left+.178,.17,.14,.72])
        sub=ora[ora.group.eq(g)].sort_values(['direction','library','BH_FDR'],ascending=[False,True,True])
        assert len(sub)==8
        ticks=[]
        for j,r in enumerate(sub.itertuples()):
            label=r.term.split('_',1)[1].replace('_',' ').lower().capitalize().replace('Natural killer cell mediated','NK cell-mediated').replace('Granzyme mediated programmed cell death signaling pathway','Granzyme-mediated cell death signalling').replace('Leukocyte transendothelial migration','Transendothelial leukocyte migration')
            prefix='GO BP' if r.library=='GO_BP' else 'KEGG'
            ticks.append(fill(label,29)+'  ['+prefix+']')
            val=-np.log10(r.BH_FDR);c=colors[r.direction]
            if kind=='bars':ax.barh(j,val,color=c,height=.57)
            else:
                ax.hlines(j,0,val,color=c,lw=1.7);ax.scatter(val,j,s=37,color=c,zorder=3)
        ax.set_yticks(range(8),ticks,fontsize=8.9);ax.set_ylim(7.5,-.5);ax.set_xlim(0,12);ax.set_xticks([0,4,8,12]);ax.tick_params(axis='y',length=0,pad=4)
        ax.set_xlabel('−log₁₀(FDR)',fontsize=10)
        fig.text(left+.012,.92,'ghi'[i],fontsize=16,weight='bold')
    for idx,(k,txt) in enumerate([('down','Downregulated in risk cells'),('up','Upregulated in risk cells')]):
        fig.text(.37+.15*idx,.94,'■',color=colors[k],fontsize=12);fig.text(.384+.15*idx,.94,txt,fontsize=10)
    return vector(fig)
for kind in ['bars']:
    orafig=oraplot(kind)
    doc=fitz.open();page=doc.new_page(width=1080,height=354+volcano[0].rect.height+orafig[0].rect.height+heat[0].rect.height*1080/heat[0].rect.width)
    def strip(y1,y2,target):
        box=fitz.Rect(0,y1,1080,y2);page.show_pdf_page(fitz.Rect(0,target,1080,target+box.height),old,0,clip=box)
    strip(7,39,0);strip(42,268,32);strip(278,331,259);strip(339,357,313);strip(368,387,333)
    y0=354
    for part in [volcano,orafig,heat]:
        h=part[0].rect.height*1080/part[0].rect.width
        page.show_pdf_page(fitz.Rect(0,y0,1080,y0+h),part,0);y0+=h
    dest=root/f'Fig5_{kind}.pdf';doc.save(dest,garbage=4,deflate=True)
    with fitz.open(dest) as ck:
        words={w[4] for w in ck[0].get_text('words')};assert set('abcdefghij')<=words
        ck[0].get_pixmap(dpi=260).save(dest.with_suffix('.png'))



