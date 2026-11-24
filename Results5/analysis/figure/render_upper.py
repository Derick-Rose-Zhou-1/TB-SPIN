from pathlib import Path
from textwrap import fill
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.transforms import Bbox
P=Path(__file__).resolve().parent.parent
old=P.parent
lab=pd.read_csv(P.parent/'scoring/ensemble_labels.tsv.gz',sep='\t',dtype={'risk':'boolean'})
lab=lab[lab.analysis_stage!='ATB'].set_index('cell_id',verify_integrity=True)
meta=pd.read_csv(old/'inputs/FIG1_QC_CELL_METADATA_UMAP.tsv.gz',sep='\t').set_index('cell_id',verify_integrity=True)
assert lab.index.isin(meta.index).all()
for col in ['analysis_stage','sample_id_std']:
    assert np.array_equal(meta.loc[lab.index,col].values,lab[col].values)
groups=['DTB','TPE','TBM']
palette={'T cell':'#4E79A7','NK cell':'#9C755F','Monocyte':'#59A14F','B cell':'#B07AA1','pDC':'#E5AE38','Neutrophil':'#76B7B2'}
color={'down':'#55548a','up':'#f48563'}
cmap=matplotlib.colors.ListedColormap(['#322958','#554786','#41649E','#6597BA','#B9D9E4','#F6F4EF','#F5DB90','#F0B05B','#E68054','#C74C53','#87284C'])
norm=matplotlib.colors.BoundaryNorm([-8,-6,-4,-3,-2,-1,1,2,3,4,6,8],cmap.N,clip=True)
plt.rcParams.update({'font.family':'Arial','font.size':5.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.45,'xtick.major.width':.4,'ytick.major.width':.4,'xtick.major.size':2,'ytick.major.size':2,'xtick.major.pad':1.4,'ytick.major.pad':1.4})
summary=[];allgenes=[];allshown=[]
for method in ['paired']:
    fig=plt.figure(figsize=(180/25.4,151/25.4),dpi=150)
    def box(x,y,w,h): return fig.add_axes([x,y,w,h])
    for i,t in enumerate(groups):
        left=i/3
        fig.text(left+.012,.982,'abc'[i],weight='bold',size=8)
        fig.text(left+.17,.982,t,weight='bold',size=7.5,ha='center')
        q=meta[meta.analysis_stage.isin(['ATB',t])]
        l=lab[lab.analysis_stage.eq(t)&lab.risk.notna()]
        pos=l[l.risk.fillna(False)]
        ax=box(left+.022,.705,.296,.264)
        # ATB and unselected target cells form the grey context.
        m=q.index.isin(pos.index)
        ax.scatter(q.loc[~m,'UMAP1'],q.loc[~m,'UMAP2'],s=.09,c='#D6D7D9',alpha=.52,lw=0,rasterized=True)
        ax.scatter(q.loc[m,'UMAP1'],q.loc[m,'UMAP2'],s=.15,c='#D55E5E',alpha=.80,lw=0,rasterized=True)
        for key,setter in [('UMAP1',ax.set_xlim),('UMAP2',ax.set_ylim)]:
            a,b=meta[key].agg(['min','max']);setter(a-.035*(b-a),b+.035*(b-a))
        ax.set_aspect('equal');ax.axis('off')
        for pt in [(.21,.07),(.04,.24)]:ax.annotate('',xy=pt,xytext=(.04,.07),xycoords='axes fraction',arrowprops=dict(arrowstyle='-|>',lw=.5,mutation_scale=4,color='#222',shrinkA=0,shrinkB=0))
        ax.text(.11,.012,'UMAP1',transform=ax.transAxes,fontsize=5,ha='center');ax.text(.01,.16,'UMAP2',transform=ax.transAxes,fontsize=5,rotation=90,ha='right',va='center')
        fig.text(left+.17,.690,f'Risk cells {len(pos)/len(l):.2%}',ha='center',size=6)
        bar=box(left+.043,.641,.253,.041);bar.axis('off');bar.set(xlim=(0,100),ylim=(-.5,.5))
        start=0;minor=[]
        for k,c in palette.items():
            n=pos.celltype_raw.eq(k).sum();v=100*n/len(pos);bar.barh(0,v,left=start,height=1,color=c,lw=0)
            if v>=10:bar.text(start+v/2,0,f'{v:.1f}%',ha='center',va='center',color='white',weight='bold',fontsize=5.2)
            elif n:minor.append(f'{k} {v:.1f}%' if v>=.1 else f'{k} <0.1%')
            if method=='paired':summary.append([t,len(l),len(pos),k,n,v])
            start+=v
        fig.text(left+.17,.625,'   '.join(minor),ha='center',size=5)
        ax=box(left+.061,.356,.258,.221)
        d=pd.read_csv(P/f'{t}_{method}.tsv.gz',sep='\t').dropna(subset=['log2FoldChange','FDR']).set_index('gene')
        x=d.log2FoldChange;y=-np.log10(d.FDR.clip(lower=1e-300));sig=d.FDR.lt(.05)&x.abs().ge(1.5)
        ax.scatter(x[~sig],y[~sig],s=.65,c='#9b9ba1',alpha=.55,lw=0,rasterized=True)
        ax.scatter(x[sig],y[sig],s=2.2,c=x[sig],cmap=cmap,norm=norm,lw=0,rasterized=True)
        for a in [-1.5,1.5]:ax.axvline(a,c='#aaa',lw=.4,ls='--',zorder=0)
        ax.axhline(-np.log10(.05),c='#aaa',lw=.4,ls='--',zorder=0)
        ax.set_xlabel('log$_{2}$ fold change',labelpad=1)
        if not i:ax.set_ylabel('−log$_{10}$(FDR)',labelpad=1)
        ax.margins(x=.08);ax.set_ylim(-.016*max(2,y.max()),1.14*max(2,y.max()))
        fig.text(left+.012,.570,'def'[i],weight='bold',size=8)
        genes=[]
        for sign in [1,-1]:
            subset=d.loc[sig&(sign*x>0)].copy()
            # Use the unadjusted P value and the signed test statistic to order ties.
            subset['absstat']=subset['stat' if method=='paired' else 'score'].abs()
            genes+=subset.sort_values(['FDR','pvalue','absstat'],ascending=[True,True,False]).head(4).index.tolist()
        fig.canvas.draw();placed=[]
        for gene in genes:
            px,py=ax.transData.transform((x[gene],y[gene]))
            opts=[]
            for dy in [5,10,-6,15,-12,20,-18,26,-24,32,-30]:
                for dx in [4,-4,10,-10,18,-18,26,-26,34,-34]:
                    a,b=np.array([px,py])+np.array([dx,dy])*fig.dpi/72
                    w,h,_=fig.canvas.get_renderer().get_text_width_height_descent(gene,matplotlib.font_manager.FontProperties(family='Arial',size=5,style='italic'),False)
                    rect=Bbox.from_bounds(a-(w if dx<0 else 0),b-h/2,w,h+2).expanded(1.10,1.22)
                    penalty=10000*(not(ax.bbox.contains(*rect.min) and ax.bbox.contains(*rect.max)))+1000*sum(rect.overlaps(r) for r in placed)
                    opts.append((penalty+np.hypot(dx,dy),dx,dy,rect))
            _,dx,dy,r=min(opts,key=lambda z:z[0]);placed.append(r)
            ax.annotate(gene,(x[gene],y[gene]),xytext=(dx,dy),textcoords='offset points',fontsize=5,fontstyle='italic',ha='right' if dx<0 else 'left',va='center',bbox=dict(facecolor='white',edgecolor='none',alpha=.70,pad=.05),arrowprops=dict(arrowstyle='-',lw=.3,color='#777'))
            allgenes.append([method,t,gene,x[gene],d.loc[gene,'FDR']])
    fig.legend(handles=[Patch(facecolor=c,label=k) for k,c in palette.items()],loc='center',bbox_to_anchor=(.5,.605),frameon=False,ncol=6,handlelength=1,fontsize=5.5)
    cax=box(.463,.297,.125,.008)
    cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=[-8,0,8]);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=5,pad=.5,length=1)
    fig.text(.455,.295,'Significant genes: log$_{2}$ fold change',ha='right',size=5)
    fig.savefig(Path(__file__).with_name('upper.pdf'),dpi=420,facecolor='white')
