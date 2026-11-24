"""Source-locked four-column lollipop redraw; no statistical recomputation.

python make_fig4_source_verified_v13.py --input-dir SOURCE --module-dir SOURCE --output-dir OUTPUT
"""
from pathlib import Path
import argparse, hashlib, json, re, shutil, textwrap, importlib.metadata, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
import fitz

FILES=['Fig4_asymptomatic_GO_KEGG_source_data.tsv',
       'Fig4_symptomatic_GO_KEGG_source_data.tsv',
       'Fig4_asymptomatic_metabolic_source_data.tsv',
       'Fig4_symptomatic_metabolic_source_data.tsv']
COLORS=[ [('#85C7AD','#F0F8F4'),('#91B9DE','#F1F6FB')],
         [('#E7A0A4','#FCEFF0'),('#AFA8D8','#F5F3FB')],
         [('#C7B4E2','#F7F3FC')], [('#F0B27B','#FDF4EA')] ]
FONT=8.1
WIDTH=366
PALETTES=[['#F3FBFA','#DDF3F0','#BFE8E1','#93D7CC','#63C1B8','#389FA7','#1F7F94','#126A85'],
          ['#FFF6F2','#FCE6DD','#F9D0C4','#F5B3A6','#ED907F','#DB726D','#C95E67','#A94A62']]
MODULE_COLORS={(2,1):'#A68FC5',(2,2):'#C8AB61',(2,3):'#79B8A5',(2,4):'#7FA6CF',
               (3,1):'#D49B75',(3,2):'#CE8FA2',(3,3):'#96B077'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def label(s):
    s=str(s).strip()
    match=re.search(r'\s*\((GO:\d+)\)',s)
    identifier=match.group(1) if match else None
    s=re.sub(r'\s*\(GO:\d+\)','',s).lower()
    s=s[:1].upper()+s[1:]
    for pattern,replacement in [(r'\bmhc\b','MHC'),(r'\bclass ii\b','class II'),(r'\biga\b','IgA'),(r'\bth1\b','Th1'),(r'\bth2\b','Th2'),(r'\bth17\b','Th17'),(r'\brap1\b','Rap1'),(r'\bnos3\b','NOS3'),(r'\bpparalpha\b','PPARalpha'),(r'\btca\b','TCA'),(r'\bb12\b','B12'),(r'\bcbl\b','CBL')]:
        s=re.sub(pattern,replacement,s,flags=re.I)
    for pattern,replacement in [(r'\bb2\b','B2'),(r'\bnod\b','NOD'),(r'\bpi\b','PI')]:
        s=re.sub(pattern,replacement,s,flags=re.I)
    s=textwrap.fill(s,36,break_long_words=False,break_on_hyphens=False)
    return s

def prepare(t,i):
    t=t.copy()
    if i<2:
        t['name']=t.term;t['block']=t.ontology.map({'GO_BP':'GO biological process','KEGG':'KEGG pathways'})
        t['fdr_display']=t.FDR;t['size_metric']=t.overlap_count
        blocks=['GO biological process','KEGG pathways']
    else:
        t['name']=t.pathway;t['block']='Metabolic pathways'
        t['fdr_display']=t.min_FDR;t['size_metric']=t.n_methods_significant
        blocks=['Metabolic pathways']
    assert np.isfinite(t.fdr_display).all() and t.fdr_display.between(0,.05,inclusive='neither').all()
    t['x']=-np.log2(t.fdr_display)
    pieces=[]
    for b in blocks:
        g=t[t.block==b].sort_values(['x','size_metric'],ascending=False,kind='stable').copy()
        g['display_label']=g.name.map(label);g['rank_in_block']=range(1,len(g)+1);pieces.append(g)
    return pd.concat(pieces,ignore_index=True)

def area(v,metabolic=False):
    v=np.asarray(v,dtype=float)
    return np.where(v==2,25,45) if metabolic else 12+6*np.sqrt(v)

def axis_mm(fig,x,y,w,h):
    fw,fh=fig.get_size_inches()*25.4
    return fig.add_axes([x/fw,y/fh,w/fw,h/fh])
def text_mm(fig,x,y,s,**kw):
    fw,fh=fig.get_size_inches()*25.4
    return fig.text(x/fw,y/fh,s,**kw)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True);parser.add_argument('--module-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);a=parser.parse_args()
    out=a.output_dir.resolve();src=a.input_dir.resolve()
    for sub in ['01_source_locked','02_render_data','03_figures','04_qc']: (out/sub).mkdir(parents=True,exist_ok=True)
    script=out/Path(__file__).name
    if Path(__file__).resolve()!=script:shutil.copy2(__file__,script)
    hashes=[];tables=[]
    for i,name in enumerate(FILES):
        original=src/name;copy=out/'01_source_locked'/name
        if original.resolve()!=copy.resolve():shutil.copy2(original,copy)
        assert sha(original)==sha(copy)
        hashes.append({'file':name,'sha256':sha(copy)})
        tables.append(prepare(pd.read_csv(copy,sep='\t'),i))
    for i,phenotype in [(2,'ASYMPTOMATIC'),(3,'SYMPTOMATIC')]:
        module_file=phenotype+'_METABOLIC_MODULES.tsv'
        msrc=a.module_dir/module_file;locked=out/'01_source_locked'/module_file
        if msrc.resolve()!=locked.resolve():shutil.copy2(msrc,locked)
        assert sha(msrc)==sha(locked)
        m=pd.read_csv(locked,sep='\t')
        cols=['pathway','module_name','module_order','module_colour','phenotype_aligned_effect','raw_median_effect_size']
        tables[i]=tables[i].merge(m[cols],on='pathway',how='left',validate='one_to_one')
        assert tables[i].module_name.notna().all()
        assert np.allclose(tables[i].median_effect_size,tables[i].raw_median_effect_size)
        tables[i]=tables[i].sort_values(['module_order','x','pathway'],ascending=[True,False,True],kind='stable').reset_index(drop=True)
        tables[i].to_csv(out/'02_render_data'/module_file,sep='\t',index=False)
    # Shared row geometry uses real wrapped-label line counts, not large empty slots.
    row_heights=[]
    for j in range(16):
        lines=max(tables[i].iloc[j].display_label.count('\n')+1 for i in [0,1])
        row_heights.append(lines*FONT*1.10*25.4/72+0.9)
    centers=[];cursor=0
    for j,h in enumerate(row_heights):
        if j==8:cursor+=1.7
        centers.append(cursor+h/2);cursor+=h
    scale_y=max(1,108/cursor)
    centers=[v*scale_y for v in centers]
    row_heights=[v*scale_y for v in row_heights]
    ph=cursor*scale_y;bottom=22;top=bottom+ph;height=top+14
    plt.rcParams.update({'font.family':'Arial','font.size':FONT,'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':True})
    fig=plt.figure(figsize=(WIDTH/25.4,height/25.4),facecolor='white')
    axes=[];bracket_text=[];module_text=[];colorbars=[];strip_counts=[];color_key=[];mapped_points=[]
    for i,t in enumerate(tables):
        start=[2,90,178,266][i];x=[50,138,226,314][i]
        if i<2:t['plot_y']=centers
        else:
            positions=[];p=0;prev=None
            for mod in t.module_order:
                if prev is not None and mod!=prev:p+=.35
                positions.append(p);p+=1;prev=mod
            t['plot_y']=np.interp(positions,[min(positions),max(positions)],[centers[0],centers[-1]])
        t['render_rank']=range(1,len(t)+1)
        t.to_csv(out/'02_render_data'/f'PANEL_{"abcd"[i]}_RENDER_ORDER.tsv',sep='\t',index=False)
        ax=axis_mm(fig,x,bottom,22,ph);axes.append(ax)
        norm=Normalize(float(t.x.min()),float(t.x.max()))
        cmap=LinearSegmentedColormap.from_list('phenotype_'+str(i),PALETTES[i%2],N=256)
        for b,(block,g) in enumerate(t.groupby('block',sort=False)):
            if i>=2:continue
            col,bg=COLORS[i][b]
            lo,hi=(0,sum(row_heights[:8])) if i<2 and b==0 else ((sum(row_heights[:8])+1.7,ph) if i<2 else (0,ph))
            ax.axhspan(lo,hi,color=bg,zorder=0)
            ax.hlines(g.plot_y,0,g.x,color=col,lw=.8,zorder=2)
            ax.scatter(g.x,g.plot_y,s=area(g.size_metric,i>=2),marker='D' if i>=2 else 'o',c=g.x,cmap=cmap,norm=norm,edgecolor='#706966',linewidth=.5,zorder=3)
            transform=ax.get_yaxis_transform()
            if i<2:ax.plot([1.012,1.026,1.026,1.012],[lo,lo,hi,hi],transform=transform,color=col,lw=.8,clip_on=False)
            if i<2:bracket_text.append(ax.text(1.035,(lo+hi)/2,block,rotation=270,transform=transform,va='center',ha='left',fontsize=7.5,color=col,clip_on=False))
        if i>=2:
            groups=list(t.groupby('module_order',sort=False))
            for j,(_,g) in enumerate(groups):
                lo=0 if j==0 else (groups[j-1][1].plot_y.max()+g.plot_y.min())/2+.6
                hi=ph if j==len(groups)-1 else (g.plot_y.max()+groups[j+1][1].plot_y.min())/2-.6
                base=MODULE_COLORS[(i,int(g.module_order.iloc[0]))]
                rgb=np.asarray(matplotlib.colors.to_rgb(base))
                col=tuple(rgb*.75);bg=tuple(rgb*.10+.90)
                gcmap=LinearSegmentedColormap.from_list('module_'+str(i)+'_'+str(j),[rgb*.20+.80,rgb,rgb*.62],N=256)
                ax.axhspan(lo,hi,color=bg,zorder=0)
                ax.hlines(g.plot_y,0,g.x,color=base,lw=.8,zorder=2)
                points=ax.scatter(g.x,g.plot_y,s=area(g.size_metric,True),marker='D',c=g.x,cmap=gcmap,norm=norm,edgecolor='#706966',linewidth=.5,zorder=3)
                mapped_points.append((points,gcmap,norm))
                trans=ax.get_yaxis_transform()
                ax.plot([1.012,1.026,1.026,1.012],[lo,lo,hi,hi],transform=trans,color=col,lw=.8,clip_on=False)
                module_text.append(ax.text(1.05,(lo+hi)/2,textwrap.fill(g.module_name.iloc[0],14,break_long_words=False),rotation=270,transform=trans,ha='left',va='center',fontsize=7.5,color=col,clip_on=False))
                bar_height=min(23,hi-lo-3)
                cax=axis_mm(fig,x+35,top-(lo+hi)/2-bar_height/2,2.4,bar_height)
                cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=gcmap),cax=cax)
                cb.solids.set_rasterized(False);cb.outline.set_visible(False)
                cb.set_ticks(np.linspace(norm.vmin,norm.vmax,3));cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
                cb.ax.tick_params(labelsize=7,length=2,pad=1);colorbars.append(cb)
                color_key.append({'panel':'abcd'[i],'module':g.module_name.iloc[0],'base_colour':base,'background':matplotlib.colors.to_hex(bg),'colour_variable':'-log2(min FDR)','vmin':norm.vmin,'vmax':norm.vmax})
            strip_counts.append(len(groups))
        cbar_offset=30 if i<2 else 35
        if i<2:
            cax=axis_mm(fig,x+cbar_offset,top-36,2.4,31)
            cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax)
            cb.solids.set_rasterized(False);cb.outline.set_visible(False)
            cb.locator=MaxNLocator(nbins=3);cb.update_ticks();cb.ax.tick_params(labelsize=7,length=2,pad=1)
            colorbars.append(cb)
        text_mm(fig,x+cbar_offset+1,top-2 if i<2 else top+2,'−log2(FDR)' if i<2 else '−log2(min FDR)',fontsize=7.2,ha='center')
        xmax=[55,35,20,20][i];ticks=[[0,20,40],[0,10,20,30],[0,5,10,15,20],[0,5,10,15,20]][i]
        ax.set_xlim(0,xmax);ax.set_ylim(ph,0);ax.set_xticks(ticks)
        ax.set_yticks(t.plot_y,t.display_label,fontsize=FONT,linespacing=1.1)
        ax.tick_params(axis='y',length=0,pad=1.7)
        ax.tick_params(axis='x',labelsize=7.8,pad=2,length=2.5,width=.6)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_linewidth(.7)
        ax.grid(axis='x',color='#DADADA',lw=.45)
        ax.set_xlabel('−log2(FDR)' if i<2 else '−log2(min FDR)',fontsize=8.1,labelpad=3)
        # Figure 4 has two lower blocks: e = GO/KEGG, f = metabolism.
        # The four condition headings remain, without stale a-d subletters.
        outer_letter = ''
        if outer_letter:
            text_mm(fig,start,top+12,outer_letter,fontsize=10.5,weight='bold',va='top')
        text_mm(fig,start+6,top+12,('Asymptomatic' if i%2==0 else 'Symptomatic')+' TB',fontsize=10.2,weight='semibold',va='top',color='#3E3E3E')
        text_mm(fig,start+6,top+6.5,'GO and KEGG enrichment' if i<2 else 'Metabolic pathway activity',fontsize=8.0,va='top',color='#666666')
    gene_values=[4,15,70]
    handles=[Line2D([],[],marker='o',markersize=np.sqrt(area([v])[0]),mfc='white',mec='#706966',mew=.5,ls='',label=str(v)) for v in gene_values]
    leg1=fig.legend(handles=handles,title='GO/KEGG overlap genes',loc='lower center',bbox_to_anchor=(.28,1/height),ncol=3,fontsize=7.5,title_fontsize=7.8,frameon=False,borderpad=0,labelspacing=.3,columnspacing=1.0)
    handles=[Line2D([],[],marker='D',markersize=np.sqrt(area([v],True)[0]),mfc='white',mec='#706966',mew=.5,ls='',label=f'{v} methods') for v in [2,3]]
    leg2=fig.legend(handles=handles,title='Metabolic pathway support',loc='lower center',bbox_to_anchor=(.76,1/height),ncol=2,fontsize=7.5,title_fontsize=7.8,frameon=False,borderpad=0,labelspacing=.3)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox;issues=[]
    text_objects=[t for ax in axes for t in ax.get_yticklabels()+ax.get_xticklabels()+[ax.xaxis.label]]+list(fig.texts)+bracket_text+module_text+[t for cb in colorbars for t in cb.ax.get_yticklabels() if cb.vmin<=t.get_position()[1]<=cb.vmax]
    for t in text_objects:
        b=t.get_window_extent(renderer)
        if b.x0<bounds.x0 or b.x1>bounds.x1 or b.y0<bounds.y0 or b.y1>bounds.y1:issues.append('clipped: '+t.get_text())
    for i,ax in enumerate(axes):
        labels=ax.get_yticklabels()
        for j,t in enumerate(labels):
            for u in labels[j+1:]:
                if t.get_window_extent(renderer).overlaps(u.get_window_extent(renderer)):issues.append('overlapping labels')
        if i>=2:
            group_labels=module_text[:4] if i==2 else module_text[4:]
            if any(m.get_window_extent(renderer).overlaps(t.get_window_extent(renderer)) for m in group_labels for t in labels):issues.append('module/pathway label overlap')
    for lg in [leg1,leg2]:
        if any(lg.get_window_extent(renderer).overlaps(ax.get_window_extent(renderer)) for ax in axes):issues.append('legend overlaps plot')
    for cb in colorbars:
        ticks=[t for t in cb.ax.get_yticklabels() if cb.vmin<=t.get_position()[1]<=cb.vmax]
        for tick in ticks:
            for m in module_text:
                if tick.get_window_extent(renderer).overlaps(m.get_window_extent(renderer)):issues.append('colourbar/module overlap')
    stem=out/'03_figures'/'Fig4_bottom_new_editable'
    for ext in ['png','pdf','svg']:fig.savefig(stem.with_suffix('.'+ext),dpi=600,facecolor='white')
    with fitz.open(stem.with_suffix('.pdf')) as doc:
        vector=not any(p.get_images() for p in doc)
        embedded=all(doc.extract_font(f[0])[3] for p in doc for f in p.get_fonts())
    checks={'no_clipped_or_overlapping_text':not issues,'source_bytes_unchanged':all(sha(src/r['file'])==r['sha256'] for r in hashes),'vector_PDF_no_rasters':vector,'PDF_fonts_embedded':embedded,'SVG_editable_text':'<text' in stem.with_suffix('.svg').read_text(encoding='utf-8'),'all_source_FDR_below_005':all(t.fdr_display.lt(.05).all() for t in tables)}
    checks['two_GO_and_seven_module_colorbars']=len(colorbars)==9 and all(cb.ax.get_visible() and cb.ax.collections for cb in colorbars)
    checks['seven_distinct_module_hues']=len(color_key)==7 and len({r['base_colour'] for r in color_key})==7
    checks['point_colours_match_module_colorbar']=all(np.allclose(points.get_facecolors(),cmap(norm(points.get_array()))) for points,cmap,norm in mapped_points)
    checks['all_metabolic_pathways_classified']=all(t.module_name.notna().all() for t in tables[2:])
    checks['right_module_brackets_present']=strip_counts==[4,3] and all(m.get_position()[0]>1 for m in module_text)
    checks['GO_identifiers_hidden_in_figure']=all('GO:' not in t.get_text() for t in text_objects)
    checks['module_order_and_descending_significance']=all(t.module_order.is_monotonic_increasing and all(g.x.is_monotonic_decreasing for _,g in t.groupby('module_order')) for t in tables[2:])
    checks['common_colour_limits_within_each_metabolic_panel']=all(len({(r['vmin'],r['vmax']) for r in color_key if r['panel']==p})==1 for p in ['c','d'])
    checks['all_points_inside_x_limits']=all(t.x.le(ax.get_xlim()[1]).all() for t,ax in zip(tables,axes))
    pd.DataFrame(hashes).to_csv(out/'04_qc'/'SOURCE_HASHES.tsv',sep='\t',index=False)
    pd.DataFrame(color_key).to_csv(out/'02_render_data'/'MODULE_COLOR_KEY.tsv',sep='\t',index=False)
    (out/'04_qc'/'ENVIRONMENT.json').write_text(json.dumps({'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','matplotlib','PyMuPDF']},'font':'Arial'},indent=2),encoding='utf-8')
    pd.DataFrame([{'check':k,'status':'PASS' if v else 'FAIL'} for k,v in checks.items()]).to_csv(out/'04_qc'/'QC.tsv',sep='\t',index=False)
    (out/'04_qc'/'LAYOUT.json').write_text(json.dumps({'width_mm':WIDTH,'height_mm':height,'axis_width_mm':22,'axis_height_mm':ph,'pathway_font_pt':FONT,'GO_KEGG_gap_mm':1.7,'issues':issues},indent=2))
    (out/'FIGURE_LEGEND.txt').write_text('a,b, Selected existing significant C05 direct asymptomatic-versus-symptomatic GO biological-process and KEGG enrichment terms. Horizontal position and point colour: -log2(existing BH-FDR); circle size increases with overlapping gene count, using area=12+6*sqrt(count). c,d, Existing C05 ALL_CELLS metabolic pathway summaries with at least two concordant significant methods (AUCell, VISION, ssGSEA). Horizontal position and within-module colour: -log2(minimum existing FDR across methods); diamond size denotes two or three significant methods. The minimum FDR is descriptive, not a newly combined adjusted P value. Each module has a distinct macaron hue and a matching right-side colourbar. All module colourbars WITHIN a panel share the same numerical limits, recorded in MODULE_COLOR_KEY.tsv; numerical limits differ between panels. Modules are display categories, not separately tested enrichment results. Within modules, pathways are sorted by descending -log2(min FDR). These are pathway activity associations, not measured metabolite abundances. GO IDs are removed only from display. The full per-method evidence and exact source rows are supplied in qc. Term selection is presentation-driven and not exhaustive. No enrichment, scoring, model, permutation or FDR was rerun.\n',encoding='utf-8')
    (out/'README.txt').write_text('V13 is a source-verified revision of the four-column lollipop master: 3 GO/KEGG replacements, 2 asymptomatic and 3 symptomatic metabolic additions. All 61 displayed records match existing C05 source rows. Original versions and all scientific results remain unchanged.\nThe KEGG Riboflavin and Pyruvate candidates have only one significant method and were excluded. REACTOME Vitamin B2 riboflavin metabolism and Pyruvate metabolism meet the >=2-method rule and retain their exact source identities.\nReproduce selection verification: python prepare_fig4_v13.py --out .\nReproduce figure: python make_fig4_source_verified_v13.py --input-dir 01_source_locked --module-dir 01_source_locked --output-dir reproduced\nDependencies: numpy, pandas, matplotlib, PyMuPDF; Arial font. Source tables, module assignments, complete row audit and per-method FDR/effect evidence are included.\nThe 366-mm-wide editable master preserves the requested four-column layout. It is not certified for final journal-size legibility: reducing to 183 mm halves the 8.1-pt pathway font.\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'height_mm':height,'QC':'PASS' if all(checks.values()) else 'FAIL','issues':issues},indent=2))
    if not all(checks.values()):raise SystemExit(2)

if __name__=='__main__':main()
