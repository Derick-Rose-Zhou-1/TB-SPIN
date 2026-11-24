from pathlib import Path
from io import BytesIO
import fitz
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np

P=Path(__file__).resolve().parent
out=P/'bidirectional_functions'
out.mkdir(exist_ok=True)
pair=pd.read_csv(P/'ORA_all.tsv.gz',sep='\t').query("method=='paired'")
gsea=pd.concat([pd.read_csv(f,sep='\t') for f in (P/'GSEA').glob('*.tsv.gz')])
# Within each block: 2 GO BP and 2 KEGG up, followed by 2 GO BP and 2 KEGG down.
terms={
'DTB':[
('GOBP_B_CELL_ACTIVATION','B cell activation'),('GOBP_PEPTIDE_ANTIGEN_ASSEMBLY_WITH_MHC_CLASS_II_PROTEIN_COMPLEX','Peptide antigen–MHC II assembly'),
('KEGG_B_CELL_RECEPTOR_SIGNALING_PATHWAY','B cell receptor signalling'),('KEGG_ANTIGEN_PROCESSING_AND_PRESENTATION','Antigen processing and presentation'),
('GOBP_T_CELL_ACTIVATION','T cell activation'),('GOBP_NATURAL_KILLER_CELL_MEDIATED_IMMUNITY','NK cell-mediated immunity'),
('KEGG_NATURAL_KILLER_CELL_MEDIATED_CYTOTOXICITY','NK cell-mediated cytotoxicity'),('KEGG_T_CELL_RECEPTOR_SIGNALING_PATHWAY','T cell receptor signalling')],
'TPE':[
('GOBP_CHEMOKINE_PRODUCTION','Chemokine production'),('GOBP_NEUTROPHIL_CHEMOTAXIS','Neutrophil chemotaxis'),
('KEGG_TRYPTOPHAN_METABOLISM','Tryptophan metabolism'),('KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY','NOD-like receptor signalling'),
('GOBP_T_CELL_ACTIVATION','T cell activation'),('GOBP_T_CELL_DIFFERENTIATION','T cell differentiation'),
('KEGG_T_CELL_RECEPTOR_SIGNALING_PATHWAY','T cell receptor signalling'),('KEGG_CELL_ADHESION_MOLECULES_CAMS','Cell adhesion molecules')],
'TBM':[
('GOBP_MYELOID_LEUKOCYTE_ACTIVATION','Myeloid leukocyte activation'),('GOBP_COMPLEMENT_ACTIVATION','Complement activation'),
('KEGG_LYSOSOME','Lysosome'),('KEGG_FC_GAMMA_R_MEDIATED_PHAGOCYTOSIS','FcγR-mediated phagocytosis'),
('GOBP_NATURAL_KILLER_CELL_MEDIATED_IMMUNITY','NK cell-mediated immunity'),('GOBP_LEUKOCYTE_MEDIATED_CYTOTOXICITY','Leukocyte-mediated cytotoxicity'),
('KEGG_NATURAL_KILLER_CELL_MEDIATED_CYTOTOXICITY','NK cell-mediated cytotoxicity'),('KEGG_T_CELL_RECEPTOR_SIGNALING_PATHWAY','T cell receptor signalling')]}
colors={'up':'#f48563','down':'#55548a'}
plt.rcParams.update({'font.family':'Arial','pdf.fonttype':42,'font.size':5.2,'axes.linewidth':.45,'xtick.major.size':2})
fig=plt.figure(figsize=(180/25.4,48/25.4))
rows=[]
for i,(target,items) in enumerate(terms.items()):
    ax=fig.add_axes([i/3+.19,.23,.13,.64])
    ticks=[]
    for j,(term,label) in enumerate(items):
        direction='up' if j<4 else 'down'
        r=pair[(pair.group==target)&(pair.direction==direction)&(pair.term==term)]
        assert len(r)==1 and r.iloc[0].full_BH<.05 and r.iloc[0].BH_FDR<.05
        row=r.iloc[0].to_dict();row.update(panel='ghi'[i],display_name=label)
        rows.append(row)
        suffix='GO BP' if row['library']=='GO_BP' else 'KEGG'
        if len(label)>24:
            words=label.split(); mid=min(range(1,len(words)),key=lambda k:abs(len(' '.join(words[:k]))-len(' '.join(words[k:])+' ['+suffix+']')))
            label=' '.join(words[:mid])+'\n'+' '.join(words[mid:])
        ticks.append(label+' ['+suffix+']')
        ax.barh(j,-np.log10(row['full_BH']),color=colors[direction],height=.59)
    ax.set_yticks(range(8),ticks,fontsize=5.1);ax.tick_params(axis='y',length=0,pad=1.4);ax.set_ylim(7.6,-.6);ax.set_xlim(left=0)
    ax.spines[['right','top']].set_visible(False);ax.set_xlabel('−log$_{10}$(FDR)',labelpad=1)
    fig.text(i/3+.013,.91,'ghi'[i],weight='bold',size=8)
    fig.text(i/3+.17,.92,target,weight='bold',size=6,ha='center')
fig.legend(handles=[Patch(color=c,label=t) for c,t in [(colors['up'],'Upregulated genes'),(colors['down'],'Downregulated genes')]],loc='upper center',bbox_to_anchor=(.5,.06),ncol=2,frameon=False,fontsize=5.2,borderaxespad=0,handlelength=1)
panel=out/'Figure5_g-i.pdf';fig.savefig(panel);fig.savefig(panel.with_suffix('.png'),dpi=350);plt.close(fig)
data=pd.DataFrame(rows)
keys=data[['term','library','direction']].drop_duplicates()
audit=keys.merge(pair,on=['term','library','direction'],how='left').merge(gsea[['branch','database','term','NES','FDR']],left_on=['group','library','term'],right_on=['branch','database','term'],how='left')
audit.to_csv(out/'cross_phenotype.tsv',sep='\t',index=False)
data.merge(gsea[['branch','database','term','NES','FDR']],left_on=['group','library','term'],right_on=['branch','database','term'],how='left',validate='one_to_one').to_csv(out/'panel_source.tsv',sep='\t',index=False)

