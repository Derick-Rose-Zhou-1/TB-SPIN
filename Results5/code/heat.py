from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


import os
package=Path(__file__).resolve().parents[1]
root=Path(os.environ["FIG5_WORK"])
df=pd.read_csv(package/"results"/"GSEA_all.tsv.gz",sep="\t")
rows=["DTB","TPE","TBM"]
terms=[
("HALLMARK_TNFA_SIGNALING_VIA_NFKB","TNFα via\nNF-κB"),
("HALLMARK_MTORC1_SIGNALING","mTORC1\nsignalling"),
("HALLMARK_TGF_BETA_SIGNALING","TGF-β\nsignalling"),
("HALLMARK_CHOLESTEROL_HOMEOSTASIS","Cholesterol\nhomeostasis"),
("GOMF_EXTRACELLULAR_MATRIX_STRUCTURAL_CONSTITUENT_CONFERRING_TENSILE_STRENGTH","ECM tensile\nstrength"),
("REACTOME_ASSEMBLY_OF_COLLAGEN_FIBRILS_AND_OTHER_MULTIMERIC_STRUCTURES","Collagen fibril\nassembly"),
("GOMF_G_PROTEIN_COUPLED_RECEPTOR_ACTIVITY","GPCR\nactivity"),
("REACTOME_FRS_MEDIATED_FGFR2_SIGNALING","FRS-mediated\nFGFR2 signalling"),
("GOCC_CYTOSOLIC_LARGE_RIBOSOMAL_SUBUNIT","Cytosolic large\nribosomal subunit"),
("REACTOME_EUKARYOTIC_TRANSLATION_ELONGATION","Translation\nelongation"),
("REACTOME_NONSENSE_MEDIATED_DECAY_NMD","Nonsense-mediated\nmRNA decay"),
("REACTOME_RESPONSE_OF_EIF2AK4_GCN2_TO_AMINO_ACID_DEFICIENCY","GCN2 amino acid\ndeficiency response")]
data=[]
for k,(term,label) in enumerate(terms):
    z=df[df.term.eq(term)].set_index("branch").loc[rows].copy()
    group=rows[k//4]
    pos=z.NES.gt(0)&z.FDR.lt(.05)
    assert list(z.index[pos])==[group] and z.NES.idxmax()==group
    z["display_group"]=group
    z["display_column"]=k+1
    data.append(z.reset_index())
data=pd.concat(data,ignore_index=True)
def stars(q):
    return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else ""
data["stars"] = data.FDR.map(stars)
data.to_csv(root/"plot_data.tsv",sep="\t",index=False)
def mat(key):
    return data.pivot(index="branch",columns="display_column",values=key).loc[rows].values
v,q=mat("NES"),mat("FDR")
palettes = [
    (code, positive, mpl.colors.LinearSegmentedColormap.from_list(
        code, ["#55548a", "#faf8f5", positive], N=257))
    for code, positive in [("coral", "#f48563"), ("rose", "#a24a54")]
]
mpl.rcParams.update({"pdf.fonttype":42,"font.family":"DejaVu Sans","font.size":10})
def draw(fig,ax,cmap):
    im=ax.imshow(v,aspect="auto",cmap=cmap,vmin=-3,vmax=3,interpolation="nearest")
    ax.set_yticks(range(3),rows,fontweight="bold")
    ax.set_xticks(range(12),[label for _,label in terms],fontsize=9)
    ax.tick_params(length=0,pad=5)
    ax.set_xticks(np.arange(13)-.5,minor=True);ax.set_yticks(np.arange(4)-.5,minor=True)
    ax.tick_params(which="minor",length=0)
    ax.grid(which="minor",color="white",lw=.9)
    for x in [3.5,7.5]: ax.axvline(x,color="white",lw=4)
    for spine in ax.spines.values(): spine.set_visible(False)
    for i,j in np.ndindex(v.shape):
        rgb=np.array(cmap(im.norm(v[i,j]))[:3])
        lum=np.dot(np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4),[.2126,.7152,.0722])
        ax.text(j,i,f"{v[i,j]:+.2f}"+stars(q[i,j]),ha="center",va="center",color="white" if lum<.179 else "black",fontsize=10)
    return im
fig,ax=plt.subplots(figsize=(15.2,2.8));fig.subplots_adjust(left=.05,right=.994,top=.92,bottom=.42)
im=draw(fig,ax,palettes[0][2])
fig.text(.009,.90,"j",fontweight="bold",fontsize=17)
cb=fig.colorbar(im,cax=fig.add_axes([.22,.12,.14,.035]),orientation="horizontal",ticks=[-3,0,3]);cb.outline.set_visible(False)
fig.text(.19,.12,"NES");fig.text(.40,.12,"* q < 0.05     ** q < 0.01     *** q < 0.001     (within-disease, within-library GSEA FDR)")
fig.savefig(root/"heat.pdf")

