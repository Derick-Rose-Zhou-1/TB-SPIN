from pathlib import Path
import fitz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

p=Path(__file__).resolve().parent
out=p/'discrete_NES';out.mkdir(exist_ok=True)
data=pd.concat([pd.read_csv(f,sep='\t') for f in sorted((p/'GSEA').glob('*.tsv.gz'))],ignore_index=True)
assert data.groupby('branch').database.nunique().eq(6).all()
groups=['DTB','TPE','TBM']
terms=[
('DTB','REACTOME_SIGNALING_BY_THE_B_CELL_RECEPTOR_BCR','B cell receptor\nsignalling',1),
('DTB','GOBP_CELL_KILLING','Cell killing',-1),
('DTB','KEGG_NATURAL_KILLER_CELL_MEDIATED_CYTOTOXICITY','NK cell-mediated\ncytotoxicity',-1),
('TPE','GOBP_ANTIGEN_PROCESSING_AND_PRESENTATION_OF_PEPTIDE_ANTIGEN_VIA_MHC_CLASS_I','MHC I peptide\nantigen presentation',1),
('TPE','KEGG_CYTOSOLIC_DNA_SENSING_PATHWAY','Cytosolic DNA\nsensing',1),
('TPE','KEGG_TRYPTOPHAN_METABOLISM','Tryptophan\nmetabolism',1),
('TBM','GOBP_VACUOLE_ORGANIZATION','Vacuole\norganization',1),
('TBM','GOBP_COMPLEMENT_ACTIVATION_CLASSICAL_PATHWAY','Classical complement\nactivation',1),
('TBM','KEGG_LYSOSOME','Lysosome',1)]
frames=[]
for j,(group,t,label,sign) in enumerate(terms):
    x=data[data.term.eq(t)].set_index('branch').loc[groups].copy()
    assert list(x.index[x.FDR.lt(.05)])==[group] and sign*x.loc[group,'NES']>0
    x['display_group']=group;x['column']=j+1;frames.append(x.reset_index())
src=pd.concat(frames);src.to_csv(out/'Figure5j_source.tsv',sep='\t',index=False)
