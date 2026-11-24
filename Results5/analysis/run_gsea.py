from pathlib import Path
import sys,os,json,time,hashlib
import numpy as np,pandas as pd,gseapy
P=Path(__file__).resolve().parent
prior=P.parent/'inputs'
import gsea_common as cfg
from run_result6_gsea import normalize_result
g=sys.argv[1];source=P/f'{g}_paired.tsv.gz'
d=pd.read_csv(source,sep='\t').rename(columns={'pvalue':'raw_P','log2FoldChange':'log2FC'})
assert d.gene.is_unique and gseapy.__version__=='1.3.1'
r=cfg.make_rank(d.loc[np.isfinite(d.raw_P)&np.isfinite(d.log2FC),['gene','raw_P','log2FC']])
assert cfg.PERMUTATIONS==10000 and cfg.SEED==20260830
folder=P/'GSEA';folder.mkdir(exist_ok=True)
for db in ['HALLMARK','KEGG','GO_CC','GO_MF','REACTOME','GO_BP']:
    f=folder/f'{g}_{db}.tsv.gz'
    if f.is_file():continue
    gmt=prior/'resources'/cfg.GMT_PATHS[db].name
    obj=gseapy.prerank(rnk=r[['gene','gsea_input_rank']],gene_sets=str(gmt),permutation_num=10000,min_size=10,max_size=500,weight=1,seed=20260830,threads=12,ascending=False,outdir=None,no_plot=True,verbose=False)
    table=normalize_result(obj.res2d,g,db);table.to_csv(f,sep='\t',index=False)
    print(db,len(table),flush=True)
    del obj


