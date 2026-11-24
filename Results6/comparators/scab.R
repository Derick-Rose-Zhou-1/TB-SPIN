#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(scAB);library(Seurat);library(Matrix);library(preprocessCore)})
# Local Matrix generic bindings for official function body; no package edits.
compat<-new.env(parent=asNamespace('scAB'))
compat$t<-Matrix::t;compat$diag<-Matrix::diag;compat$norm<-Matrix::norm
scABSparse<-scAB::scAB;environment(scABSparse)<-compat
root <- Sys.getenv("RESULT6_FORMAL_ROOT", unset="inputs/result6_formal_root")
inp <- file.path(root,'00_INPUT'); out <- file.path(root,'03_SCAB')
dir.create(out, recursive=TRUE, showWarnings=FALSE)
stamp <- function(s) {
  cat(format(Sys.time(),'%Y-%m-%d %H:%M:%S %z'),s,'\n');flush.console()
  write.table(data.frame(pid=Sys.getpid(),stage=s,time=as.character(Sys.time())),
              file.path(out,'CURRENT_STAGE.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
}
# Exact create_scAB equations, changing dense storage to sparse storage only.
# The installed scAB model and select_K functions remain untouched.
graph_terms <- function(graph) {
  A <- as(graph,'dgCMatrix');diag(A)<-0;A<-drop0(A);A@x[]<-1
  degree <- rowSums(A);stopifnot(all(degree>0))
  D <- Diagonal(x=degree);D12<-Diagonal(x=1/sqrt(degree))
  list(L=D12%*%(D-A)%*%D12,D=D12%*%D%*%D12,A=D12%*%A%*%D12)
}
make_data <- function(X,graph,Y,method) {
  G <- graph_terms(graph)
  S <- if(method=='survival') {
    ss <- scAB::guanrank(Y[,c('time','status')]);diag(1-ss[rownames(Y),3])
  } else diag(1-Y)
  obj <- list(X=X/norm(X,'F'),S=S,L=G$L,D=G$D,A=G$A,phenotype=Y,method=method)
  class(obj)<-'scAB_data';obj
}
equivalence_gate <- function() {
  stamp('SPARSE_GRAPH_OFFICIAL_EQUATION_EQUIVALENCE')
  set.seed(20260905)
  A<-matrix(0,24,24);A[cbind(1:23,2:24)]<-1;A<-A+t(A)
  D<-diag(rowSums(A));D12<-diag(1/sqrt(rowSums(A)))
  dense<-list(L=D12%*%(D-A)%*%D12,D=D12%*%D%*%D12,A=D12%*%A%*%D12)
  sp<-graph_terms(as(A,'dgCMatrix'))
  err<-max(vapply(names(dense),function(k) max(abs(dense[[k]]-as.matrix(sp[[k]]))),0.0))
  X<-matrix(runif(30*24),30,24);X<-X/norm(X,'F')
  d<-c(list(X=X,S=diag(rep(c(0,1),15)),phenotype=rep(c(0,1),15),method='binary'),dense)
  s<-c(list(X=X,S=d$S,phenotype=d$phenotype,method='binary'),sp)
  class(d)<-class(s)<-'scAB_data'
  a<-scAB::scAB(d,K=3,maxiter=30);b<-scABSparse(s,K=3,maxiter=30)
  model_err<-max(abs(as.matrix(a$H)-as.matrix(b$H)),abs(as.matrix(a$W)-as.matrix(b$W)),abs(a$loss-b$loss))
  write.table(data.frame(graph_max_error=err,official_model_max_error=model_err,
              pass=err<1e-12 && model_err<1e-8,installed_package_modified=FALSE),
              file.path(out,'SPARSE_DENSE_EQUIVALENCE.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
  stopifnot(err<1e-12,model_err<1e-8)
}
main<-function() {
  equivalence_gate()
  if('--test-only' %in% commandArgs(TRUE)) return(invisible(NULL))
  stopifnot(file.exists(file.path(inp,'INPUT_READY.json')))
  mem<-readLines('/proc/meminfo')
  avail<-as.numeric(strsplit(trimws(sub('MemAvailable:','',mem[grepl('^MemAvailable:',mem)])),' +')[[1]][1])*1024
  stopifnot(avail>220*1024^3)
  cells<-readLines(file.path(inp,'CELLS.txt'));genes<-readLines(file.path(inp,'GENES.txt'))
  stopifnot(length(cells)==214402,!anyDuplicated(cells))
  stamp('READ_FULL_FORMAL_RAW_COUNTS')
  x<-as(readMM(file.path(inp,'FORMAL_RAW_COUNTS.mtx')),'dgCMatrix');dimnames(x)<-list(genes,cells)
  obj<-CreateSeuratObject(x,min.cells=0,min.features=0);rm(x);gc()
  # Same numerical preprocessing as official run_seurat; omit unused UMAP display.
  obj<-NormalizeData(obj,normalization.method='LogNormalize',scale.factor=10000)
  obj<-FindVariableFeatures(obj,nfeatures=3000,selection.method='vst')
  obj<-ScaleData(obj);obj<-RunPCA(obj,features=VariableFeatures(obj))
  obj<-FindNeighbors(obj,dims=1:40)
  stopifnot(identical(colnames(obj),cells))
  graph<-obj@graphs$RNA_snn;expr<-obj@assays$RNA@data
  bulk<-as.matrix(read.delim(file.path(inp,'BULK_LOG2CPM.tsv'),row.names=1,check.names=FALSE))
  common<-intersect(rownames(expr),colnames(bulk))
  epdata<-read.delim(file.path(inp,'BASELINE_ENDPOINTS.tsv'),check.names=FALSE)
  coxsource<-Sys.getenv("RESULT6_PARTICIPANT_ENDPOINTS", unset="inputs/PARTICIPANT_ENDPOINTS.tsv")
  cox<-read.delim(coxsource);cox<-cox[cox$endpoint=='E_COX',]
  stopifnot(all(cox$baseline),all(cox$eligible),!anyDuplicated(paste(cox$cohort,cox$participant_id)))
  rm(obj);gc()
  write.table(data.frame(parameter=c('genes','nfeatures','neighbor_dims','alpha','alpha_2','K','maxiter','native_threshold','graph_storage'),
     value=c('all formal genes','3000','1:40','0.005','0.005','official select_K defaults','2000','official findModule/findSubset','sparse exact-equation input')),
     file.path(out,'SCAB_PARAMETERS.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
  for(ep in c('COX','H2','H3')) {
    stamp(paste(ep,'FULL_FORMAL_QUANTILE_NORMALIZATION'))
    clinical<-if(ep=='COX') cox else epdata[epdata$endpoint==if(ep=='H2') 'E_6M' else 'E_12M',]
    stopifnot(nrow(clinical)>0,all(clinical$GSM %in% rownames(bulk)),
      !anyDuplicated(paste(clinical$cohort,clinical$participant_id)))
    dataset<-cbind(t(bulk[clinical$GSM,common,drop=FALSE]),as.matrix(expr[common,,drop=FALSE]))
    dataset<-normalize.quantiles(dataset,copy=FALSE)
    nb<-nrow(clinical);X<-cor(dataset[,seq_len(nb),drop=FALSE],dataset[,-seq_len(nb),drop=FALSE])
    dimnames(X)<-list(clinical$GSM,cells);rm(dataset);gc()
    stopifnot(all(is.finite(X)))
    Y<-if(ep=='COX') data.frame(time=clinical$time_to_event_or_censor_days,status=clinical$y_true,row.names=clinical$GSM) else clinical$y_true
    if(ep=='COX') stopifnot(all(is.finite(Y$time)),all(Y$time>0))
    object<-make_data(X,graph,Y,if(ep=='COX') 'survival' else 'binary')
    stamp(paste(ep,'OFFICIAL_SELECT_K'))
    K<-scAB::select_K(object,verbose=TRUE)
    stamp(paste(ep,'OFFICIAL_SCAB_FIT_K',K))
    fit<-scABSparse(object,K=K)
    fit$H<-as.matrix(fit$H);fit$W<-as.matrix(fit$W)
    stopifnot(all(is.finite(fit$H)),all(is.finite(fit$W)))
    saveRDS(fit,file.path(out,paste0('SCAB_',ep,'_OFFICIAL_FIT.rds')))
    modules<-scAB::findModule(fit$H,tred=2,do.dip=ep!='COX')
    native<-rep('Other cells',length(cells));native[unique(unlist(modules))]<-'scAB+ cells'
    output<-data.frame(cell_id=cells,score=NA_real_,native_state_if_available=native,t(fit$H),check.names=FALSE)
    colnames(output)[4:ncol(output)]<-paste0('official_module_',seq_len(K),'_loading')
    write.table(output,file.path(out,paste0('SCAB_',ep,'_214402.tsv')),sep='\t',row.names=FALSE,quote=FALSE,na='NA')
    stamp(paste(ep,'COMPLETE_214402_MULTIMODULE_NO_OFFICIAL_SCALAR'))

  }
  stamp('SCAB_ALL_ENDPOINTS_COMPLETE')
}
tryCatch(main(),error=function(e){stamp(paste('FAILED',conditionMessage(e)));stop(e)})
