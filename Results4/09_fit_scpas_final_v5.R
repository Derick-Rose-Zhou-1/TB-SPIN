#!/usr/bin/env Rscript
options(warn = 1, stringsAsFactors = FALSE)
suppressPackageStartupMessages({
  library(data.table)
  library(Matrix)
  library(preprocessCore)
  library(scPAS)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: 09_fit_scpas_final_v5.R ROOT TASK [OUT_ROOT]")
root <- args[[1]]
task <- args[[2]]
out_root <- if (length(args) >= 3) args[[3]] else file.path(root, "20_EXPANDED_SEVEN_GROUP_MODEL_V5")
outdir <- file.path(out_root, "08_SCPAS", task)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

Sys.setenv(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
ts <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
emit <- function(stage, status, message) {
  cat(sprintf("[%s] [INFO] task=%s method=scPAS_v5_final stage=%s status=%s %s\n", ts(), task, stage, status, message))
  flush.console()
}
rss <- function() {
  z <- readLines("/proc/self/status", warn = FALSE)
  x <- z[grepl("^VmRSS:", z)]
  if (!length(x)) return(NA_real_)
  as.numeric(strsplit(trimws(x), "[[:space:]]+")[[1]][2]) / 1048576
}
fread_tsv <- function(path, ...) {
  if (grepl("\\.gz$", path, ignore.case = TRUE)) fread(cmd = paste("gzip -dc --", shQuote(path)), ...)
  else fread(path, ...)
}
json_escape <- function(x) gsub('"', "'", as.character(x))
write_status <- function(status, extra = "") {
  cat(sprintf('{"task":"%s","status":"%s","timestamp":"%s","message":"%s"}\n',
              task, status, ts(), json_escape(extra)),
      file=file.path(outdir, "V5_FINAL_MODEL_STATUS.json"))
}

cat(sprintf("[%s] [STEP START] V5_FINAL_MODEL | %s\n", ts(), task))
started <- Sys.time()
write_status("RUNNING", "started")

membership_path <- file.path(out_root, "05_TASK_REGISTRY", "TASK_SAMPLE_MEMBERSHIP.tsv")
if (!file.exists(membership_path)) stop("missing membership: ", membership_path)
membership <- fread_tsv(membership_path, sep="\t", data.table=FALSE, colClasses="character")
membership <- membership[membership$task == task, , drop=FALSE]
if (!nrow(membership)) stop("task not found in membership: ", task)
membership$task_label <- as.integer(membership$task_label)
if (anyDuplicated(membership$sample_id)) stop("duplicate sample_id in task membership")
use_samples <- membership$sample_id
y <- membership$task_label; names(y) <- use_samples
if (length(unique(y)) != 2) stop("final model requires two classes")

cohort_matrix <- list(
  PRJNA1364747 = file.path(root, "01_INPUT/bulk/PRJNA1364747/PRJNA1364747_counts_gene_symbol.tsv.gz"),
  GSE193777 = file.path(root, "01_INPUT/bulk/GSE193777/GSE193777_counts_gene_symbol.tsv.gz"),
  GSE101705 = file.path(root, "01_INPUT/bulk/CLASSIC_PILOT/GSE101705_counts_gene_symbol.tsv.gz"),
  SELF400_LOCAL_QZ = file.path(root, "01_INPUT/bulk/SELF400/LOCAL_QZ_counts_gene_symbol.tsv.gz"),
  GSE19439 = file.path(root, "01_INPUT/bulk/EXPANDED_PUBLIC/GSE19439/GSE19439_microarray_gene_expression.tsv.gz"),
  GSE19435 = file.path(root, "01_INPUT/bulk/EXPANDED_PUBLIC/GSE19435/GSE19435_microarray_gene_expression.tsv.gz"),
  GSE83456 = file.path(root, "01_INPUT/bulk/EXPANDED_PUBLIC/GSE83456/GSE83456_microarray_gene_expression.tsv.gz"),
  GSE89403 = file.path(root, "01_INPUT/bulk/EXPANDED_PUBLIC/GSE89403/GSE89403_counts_gene_symbol.tsv.gz"),
  GSE31348 = file.path(root, "01_INPUT/bulk/EXPANDED_PUBLIC/GSE31348/GSE31348_microarray_gene_expression.tsv.gz"),
  GSE36238 = file.path(root, "01_INPUT/bulk/EXPANDED_PUBLIC/GSE36238/GSE36238_microarray_gene_expression.tsv.gz"),
  GSE112104 = file.path(root, "01_INPUT/bulk/EXPANDED_PUBLIC/GSE112104/GSE112104_counts_gene_symbol.tsv.gz")
)

load_cohort <- function(cohort, samples) {
  path <- cohort_matrix[[cohort]]
  if (is.null(path) || !file.exists(path)) stop("missing matrix for cohort: ", cohort)
  dt <- fread_tsv(path, sep="\t", data.table=TRUE, showProgress=FALSE)
  gene_col <- names(dt)[1]
  present <- intersect(samples, names(dt))
  if (!length(present)) return(NULL)
  sub <- dt[, c(gene_col, present), with=FALSE]
  setnames(sub, gene_col, "gene")
  if (anyDuplicated(sub$gene)) sub <- sub[, lapply(.SD, mean), by=gene, .SDcols=present]
  mat <- as.matrix(sub[, ..present]); storage.mode(mat) <- "double"; rownames(mat) <- as.character(sub$gene)
  if (grepl("counts_gene_symbol", path)) {
    libsize <- colSums(mat)
    if (any(!is.finite(libsize)) || any(libsize <= 0)) stop("bad library size in ", cohort)
    mat <- sweep(mat, 2, libsize / 1e6, "/")
    mat <- log2(mat + 1)
  } else {
    q99 <- suppressWarnings(as.numeric(quantile(mat, 0.99, na.rm=TRUE)))
    minv <- suppressWarnings(min(mat, na.rm=TRUE))
    if (is.finite(q99) && is.finite(minv) && minv >= 0 && q99 > 50) mat <- log2(mat + 1)
  }
  rm(dt, sub); gc()
  mat
}

mat_list <- list()
for (cohort in unique(membership$cohort_id)) {
  samples <- membership$sample_id[membership$cohort_id == cohort]
  m <- load_cohort(cohort, samples)
  if (!is.null(m)) mat_list[[cohort]] <- m
}
if (!length(mat_list)) stop("no matrices loaded")
all_genes <- Reduce(intersect, lapply(mat_list, rownames))
if (length(all_genes) < 200) stop("insufficient shared genes across task cohorts: ", length(all_genes))
expr <- do.call(cbind, lapply(mat_list, function(m) m[all_genes, , drop=FALSE]))
expr <- expr[, use_samples, drop=FALSE]
rm(mat_list); gc()

network_genes_path <- file.path(root, "07_SCPAS/NETWORK/SC_NETWORK_GENES.tsv")
network_edges_path <- file.path(root, "07_SCPAS/NETWORK/SC_NETWORK_EDGES.tsv.gz")
if (!file.exists(network_genes_path) || !file.exists(network_edges_path)) stop("missing scPAS network files")
network_genes <- fread_tsv(network_genes_path, sep="\t", data.table=FALSE)$gene
edges <- fread_tsv(network_edges_path, sep="\t", data.table=FALSE)
network <- sparseMatrix(i=c(edges$i_1based, edges$j_1based), j=c(edges$j_1based, edges$i_1based), x=1,
                        dims=c(length(network_genes), length(network_genes)), dimnames=list(network_genes, network_genes))
shared <- intersect(network_genes, rownames(expr))
if (length(shared) < 100) stop("insufficient network overlap: ", length(shared))
emit("LOAD", "COMPLETE", sprintf("samples=%d neg=%d pos=%d genes_shared=%d cohorts=%d rss_gib=%.3f",
                                 length(y), sum(y==0), sum(y==1), length(shared), length(unique(membership$cohort_id)), rss()))

prepare_all <- function() {
  train_idx <- seq_along(y)
  min_n <- max(2L, floor(length(train_idx) * 0.10))
  pass <- rowSums(expr[shared, train_idx, drop=FALSE] > 0) >= min_n
  genes <- shared[pass]
  train0 <- expr[genes, train_idx, drop=FALSE]
  variable <- apply(train0, 1, sd) > 0
  genes <- genes[variable]
  train0 <- train0[variable,,drop=FALSE]
  target <- normalize.quantiles.determine.target(train0)
  train <- normalize.quantiles.use.target(train0, target)
  dimnames(train) <- dimnames(train0)
  list(genes=genes, train=t(train), target=target)
}
stratified_fold <- function(yy, k, seed) {
  set.seed(seed); out <- integer(length(yy))
  for (cl in sort(unique(yy))) {
    ids <- sample(which(yy == cl)); out[ids] <- rep(seq_len(k), length.out=length(ids))
  }
  out
}
alpha_grid <- c(0.01, 0.1, 0.3, 0.5, 0.7)
fit_model <- function(x, yy, genes, seed) {
  inner_k <- min(3L, min(table(yy)))
  if (inner_k < 2L) stop("inner CV fewer than two folds")
  foldid <- stratified_fold(yy, inner_k, seed)
  omega <- network[genes, genes, drop=FALSE]
  best <- NULL
  for (alpha in alpha_grid) {
    emit("INNER_FIT", "RUNNING", sprintf("alpha=%.4g inner_folds=%d rss_gib=%.3f", alpha, inner_k, rss()))
    fit <- APML0(x=x, y=yy, family="binomial", penalty="Net", Omega=omega, alpha=alpha, nlambda=25, nfolds=inner_k, foldid=foldid)
    if (is.null(fit) || is.null(fit$lambda.min)) next
    score <- suppressWarnings(min(fit$fit$cvm, na.rm=TRUE))
    emit("INNER_FIT", "COMPLETE", sprintf("alpha=%.4g lambda=%.8g inner_loss=%.8g", alpha, fit$lambda.min, score))
    if (is.finite(score) && (is.null(best) || score < best$score)) best <- list(alpha=alpha, lambda=fit$lambda.min, score=score)
  }
  if (is.null(best)) stop("no valid APML0 fit")
  emit("FINAL_FIT", "RUNNING", sprintf("alpha=%.4g lambda=%.8g rss_gib=%.3f", best$alpha, best$lambda, rss()))
  model <- APML0(x=x, y=yy, family="binomial", penalty="Net", Omega=omega, alpha=best$alpha, lambda=best$lambda, nfolds=1)
  beta <- as.numeric(model$Beta[,1]); names(beta) <- c("(Intercept)", genes)
  list(beta=beta, alpha=best$alpha, lambda=best$lambda, inner_loss=best$score)
}

dat <- prepare_all()
fit <- fit_model(dat$train, y, dat$genes, 20261805)
final_coef <- data.frame(gene=names(fit$beta), coefficient=fit$beta,
                         alpha=fit$alpha, lambda=fit$lambda,
                         training_samples=length(y), negative_samples=sum(y==0), positive_samples=sum(y==1),
                         task=task, model_type="FINAL_DISCOVERY_V5")
fwrite(final_coef, file.path(outdir, "FINAL_MODEL_COEFFICIENTS.tsv"), sep="\t")
manifest <- data.frame(task=task, method="scPAS_v5_official_APML0", status="FINAL_MODEL_COMPLETE",
                       samples=length(y), participants=length(unique(membership$participant_id)),
                       negative=sum(y==0), positive=sum(y==1), cohorts=length(unique(membership$cohort_id)),
                       shared_network_genes=length(shared), final_genes=length(dat$genes),
                       final_alpha=fit$alpha, final_lambda=fit$lambda, inner_loss=fit$inner_loss,
                       preprocessing="V5_bulk_loader;training_low_expression_filter;training_quantile_target",
                       runtime_seconds=as.numeric(difftime(Sys.time(), started, units="secs")),
                       gpu_visible=Sys.getenv("CUDA_VISIBLE_DEVICES", "UNSET"), rss_gib_final=rss())
fwrite(manifest, file.path(outdir, "FINAL_MODEL_MANIFEST.tsv"), sep="\t")
write_status("COMPLETE", sprintf("final_genes=%d runtime_seconds=%.1f", length(dat$genes), manifest$runtime_seconds))
emit("FINAL_DISCOVERY_MODEL", "COMPLETE", sprintf("genes=%d alpha=%.4g lambda=%.8g runtime_s=%.1f rss_gib=%.3f", length(dat$genes), fit$alpha, fit$lambda, manifest$runtime_seconds, rss()))
cat(sprintf("[%s] [STEP DONE] V5_FINAL_MODEL | %s COMPLETE\n", ts(), task))
