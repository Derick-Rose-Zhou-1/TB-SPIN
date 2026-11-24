#!/usr/bin/env Rscript
options(warn = 1, stringsAsFactors = FALSE)
suppressPackageStartupMessages({
  library(data.table)
  library(Matrix)
  library(preprocessCore)
  library(scPAS)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: 08_run_scpas_v5.R ROOT TASK [OUT_ROOT] [MODE]")
root <- args[[1]]
task <- args[[2]]
out_root <- if (length(args) >= 3) args[[3]] else file.path(root, "20_EXPANDED_SEVEN_GROUP_MODEL_V5")
mode <- if (length(args) >= 4) args[[4]] else "full"
base_dir <- if (mode == "pilot") "08_SCPAS_PILOT" else "08_SCPAS"
outdir <- file.path(out_root, base_dir, task)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

Sys.setenv(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
ts <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
emit <- function(stage, status, message) {
  cat(sprintf("[%s] [INFO] task=%s method=scPAS_v5 mode=%s stage=%s status=%s %s\n", ts(), task, mode, stage, status, message))
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
write_status <- function(status, extra = "") {
  cat(sprintf('{"task":"%s","mode":"%s","status":"%s","timestamp":"%s","message":"%s"}\n',
              task, mode, status, ts(), gsub('"', "'", extra)),
      file=file.path(outdir, "RUN_STATUS.json"))
}

cat(sprintf("[%s] [STEP START] STEP 13 | scPAS V5 %s %s\n", ts(), task, mode))
started <- Sys.time()
write_status("RUNNING", "started")

membership_path <- file.path(out_root, "05_TASK_REGISTRY", "TASK_SAMPLE_MEMBERSHIP.tsv")
outer_path <- file.path(out_root, "07_SPLIT_MANIFEST", "OUTER_SPLITS.tsv")
if (!file.exists(membership_path)) stop("missing membership: ", membership_path)
if (!file.exists(outer_path)) stop("missing outer splits: ", outer_path)
membership <- fread_tsv(membership_path, sep="\t", data.table=FALSE, colClasses="character")
membership <- membership[membership$task == task, , drop=FALSE]
if (!nrow(membership)) stop("task not found in membership: ", task)
membership$task_label <- as.integer(membership$task_label)
if (anyDuplicated(membership$sample_id)) stop("duplicate sample_id in task membership")
use_samples <- membership$sample_id
y <- membership$task_label; names(y) <- use_samples

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
    if (is.finite(q99) && is.finite(minv) && minv >= 0 && q99 > 50) {
      mat <- log2(mat + 1)
    }
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

outer <- fread_tsv(outer_path, sep="\t", data.table=FALSE, colClasses="character")
outer <- outer[outer$task == task, , drop=FALSE]
outer$task_label <- as.integer(outer$task_label)
fold_keys <- unique(outer[, c("validation_design","repeat_seed","outer_fold","heldout_unit"), drop=FALSE])
if (mode == "pilot") fold_keys <- fold_keys[seq_len(min(1, nrow(fold_keys))), , drop=FALSE]

prepare_fold <- function(train_idx, test_idx) {
  min_n <- max(2L, floor(length(train_idx) * 0.10))
  pass <- rowSums(expr[shared, train_idx, drop=FALSE] > 0) >= min_n
  genes <- shared[pass]
  train0 <- expr[genes, train_idx, drop=FALSE]
  test0 <- expr[genes, test_idx, drop=FALSE]
  variable <- apply(train0, 1, sd) > 0
  genes <- genes[variable]; train0 <- train0[variable,,drop=FALSE]; test0 <- test0[variable,,drop=FALSE]
  target <- normalize.quantiles.determine.target(train0)
  train <- normalize.quantiles.use.target(train0, target)
  test <- normalize.quantiles.use.target(test0, target)
  dimnames(train) <- dimnames(train0); dimnames(test) <- dimnames(test0)
  list(genes=genes, train=t(train), test=t(test))
}
stratified_fold <- function(yy, k, seed) {
  set.seed(seed); out <- integer(length(yy))
  for (cl in sort(unique(yy))) {
    ids <- sample(which(yy == cl)); out[ids] <- rep(seq_len(k), length.out=length(ids))
  }
  out
}
alpha_grid <- c(0.01, 0.1, 0.3, 0.5, 0.7)
fit_model <- function(x, yy, genes, seed, fold_name) {
  inner_k <- min(3L, min(table(yy)))
  if (inner_k < 2L) stop("inner CV fewer than two folds")
  foldid <- stratified_fold(yy, inner_k, seed)
  omega <- network[genes, genes, drop=FALSE]
  best <- NULL
  for (alpha in alpha_grid) {
    emit("INNER_FIT", "RUNNING", sprintf("outer=%s alpha=%.4g inner_folds=%d rss_gib=%.3f", fold_name, alpha, inner_k, rss()))
    fit <- APML0(x=x, y=yy, family="binomial", penalty="Net", Omega=omega, alpha=alpha, nlambda=25, nfolds=inner_k, foldid=foldid)
    if (is.null(fit) || is.null(fit$lambda.min)) next
    score <- suppressWarnings(min(fit$fit$cvm, na.rm=TRUE))
    if (is.finite(score) && (is.null(best) || score < best$score)) best <- list(alpha=alpha, lambda=fit$lambda.min, score=score)
  }
  if (is.null(best)) stop("no valid APML0 fit")
  emit("FINAL_FIT", "RUNNING", sprintf("outer=%s alpha=%.4g lambda=%.8g rss_gib=%.3f", fold_name, best$alpha, best$lambda, rss()))
  model <- APML0(x=x, y=yy, family="binomial", penalty="Net", Omega=omega, alpha=best$alpha, lambda=best$lambda, nfolds=1)
  beta <- as.numeric(model$Beta[,1]); names(beta) <- c("(Intercept)", genes)
  list(beta=beta, alpha=best$alpha, lambda=best$lambda, inner_loss=best$score)
}

pred_rows <- list(); coef_rows <- list(); fold_rows <- list(); skipped_rows <- list()
for (i in seq_len(nrow(fold_keys))) {
  key <- fold_keys[i,,drop=FALSE]
  split <- outer[outer$validation_design == key$validation_design & outer$repeat_seed == key$repeat_seed & outer$outer_fold == key$outer_fold, , drop=FALSE]
  train_ids <- split$sample_id[split$split_role=="train"]; test_ids <- split$sample_id[split$split_role=="test"]
  train_idx <- match(train_ids, use_samples); train_idx <- train_idx[!is.na(train_idx)]
  test_idx <- match(test_ids, use_samples); test_idx <- test_idx[!is.na(test_idx)]
  fold_name <- paste(key$validation_design, key$repeat_seed, key$outer_fold, sep="::")
  fold_start <- Sys.time()
  if (length(unique(y[train_idx])) < 2 || length(unique(y[test_idx])) < 1 || length(test_idx) == 0) {
    reason <- ifelse(length(unique(y[train_idx])) < 2, "TRAINING_SINGLE_CLASS", "NO_TEST_SAMPLES")
    emit("OUTER_FOLD", "SKIPPED", sprintf("fold=%s reason=%s train=%d test=%d", fold_name, reason, length(train_idx), length(test_idx)))
    skipped_rows[[length(skipped_rows)+1]] <- data.frame(validation_design=key$validation_design, repeat_seed=key$repeat_seed, outer_fold=key$outer_fold, train_n=length(train_idx), test_n=length(test_idx), status="SKIPPED", reason=reason)
    next
  }
  emit("OUTER_FOLD", "STARTING", sprintf("fold=%s train=%d test=%d train_neg=%d train_pos=%d test_neg=%d test_pos=%d", fold_name, length(train_idx), length(test_idx), sum(y[train_idx]==0), sum(y[train_idx]==1), sum(y[test_idx]==0), sum(y[test_idx]==1)))
  dat <- prepare_fold(train_idx, test_idx)
  fit <- fit_model(dat$train, y[train_idx], dat$genes, 20260806+i, fold_name)
  prob <- plogis(as.numeric(fit$beta[1] + dat$test %*% fit$beta[-1]))
  pred_rows[[length(pred_rows)+1]] <- data.frame(task=task, validation_design=key$validation_design, repeat_seed=key$repeat_seed, outer_fold=key$outer_fold, sample_id=use_samples[test_idx], participant_id=membership$participant_id[test_idx], cohort_id=membership$cohort_id[test_idx], study_family_id=membership$study_family_id[test_idx], source_cohort=membership$source_cohort[test_idx], observed_label=y[test_idx], probability=prob)
  coef_rows[[length(coef_rows)+1]] <- data.frame(validation_design=key$validation_design, repeat_seed=key$repeat_seed, outer_fold=key$outer_fold, gene=names(fit$beta), coefficient=fit$beta, alpha=fit$alpha, lambda=fit$lambda, gene_count=length(dat$genes))
  fold_rows[[length(fold_rows)+1]] <- data.frame(validation_design=key$validation_design, repeat_seed=key$repeat_seed, outer_fold=key$outer_fold, train_n=length(train_idx), test_n=length(test_idx), train_negative=sum(y[train_idx]==0), train_positive=sum(y[train_idx]==1), test_negative=sum(y[test_idx]==0), test_positive=sum(y[test_idx]==1), genes=length(dat$genes), alpha=fit$alpha, lambda=fit$lambda, inner_loss=fit$inner_loss, runtime_seconds=as.numeric(difftime(Sys.time(), fold_start, units="secs")), rss_gib=rss(), status="COMPLETE")
  emit("OUTER_FOLD", "COMPLETE", sprintf("fold=%s genes=%d runtime_s=%.1f", fold_name, length(dat$genes), as.numeric(difftime(Sys.time(), fold_start, units="secs"))))
}
raw_pred <- if (length(pred_rows)) rbindlist(pred_rows) else data.table()
coef_dt <- if (length(coef_rows)) rbindlist(coef_rows) else data.table()
fold_dt <- if (length(fold_rows)) rbindlist(fold_rows) else data.table()
skip_dt <- if (length(skipped_rows)) rbindlist(skipped_rows) else data.table()
fwrite(raw_pred, file.path(outdir, "OOF_PREDICTIONS_PER_FOLD.tsv"), sep="\t")
fwrite(raw_pred, file.path(outdir, "OOF_PREDICTIONS.tsv"), sep="\t")
fwrite(coef_dt, file.path(outdir, "FOLD_GENE_COEFFICIENTS.tsv"), sep="\t")
fwrite(fold_dt, file.path(outdir, "FOLD_PARAMETERS.tsv"), sep="\t")
if (nrow(skip_dt)) fwrite(skip_dt, file.path(outdir, "SKIPPED_FOLDS.tsv"), sep="\t")
status <- if (nrow(raw_pred)) "COMPLETE" else "FAILED_NO_PREDICTIONS"
write_status(status, sprintf("pred_rows=%d fold_rows=%d runtime_seconds=%.1f", nrow(raw_pred), nrow(fold_dt), as.numeric(difftime(Sys.time(), started, units="secs"))))
cat(sprintf("[%s] [STEP DONE] STEP 13 | scPAS V5 %s %s status=%s pred_rows=%d\n", ts(), task, mode, status, nrow(raw_pred)))
if (status != "COMPLETE") quit(status=2)
