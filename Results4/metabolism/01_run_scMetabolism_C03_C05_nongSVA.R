#!/usr/bin/env Rscript

project_library <- Sys.getenv(
  "R4_SCMETABOLISM_LIB",
  unset = ".r_libs/scmetabolism_R45"
)
if (dir.exists(project_library)) .libPaths(c(project_library, .libPaths()))

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(Matrix)
  library(data.table)
  library(scMetabolism)
})

options(stringsAsFactors = FALSE)
set.seed(20260910)

argv <- commandArgs(trailingOnly = TRUE)
arg_value <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), argv, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^--", name, "="), "", hit[[1L]])
}

input_rds <- arg_value(
  "rds",
  "inputs/Tian_PBMC_16986genes_HVG3000_res2_QCclean_fullgenes.rds"
)
label_zip <- arg_value(
  "labels-zip",
  paste0(
    "results4/FIG4_PUBLICATION_REVISED/source_tables/",
    "harmonised_remote/TB_STATE_R4_SIDISH_STYLE_CELLLEVEL_DEG_GPT_REVIEW.zip"
  )
)
out_dir <- arg_value(
  "out",
  "results4/R4_SCPAS_SCMETABOLISM_C03_C05_V1"
)
ncores <- as.integer(arg_value(
  "ncores",
  as.character(max(1L, min(8L, parallel::detectCores(logical = TRUE) - 2L)))
))
imputation_request <- tolower(arg_value("imputation", "false"))
contrast_request <- toupper(arg_value("contrasts", "ALL"))
durable_checkpoints <- identical(
  tolower(arg_value("durable-checkpoints", "false")),
  "true"
)

if (!imputation_request %in% c("false", "true", "both")) {
  stop("--imputation must be false, true, or both")
}
imputation_modes <- switch(
  imputation_request,
  false = FALSE,
  true = TRUE,
  both = c(FALSE, TRUE)
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
table_dir <- file.path(out_dir, "tables")
checkpoint_dir <- if (durable_checkpoints) {
  file.path(table_dir, "checkpoints")
} else {
  tempfile(pattern = "r4_scmetabolism_ephemeral_checkpoints_")
}
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)
if (!durable_checkpoints) {
  on.exit(unlink(checkpoint_dir, recursive = TRUE, force = TRUE), add = TRUE)
}

required_packages <- c(
  "Seurat", "SeuratObject", "Matrix", "data.table", "scMetabolism",
  "VISION", "AUCell", "GSEABase", "GSVA", "BiocParallel"
)
missing_packages <- required_packages[!vapply(
  required_packages,
  requireNamespace,
  logical(1),
  quietly = TRUE
)]
if (length(missing_packages)) {
  stop(
    "Missing packages: ", paste(missing_packages, collapse = ", "),
    ". Run 00_install_dependencies.R first."
  )
}

if (!file.exists(input_rds)) stop("RDS not found: ", input_rds)
if (!file.exists(label_zip)) stop("Label ZIP not found: ", label_zip)

contrast_map <- data.table(
  contrast = c("C03", "C04", "C05"),
  comparison = c(
    "Asymptomatic TB vs M. tuberculosis infection",
    "Symptomatic TB vs M. tuberculosis infection",
    "Direct asymptomatic vs symptomatic"
  ),
  reference_state = c(
    "M. tuberculosis infection",
    "M. tuberculosis infection",
    "Asymptomatic TB"
  ),
  target_state = c("Asymptomatic TB", "Symptomatic TB", "Symptomatic TB"),
  plus_side = c("Asymptomatic TB", "Symptomatic TB", "Symptomatic TB"),
  minus_side = c(
    "M. tuberculosis infection",
    "M. tuberculosis infection",
    "Asymptomatic TB"
  ),
  label_member = paste0(
    "01_INPUT_AUDIT/", c("C03", "C04", "C05"),
    "_ALIGNED_LABELS.tsv.gz"
  )
)

selected_contrasts <- if (contrast_request == "ALL") {
  contrast_map$contrast
} else {
  trimws(strsplit(contrast_request, ",", fixed = TRUE)[[1L]])
}
unknown_contrasts <- setdiff(selected_contrasts, contrast_map$contrast)
if (length(unknown_contrasts)) {
  stop("Unknown --contrasts value(s): ", paste(unknown_contrasts, collapse = ", "))
}

read_zip_gzip_tsv <- function(zip_file, member) {
  tmp <- tempfile(pattern = "r4_scmetabolism_labels_")
  dir.create(tmp)
  on.exit(unlink(tmp, recursive = TRUE, force = TRUE), add = TRUE)
  utils::unzip(zip_file, files = member, exdir = tmp, overwrite = TRUE)
  extracted <- file.path(tmp, member)
  if (!file.exists(extracted)) stop("Could not extract ZIP member: ", member)
  fread(extracted)
}

message("Reading formal aligned scPAS labels...")
labels <- rbindlist(lapply(seq_len(nrow(contrast_map)), function(i) {
  z <- read_zip_gzip_tsv(label_zip, contrast_map$label_member[[i]])
  if (!identical(names(z), c("cell_id", "scPAS_label"))) {
    stop("Unexpected label columns for ", contrast_map$contrast[[i]])
  }
  if (anyDuplicated(z$cell_id)) stop("Duplicated cell IDs in ", contrast_map$contrast[[i]])
  z[, contrast := contrast_map$contrast[[i]]]
  z[, scPAS_status := fifelse(
    scPAS_label %chin% c("1", "scPAS+", "PAS+"),
    "scPAS+",
    fifelse(
      scPAS_label %chin% c("-1", "scPAS-", "PAS-"),
      "scPAS-",
      "neutral"
    )
  )]
  z[scPAS_status != "neutral", .(cell_id, contrast, scPAS_status)]
}))

labels <- merge(
  labels,
  contrast_map[, .(
    contrast, comparison, reference_state, target_state, plus_side, minus_side
  )],
  by = "contrast",
  all.x = TRUE,
  sort = FALSE
)
labels[, biological_side := fifelse(scPAS_status == "scPAS+", plus_side, minus_side)]

expected_counts <- data.table(
  contrast = c("C03", "C04", "C05"),
  expected_plus = c(5010L, 3470L, 437L),
  expected_minus = c(8L, 9287L, 2250L)
)
observed_counts <- dcast(
  labels[, .N, by = .(contrast, scPAS_status)],
  contrast ~ scPAS_status,
  value.var = "N",
  fill = 0L
)
setnames(observed_counts, c("scPAS+", "scPAS-"), c("observed_plus", "observed_minus"))
count_check <- merge(expected_counts, observed_counts, by = "contrast", all.x = TRUE)
if (any(count_check$expected_plus != count_check$observed_plus) ||
    any(count_check$expected_minus != count_check$observed_minus)) {
  stop("Formal scPAS label counts do not match the frozen audit")
}

# A Linux launcher can run C03/C04/C05 as independent CPU workers.  Filtering
# happens only after all frozen label counts have passed the shared audit.
contrast_map <- contrast_map[contrast %chin% selected_contrasts]
labels <- labels[contrast %chin% selected_contrasts]

message("Reading frozen Seurat object once...")
obj <- readRDS(input_rds)
if (!inherits(obj, "Seurat")) stop("Input is not a Seurat object")
if (length(dim(obj)) != 2L || any(as.numeric(dim(obj)) != c(16986, 356949))) {
  stop("Unexpected Seurat dimensions: ", paste(dim(obj), collapse = " x "))
}
if (anyDuplicated(colnames(obj))) stop("Duplicated Seurat cell IDs")
if (!all(labels$cell_id %chin% colnames(obj))) {
  stop("Some formal scPAS cell IDs are absent from the Seurat object")
}

meta_columns <- c("sample_id_std", "celltype_major", "celltype_fine")
if (!all(meta_columns %chin% colnames(obj[[]]))) {
  stop(
    "Required metadata columns missing: ",
    paste(setdiff(meta_columns, colnames(obj[[]])), collapse = ", ")
  )
}

rna <- obj[["RNA"]]
count_layers <- Layers(rna, search = "^counts($|\\.)")
if (length(count_layers) != 1L) {
  stop("Exactly one RNA counts layer is required; found: ", paste(count_layers, collapse = ", "))
}
counts_all <- LayerData(rna, layer = count_layers[[1L]])
if (!inherits(counts_all, "sparseMatrix")) {
  warning("RNA counts layer is not sparse; coercing to dgCMatrix")
  counts_all <- as(counts_all, "dgCMatrix")
}
if (anyDuplicated(rownames(counts_all))) stop("Duplicated gene symbols in counts matrix")
if (any(!is.finite(counts_all@x))) stop("Non-finite values in counts matrix")
if (any(counts_all@x < 0)) stop("Negative values in counts matrix")

meta <- as.data.table(obj[[]], keep.rownames = "cell_id")
meta <- meta[, .(cell_id, sample_id_std, celltype_major, celltype_fine)]
setkey(meta, cell_id)
obj_dims <- as.numeric(dim(obj))

# The frozen RDS also contains large normalized and scaled layers that are not
# inputs to scMetabolism. Keep only the sparse raw-count matrix and metadata so
# the four scoring backends have sufficient working memory.
rm(obj, rna)
invisible(gc(verbose = FALSE))

input_provenance <- data.table(
  item = c(
    "SEURAT_RDS", "FORMAL_SCPAS_LABEL_ZIP", "SEURAT_DIMENSIONS",
    "RNA_ASSAY_LAYER", "GENE_IDENTIFIER", "SCORE_INPUT", "SCPAS_LABEL_SCOPE",
    "SCMETABOLISM_REPOSITORY", "SCMETABOLISM_COMMIT", "RANDOM_SEED",
    "NCORES", "IMPUTATION_REQUEST", "CONTRASTS", "DURABLE_CHECKPOINTS"
  ),
  value = c(
    normalizePath(input_rds, winslash = "/"),
    normalizePath(label_zip, winslash = "/"),
    paste0(obj_dims[[1L]], " genes x ", obj_dims[[2L]], " cells"),
    paste0("RNA/", count_layers[[1L]]),
    "Human gene SYMBOL (Seurat rownames)",
    "Raw UMI counts, as required by scMetabolism",
    "Only non-neutral scPAS+ and scPAS- cells from formal C03/C04/C05",
    "https://github.com/wu-yc/scMetabolism",
    "59d6d2b35482013a0ad8bc4db9d588aa8bb37b75",
    "20260910",
    as.character(ncores),
    imputation_request,
    paste(selected_contrasts, collapse = ","),
    as.character(durable_checkpoints)
  )
)
fwrite(input_provenance, file.path(table_dir, "00_INPUT_PROVENANCE.tsv"), sep = "\t")

label_counts <- labels[, .(
  cells = .N,
  participants = uniqueN(meta[cell_id, on = "cell_id"]$sample_id_std)
), by = .(
  contrast, comparison, reference_state, target_state, scPAS_status, biological_side
)]
fwrite(label_counts, file.path(table_dir, "01_SCPAS_LABEL_COUNTS.tsv"), sep = "\t")

labels_with_meta <- meta[labels, on = "cell_id"]
composition_tables <- list()
for (level in c("ALL", "MAJOR", "FINE")) {
  z <- copy(labels_with_meta)
  if (level == "ALL") z[, subpopulation := "ALL_CELLS"]
  if (level == "MAJOR") z[, subpopulation := as.character(celltype_major)]
  if (level == "FINE") z[, subpopulation := as.character(celltype_fine)]
  q <- z[, .(
    n_cells = .N,
    n_participants = uniqueN(sample_id_std)
  ), by = .(
    contrast, comparison, scPAS_status, biological_side, subpopulation
  )]
  q[, annotation_level := level]
  q[, fraction_within_status := n_cells / sum(n_cells), by = .(contrast, scPAS_status)]
  composition_tables[[level]] <- q
}
composition <- rbindlist(composition_tables, use.names = TRUE)
setcolorder(
  composition,
  c(
    "contrast", "comparison", "annotation_level", "subpopulation",
    "scPAS_status", "biological_side", "n_cells", "n_participants",
    "fraction_within_status"
  )
)
fwrite(
  composition,
  file.path(table_dir, "02_SCPAS_CELL_SUBPOPULATION_COUNTS.tsv"),
  sep = "\t"
)

get_gmt_file <- function(database) {
  filename <- switch(
    database,
    KEGG = "KEGG_metabolism_nc.gmt",
    REACTOME = "REACTOME_metabolism.gmt"
  )
  path <- system.file("data", filename, package = "scMetabolism")
  if (!nzchar(path) || !file.exists(path)) stop("Missing scMetabolism GMT: ", filename)
  path
}

get_gene_sets <- function(database) {
  gmt <- GSEABase::getGmt(get_gmt_file(database))
  ans <- lapply(gmt, GSEABase::geneIds)
  names(ans) <- vapply(gmt, GSEABase::setName, character(1))
  ans
}

score_with_current_gsva <- function(expr, database, method, ncores) {
  gene_sets <- get_gene_sets(database)
  bpparam <- if (.Platform$OS.type == "windows") {
    # GSVA 2.4.x does not serialize its parameter object correctly through
    # SnowParam on Windows (the remote worker receives a malformed gene-set
    # object).  SerialParam is deterministic and avoids that upstream issue.
    BiocParallel::SerialParam(progressbar = FALSE)
  } else {
    BiocParallel::MulticoreParam(workers = ncores, progressbar = FALSE)
  }
  ns <- asNamespace("GSVA")
  if (method == "ssGSEA") {
    if (exists("ssgseaParam", envir = ns, inherits = FALSE)) {
      param <- GSVA::ssgseaParam(exprData = expr, geneSets = gene_sets, normalize = TRUE)
      return(GSVA::gsva(param, BPPARAM = bpparam, verbose = FALSE))
    }
    return(GSVA::gsva(
      expr, gene_sets, method = "ssgsea", kcdf = "Poisson",
      parallel.sz = ncores, verbose = FALSE
    ))
  }
  if (exists("gsvaParam", envir = ns, inherits = FALSE)) {
    param <- GSVA::gsvaParam(exprData = expr, geneSets = gene_sets, kcdf = "Poisson")
    return(GSVA::gsva(param, BPPARAM = bpparam, verbose = FALSE))
  }
  GSVA::gsva(
    expr, gene_sets, method = "gsva", kcdf = "Poisson",
    parallel.sz = ncores, verbose = FALSE
  )
}

score_metabolism <- function(expr, database, method, ncores) {
  if (method == "VISION") {
    # Matrix 1.7 emits a deprecation warning inside VISION 3.0.2.  It does not
    # affect the score calculation, but VISION may promote it to an error.
    ans <- suppressWarnings(scMetabolism::sc.metabolism(
      countexp = expr,
      method = method,
      imputation = FALSE,
      # VISION uses fork-based mc.cores internally; Windows requires one.
      ncores = if (.Platform$OS.type == "windows") 1L else ncores,
      metabolism.type = database
    ))
  } else if (method == "AUCell") {
    # This is the AUCell branch of scMetabolism::sc.metabolism verbatim in
    # algorithmic terms, without data.frame(), which otherwise sanitizes cell
    # IDs such as "cohort:barcode" and breaks the frozen-ID audit.
    rankings <- suppressWarnings(AUCell::AUCell_buildRankings(
      as.matrix(expr), nCores = ncores, plotStats = FALSE, verbose = FALSE
    ))
    gene_sets <- GSEABase::getGmt(get_gmt_file(database))
    auc <- suppressWarnings(AUCell::AUCell_calcAUC(gene_sets, rankings))
    ans <- AUCell::getAUC(auc)
  } else {
    # scMetabolism 0.2.2 calls the retired GSVA matrix API. This compatibility
    # branch uses the same bundled scMetabolism GMT and the same GSVA/ssGSEA
    # algorithm through the installed GSVA API.
    ans <- score_with_current_gsva(expr, database, method, ncores)
  }
  ans <- as.matrix(ans)
  # Older scMetabolism uses data.frame() on score matrices.  Restore original
  # frozen cell IDs only when the dimensions and positional sanitized names
  # prove that no reordering occurred.
  if (ncol(ans) == ncol(expr) &&
      identical(colnames(ans), make.names(colnames(expr), unique = TRUE))) {
    colnames(ans) <- colnames(expr)
  }
  if (all(colnames(expr) %chin% colnames(ans))) {
    ans <- ans[, colnames(expr), drop = FALSE]
  } else if (all(colnames(expr) %chin% rownames(ans))) {
    ans <- t(ans[colnames(expr), , drop = FALSE])
  } else {
    stop("Metabolism score matrix cell IDs do not match the expression matrix")
  }
  storage.mode(ans) <- "double"
  # Some pathways can be non-estimable for a particular algorithm when too
  # few of their genes are present or all contributing values are constant.
  # Preserve those pathways as NA and mark them non-estimable downstream;
  # do not discard an otherwise valid scoring configuration.
  ans[!is.finite(ans)] <- NA_real_
  ans
}

prepare_dense_expression <- function(counts_sparse, imputation) {
  expr <- as.matrix(counts_sparse)
  storage.mode(expr) <- "double"
  if (!imputation) return(expr)
  message("Running the scMetabolism ALRA imputation sensitivity analysis...")
  completed <- scMetabolism::alra(expr)
  imputed <- completed[[3L]]
  rownames(imputed) <- rownames(expr)
  colnames(imputed) <- colnames(expr)
  rm(completed, expr)
  gc(verbose = FALSE)
  imputed
}

safe_stats <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(c(mean = NA, sd = NA, median = NA, q25 = NA, q75 = NA))
  c(
    mean = mean(x),
    sd = if (length(x) > 1L) stats::sd(x) else NA_real_,
    median = stats::median(x),
    q25 = unname(stats::quantile(x, 0.25, names = FALSE)),
    q75 = unname(stats::quantile(x, 0.75, names = FALSE))
  )
}

analyse_score_matrix <- function(scores, cell_info, config) {
  stopifnot(identical(colnames(scores), cell_info$cell_id))
  pathways <- rownames(scores)
  summaries <- list()
  comparisons <- list()
  k <- 0L

  for (annotation_level in c("ALL", "MAJOR", "FINE")) {
    strata <- switch(
      annotation_level,
      ALL = rep("ALL_CELLS", nrow(cell_info)),
      MAJOR = as.character(cell_info$celltype_major),
      FINE = as.character(cell_info$celltype_fine)
    )
    for (subpopulation in sort(unique(strata))) {
      idx <- which(strata == subpopulation)
      if (!length(idx)) next
      info_sub <- cell_info[idx]
      score_sub <- scores[, idx, drop = FALSE]

      for (status in c("scPAS+", "scPAS-")) {
        j <- which(info_sub$scPAS_status == status)
        if (!length(j)) next
        st <- t(apply(score_sub[, j, drop = FALSE], 1L, safe_stats))
        k <- k + 1L
        summaries[[k]] <- data.table(
          contrast = config$contrast,
          comparison = config$comparison,
          database = config$database,
          method = config$method,
          imputation = config$imputation,
          annotation_level = annotation_level,
          subpopulation = subpopulation,
          scPAS_status = status,
          biological_side = unique(info_sub$biological_side[j])[[1L]],
          pathway = pathways,
          n_cells = length(j),
          n_participants = uniqueN(info_sub$sample_id_std[j]),
          score_mean = st[, "mean"],
          score_sd = st[, "sd"],
          score_median = st[, "median"],
          score_q25 = st[, "q25"],
          score_q75 = st[, "q75"]
        )
      }

      # Formal inference is donor-aware: average cell scores within each
      # sample/status and perform a paired sample-level Wilcoxon test.
      group_dt <- info_sub[, .(
        cell_index = .I
      ), by = .(sample_id_std, scPAS_status)]
      group_sizes <- group_dt[, .N, by = .(sample_id_std, scPAS_status)]
      group_names <- paste(group_sizes$sample_id_std, group_sizes$scPAS_status, sep = "||")
      aggregate_matrix <- sparseMatrix(
        i = unlist(group_dt$cell_index),
        j = rep(seq_len(nrow(group_sizes)), group_sizes$N),
        x = rep(1 / group_sizes$N, group_sizes$N),
        dims = c(ncol(score_sub), nrow(group_sizes)),
        dimnames = list(colnames(score_sub), group_names)
      )
      sample_means <- as.matrix(score_sub %*% aggregate_matrix)
      plus_names <- paste(group_sizes$sample_id_std[group_sizes$scPAS_status == "scPAS+"], "scPAS+", sep = "||")
      minus_names <- paste(group_sizes$sample_id_std[group_sizes$scPAS_status == "scPAS-"], "scPAS-", sep = "||")
      paired_samples <- intersect(
        sub("\\|\\|scPAS\\+$", "", plus_names),
        sub("\\|\\|scPAS-$", "", minus_names)
      )

      cmp <- data.table(
        contrast = config$contrast,
        comparison = config$comparison,
        database = config$database,
        method = config$method,
        imputation = config$imputation,
        annotation_level = annotation_level,
        subpopulation = subpopulation,
        pathway = pathways,
        plus_side = config$plus_side,
        minus_side = config$minus_side,
        plus_cells = sum(info_sub$scPAS_status == "scPAS+"),
        minus_cells = sum(info_sub$scPAS_status == "scPAS-"),
        paired_participants = length(paired_samples),
        finite_paired_participants = NA_integer_,
        paired_mean_difference = NA_real_,
        paired_median_difference = NA_real_,
        wilcoxon_statistic = NA_real_,
        P = NA_real_,
        estimability = "STRUCTURAL_NE_LT5_PAIRED_PARTICIPANTS"
      )

      if (length(paired_samples) >= 5L) {
        plus_cols <- paste(paired_samples, "scPAS+", sep = "||")
        minus_cols <- paste(paired_samples, "scPAS-", sep = "||")
        differences <- sample_means[, plus_cols, drop = FALSE] -
          sample_means[, minus_cols, drop = FALSE]
        mean_difference <- rowMeans(differences, na.rm = TRUE)
        median_difference <- apply(
          differences, 1L, stats::median, na.rm = TRUE
        )
        mean_difference[!is.finite(mean_difference)] <- NA_real_
        median_difference[!is.finite(median_difference)] <- NA_real_
        cmp[, paired_mean_difference := mean_difference]
        cmp[, paired_median_difference := median_difference]
        tests <- lapply(seq_len(nrow(differences)), function(ii) {
          d <- differences[ii, ]
          d <- d[is.finite(d)]
          if (length(d) < 5L) {
            return(c(n = length(d), statistic = NA_real_, p = NA_real_))
          }
          if (all(d == 0)) return(c(n = length(d), statistic = 0, p = 1))
          fit <- suppressWarnings(stats::wilcox.test(
            d,
            mu = 0,
            alternative = "two.sided",
            paired = FALSE,
            exact = FALSE,
            correct = TRUE
          ))
          c(n = length(d), statistic = unname(fit$statistic), p = fit$p.value)
        })
        tests <- do.call(rbind, tests)
        cmp[, finite_paired_participants := as.integer(tests[, "n"])]
        cmp[, wilcoxon_statistic := tests[, "statistic"]]
        cmp[, P := tests[, "p"]]
        cmp[, estimability := fifelse(
          finite_paired_participants >= 5L,
          "ESTIMABLE_PAIRED_PARTICIPANT_WILCOXON",
          "STRUCTURAL_NE_LT5_FINITE_PAIRED_PARTICIPANTS"
        )]
      }
      cmp[, FDR := stats::p.adjust(P, method = "BH")]
      cmp[, higher_in := fifelse(
        is.na(paired_median_difference),
        NA_character_,
        fifelse(paired_median_difference > 0, plus_side,
          fifelse(paired_median_difference < 0, minus_side, "NO_MEDIAN_DIFFERENCE")
        )
      )]
      comparisons[[length(comparisons) + 1L]] <- cmp
    }
  }
  list(
    summary = rbindlist(summaries, use.names = TRUE, fill = TRUE),
    comparison = rbindlist(comparisons, use.names = TRUE, fill = TRUE)
  )
}

methods <- c("VISION", "AUCell", "ssGSEA")
databases <- c("KEGG", "REACTOME")
run_manifest <- list()

for (contrast_id in selected_contrasts) {
  contrast_info <- contrast_map[contrast == contrast_id]
  contrast_labels <- labels[contrast == contrast_id]
  contrast_cells <- contrast_labels$cell_id
  cell_info <- meta[contrast_cells, on = "cell_id"]
  cell_info <- merge(
    cell_info,
    contrast_labels[, .(cell_id, scPAS_status, biological_side)],
    by = "cell_id",
    all.x = TRUE,
    sort = FALSE
  )
  cell_info[, input_order__ := match(cell_id, contrast_cells)]
  setorder(cell_info, input_order__)
  cell_info[, input_order__ := NULL]
  if (!identical(cell_info$cell_id, contrast_cells)) stop("Cell order alignment failure")
  counts_sparse <- counts_all[, contrast_cells, drop = FALSE]
  if (any(Matrix::colSums(counts_sparse) <= 0)) stop("Zero-library cell in ", contrast_id)

  for (imputation in imputation_modes) {
    message(
      "Preparing ", contrast_id, " expression; imputation=", imputation,
      "; cells=", ncol(counts_sparse)
    )
    expr <- prepare_dense_expression(counts_sparse, imputation)

    for (database in databases) {
      for (method in methods) {
        key <- paste(
          contrast_id,
          database,
          method,
          if (imputation) "ALRA" else "NOIMPUTE",
          sep = "__"
        )
        summary_file <- file.path(checkpoint_dir, paste0(key, "__SUMMARY.tsv.gz"))
        comparison_file <- file.path(checkpoint_dir, paste0(key, "__COMPARISON.tsv.gz"))
        manifest_file <- file.path(checkpoint_dir, paste0(key, "__STATUS.tsv"))
        started <- Sys.time()

        if (file.exists(summary_file) && file.exists(comparison_file)) {
          message("Checkpoint exists, skipping: ", key)
          run_manifest[[key]] <- data.table(
            contrast = contrast_id,
            database = database,
            method = method,
            imputation = imputation,
            status = "CHECKPOINT_REUSED",
            started = as.character(started),
            finished = as.character(Sys.time()),
            elapsed_seconds = 0,
            error = NA_character_
          )
          next
        }

        message("Scoring ", key)
        result <- tryCatch({
          scores <- score_metabolism(expr, database, method, ncores)
          analysed <- analyse_score_matrix(
            scores,
            cell_info,
            list(
              contrast = contrast_id,
              comparison = contrast_info$comparison,
              plus_side = contrast_info$plus_side,
              minus_side = contrast_info$minus_side,
              database = database,
              method = method,
              imputation = imputation
            )
          )
          fwrite(analysed$summary, summary_file, sep = "\t", compress = "gzip")
          fwrite(analysed$comparison, comparison_file, sep = "\t", compress = "gzip")
          rm(scores, analysed)
          gc(verbose = FALSE)
          list(status = "SUCCESS", error = NA_character_)
        }, error = function(e) {
          list(status = "FAILED", error = conditionMessage(e))
        })

        finished <- Sys.time()
        status_row <- data.table(
          contrast = contrast_id,
          database = database,
          method = method,
          imputation = imputation,
          status = result$status,
          started = as.character(started),
          finished = as.character(finished),
          elapsed_seconds = as.numeric(difftime(finished, started, units = "secs")),
          error = result$error
        )
        fwrite(status_row, manifest_file, sep = "\t")
        run_manifest[[key]] <- status_row
      }
    }
    rm(expr)
    gc(verbose = FALSE)
  }
  rm(counts_sparse)
  gc(verbose = FALSE)
}

manifest <- rbindlist(run_manifest, use.names = TRUE, fill = TRUE)
fwrite(manifest, file.path(table_dir, "03_SCORING_RUN_MANIFEST.tsv"), sep = "\t")
if (any(manifest$status == "FAILED")) {
  warning("Some scoring configurations failed; see 03_SCORING_RUN_MANIFEST.tsv")
}

summary_files <- list.files(
  checkpoint_dir,
  pattern = "__SUMMARY\\.tsv\\.gz$",
  full.names = TRUE
)
comparison_files <- list.files(
  checkpoint_dir,
  pattern = "__COMPARISON\\.tsv\\.gz$",
  full.names = TRUE
)
if (!length(summary_files) || !length(comparison_files)) {
  stop("No successful metabolism results were generated")
}
score_summary <- rbindlist(lapply(summary_files, fread), use.names = TRUE, fill = TRUE)
comparison_results <- rbindlist(
  lapply(comparison_files, fread),
  use.names = TRUE,
  fill = TRUE
)
fwrite(
  score_summary,
  file.path(table_dir, "04_METABOLISM_SCORE_SUMMARY.tsv.gz"),
  sep = "\t",
  compress = "gzip"
)
fwrite(
  comparison_results,
  file.path(table_dir, "05_SCPAS_METABOLISM_ASSOCIATIONS.tsv.gz"),
  sep = "\t",
  compress = "gzip"
)

all_effects <- comparison_results[
  annotation_level == "ALL" & estimability == "ESTIMABLE_PAIRED_PARTICIPANT_WILCOXON",
  .(
    contrast, database, imputation, method, pathway,
    paired_median_difference
  )
]
method_concordance <- list()
if (nrow(all_effects)) {
  for (contrast_id in unique(all_effects$contrast)) {
    for (database_id in unique(all_effects$database)) {
      for (imputation_id in unique(all_effects$imputation)) {
        z <- all_effects[
          contrast == contrast_id & database == database_id & imputation == imputation_id
        ]
        for (i in seq_len(length(methods) - 1L)) {
          for (j in seq.int(i + 1L, length(methods))) {
            a <- z[method == methods[[i]], .(pathway, effect_a = paired_median_difference)]
            b <- z[method == methods[[j]], .(pathway, effect_b = paired_median_difference)]
            q <- merge(a, b, by = "pathway")
            method_concordance[[length(method_concordance) + 1L]] <- data.table(
              contrast = contrast_id,
              database = database_id,
              imputation = imputation_id,
              method_a = methods[[i]],
              method_b = methods[[j]],
              pathways_compared = nrow(q),
              spearman_rho = if (nrow(q) >= 3L) {
                suppressWarnings(stats::cor(q$effect_a, q$effect_b, method = "spearman"))
              } else {
                NA_real_
              }
            )
          }
        }
      }
    }
  }
}
fwrite(
  rbindlist(method_concordance, use.names = TRUE, fill = TRUE),
  file.path(table_dir, "06_METHOD_CONCORDANCE.tsv"),
  sep = "\t"
)

imputation_sensitivity <- data.table()
if (length(unique(comparison_results$imputation)) == 2L) {
  a <- comparison_results[
    annotation_level == "ALL" & imputation == FALSE,
    .(
      contrast, database, method, pathway,
      noimpute_effect = paired_median_difference,
      noimpute_fdr = FDR
    )
  ]
  b <- comparison_results[
    annotation_level == "ALL" & imputation == TRUE,
    .(
      contrast, database, method, pathway,
      alra_effect = paired_median_difference,
      alra_fdr = FDR
    )
  ]
  q <- merge(a, b, by = c("contrast", "database", "method", "pathway"))
  imputation_sensitivity <- q[, .(
    pathways_compared = sum(is.finite(noimpute_effect) & is.finite(alra_effect)),
    spearman_rho = suppressWarnings(stats::cor(
      noimpute_effect,
      alra_effect,
      method = "spearman",
      use = "pairwise.complete.obs"
    )),
    same_effect_direction_fraction = mean(
      sign(noimpute_effect) == sign(alra_effect),
      na.rm = TRUE
    ),
    noimpute_fdr005_n = sum(noimpute_fdr < 0.05, na.rm = TRUE),
    alra_fdr005_n = sum(alra_fdr < 0.05, na.rm = TRUE)
  ), by = .(contrast, database, method)]
}
fwrite(
  imputation_sensitivity,
  file.path(table_dir, "07_IMPUTATION_SENSITIVITY.tsv"),
  sep = "\t"
)

package_versions <- data.table(
  package = required_packages,
  version = vapply(required_packages, function(x) {
    as.character(utils::packageVersion(x))
  }, character(1))
)
package_versions <- rbind(
  package_versions,
  data.table(package = "R", version = R.version.string)
)
fwrite(package_versions, file.path(table_dir, "08_PACKAGE_VERSIONS.tsv"), sep = "\t")

significant_summary <- comparison_results[, .(
  tested_pathways = sum(is.finite(P)),
  FDR_lt_0_05 = sum(FDR < 0.05, na.rm = TRUE),
  higher_in_plus_FDR_lt_0_05 = sum(
    FDR < 0.05 & paired_median_difference > 0,
    na.rm = TRUE
  ),
  higher_in_minus_FDR_lt_0_05 = sum(
    FDR < 0.05 & paired_median_difference < 0,
    na.rm = TRUE
  ),
  estimability = if (any(estimability == "ESTIMABLE_PAIRED_PARTICIPANT_WILCOXON")) {
    "ESTIMABLE"
  } else {
    "STRUCTURAL_NE"
  }
), by = .(
  contrast, comparison, database, method, imputation,
  annotation_level, subpopulation, plus_side, minus_side
)]
fwrite(
  significant_summary,
  file.path(table_dir, "09_SIGNIFICANT_PATHWAY_COUNTS.tsv"),
  sep = "\t"
)

message("Completed. No figures were generated.")
