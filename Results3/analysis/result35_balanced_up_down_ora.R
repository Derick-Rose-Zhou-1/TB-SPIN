suppressPackageStartupMessages({
  library(clusterProfiler)
  library(data.table)
  library(jsonlite)
})

options(stringsAsFactors = FALSE)
set.seed(20260831)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("Usage: balanced_up_down_ora.R <V5_ROOT> <OUT_ROOT> <DEST_ROOT>")
v5_root <- normalizePath(args[[1]], mustWork = TRUE)
out_root <- normalizePath(args[[2]], mustWork = TRUE)
dest_root <- args[[3]]
dir.create(dest_root, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(dest_root, "02_GO_BP_ORA"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(dest_root, "03_KEGG_ORA"), recursive = TRUE, showWarnings = FALSE)

axes <- c("H180", "H365", "COX")
directions <- c("UP", "DOWN")
db_files <- c(
  GO_BP = "c5.go.bp.v2025.1.Hs.symbols.gmt",
  KEGG = "c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt"
)

read_gmt <- function(path) {
  lines <- readLines(path, warn = FALSE)
  pieces <- strsplit(lines, "\t", fixed = TRUE)
  term2gene <- rbindlist(lapply(pieces, function(x) {
    if (length(x) < 3) return(NULL)
    data.table(term = x[[1]], gene = x[3:length(x)])
  }))
  term2gene <- unique(term2gene[nzchar(gene)])
  term2name <- unique(term2gene[, .(term, name = term)])
  list(term2gene = as.data.frame(term2gene), term2name = as.data.frame(term2name))
}

empty_standard <- function() {
  data.frame(
    axis = character(),
    direction = character(),
    database = character(),
    term_id = character(),
    term = character(),
    gene_set_size = integer(),
    query_gene_n = integer(),
    universe_gene_n = integer(),
    overlap_n = integer(),
    overlap_genes = character(),
    GeneRatio = character(),
    BgRatio = character(),
    odds_ratio_if_available = numeric(),
    raw_pvalue = numeric(),
    FDR = numeric(),
    minus_log10_FDR = numeric(),
    MSIGDB_release = character(),
    GMT_sha256 = character(),
    stringsAsFactors = FALSE
  )
}

safe_write <- function(df, path) {
  if (is.null(df) || nrow(df) == 0) df <- empty_standard()
  fwrite(df, path, sep = "\t", na = "NA", quote = FALSE)
}

standardize <- function(z, axis, direction, database, query_n, universe_n, gmt_sha) {
  if (is.null(z) || nrow(z) == 0) return(empty_standard())
  overlap_genes <- vapply(strsplit(as.character(z$geneID), "/", fixed = TRUE), function(x) {
    paste(sort(unique(x[nzchar(x)])), collapse = ";")
  }, character(1))
  gene_set_size <- as.integer(sub("/.*$", "", as.character(z$BgRatio)))
  overlap_n <- as.integer(z$Count)
  odds_ratio <- mapply(function(a, gs) {
    b <- query_n - a
    c <- gs - a
    d <- universe_n - a - b - c
    if (b < 0 || c < 0 || d < 0) return(NA_real_)
    if (b == 0 || c == 0) return(Inf)
    (a * d) / (b * c)
  }, overlap_n, gene_set_size)
  fdr <- as.numeric(z$p.adjust)
  out <- data.frame(
    axis = axis,
    direction = direction,
    database = database,
    term_id = as.character(z$ID),
    term = as.character(z$Description),
    gene_set_size = gene_set_size,
    query_gene_n = as.integer(query_n),
    universe_gene_n = as.integer(universe_n),
    overlap_n = overlap_n,
    overlap_genes = overlap_genes,
    GeneRatio = as.character(z$GeneRatio),
    BgRatio = as.character(z$BgRatio),
    odds_ratio_if_available = as.numeric(odds_ratio),
    raw_pvalue = as.numeric(z$pvalue),
    FDR = fdr,
    minus_log10_FDR = -log10(pmax(fdr, 1e-300)),
    MSIGDB_release = "2025.1.Hs",
    GMT_sha256 = gmt_sha,
    stringsAsFactors = FALSE
  )
  out[order(out$FDR, out$raw_pvalue, out$term_id, method = "radix"), , drop = FALSE]
}

resource_lock <- fread(file.path(out_root, "00_PROVENANCE", "MSIGDB_RESOURCE_LOCK.tsv"))
gmts <- list()
for (db_name in names(db_files)) {
  gmts[[db_name]] <- read_gmt(file.path(v5_root, "08_MSIGDB", db_files[[db_name]]))
}

for (axis in axes) {
  universe <- unique(as.character(fread(file.path(out_root, "01_GENESETS", paste0(axis, "_UNIVERSE.tsv")))$gene))
  universe <- universe[nzchar(universe)]
  for (direction in directions) {
    query_path <- file.path(out_root, "01_GENESETS", paste0(axis, "_", direction, "_FDR05_LFC10.tsv"))
    query <- unique(as.character(fread(query_path)$gene))
    query <- intersect(query[nzchar(query)], universe)
    for (db_name in names(db_files)) {
      gmt <- gmts[[db_name]]
      result <- enricher(
        gene = query,
        universe = universe,
        TERM2GENE = gmt$term2gene,
        TERM2NAME = gmt$term2name,
        pvalueCutoff = 1,
        qvalueCutoff = 1,
        pAdjustMethod = "BH",
        minGSSize = 10,
        maxGSSize = 500
      )
      z <- if (is.null(result)) data.frame() else as.data.frame(result)
      gmt_sha <- as.character(resource_lock[database == db_name, sha256][[1]])
      out <- standardize(z, axis, direction, db_name, length(query), length(universe), gmt_sha)
      folder <- if (db_name == "GO_BP") "02_GO_BP_ORA" else "03_KEGG_ORA"
      prefix <- paste(axis, direction, db_name, sep = "_")
      safe_write(out, file.path(dest_root, folder, paste0(prefix, "_ALL.tsv")))
      safe_write(out[out$FDR < 0.05, , drop = FALSE], file.path(dest_root, folder, paste0(prefix, "_FDR05.tsv")))
      message("COMPLETE ", axis, " ", direction, " ", db_name, " all=", nrow(out), " fdr05=", sum(out$FDR < 0.05))
    }
  }
}

if (normalizePath(dest_root, mustWork = TRUE) == normalizePath(out_root, mustWork = TRUE)) {
  env <- data.frame(
    R_version = R.version.string,
    clusterProfiler_version = as.character(packageVersion("clusterProfiler")),
    data_table_version = as.character(packageVersion("data.table")),
    jsonlite_version = as.character(packageVersion("jsonlite")),
    Rscript = normalizePath(file.path(R.home("bin"), "Rscript"), mustWork = TRUE),
    stringsAsFactors = FALSE
  )
  fwrite(env, file.path(out_root, "00_PROVENANCE", "R_ENVIRONMENT.tsv"), sep = "\t", quote = FALSE)
}

cat("CLUSTERPROFILER_ORA_COMPLETE\n")
