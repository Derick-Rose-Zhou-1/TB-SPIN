#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(scPAS); library(Seurat); library(Matrix)})

formal <- Sys.getenv("RESULT6_FORMAL_ROOT", unset="inputs/result6_formal_root")
aux <- Sys.getenv("RESULT6_DEGAS_INPUT_ROOT", unset="inputs/result6_degas_input_root")
inp <- file.path(formal, '00_INPUT')
recovery <- Sys.getenv('SCPAS_RECOVERY_OUTPUT', unset='')
if (!nzchar(recovery)) stop('SCPAS_RECOVERY_OUTPUT is required')
out <- file.path(recovery, '05_SCPAS')
if (dir.exists(out)) stop('refusing existing result directory: ', out)
dir.create(out, recursive=TRUE, showWarnings=FALSE)

stamp <- function(stage) {
  row <- data.frame(pid=Sys.getpid(), stage=stage, time=as.character(Sys.time()))
  write.table(row, file.path(out, 'CURRENT_STAGE.tsv'), sep='\t', row.names=FALSE, quote=FALSE)
  cat(format(Sys.time(), '%Y-%m-%d %H:%M:%S %z'), stage, '\n')
  flush.console()
}

main <- function() {
  version <- as.character(packageVersion('scPAS'))
  stopifnot(identical(version, '0.2.0'))
  cells <- readLines(file.path(inp, 'CELLS.txt'))
  genes <- readLines(file.path(inp, 'GENES.txt'))
  hvg <- readLines(file.path(aux, '04_DEGAS', 'input', 'DEGAS_HVG3000.txt'))
  stopifnot(length(cells) == 214402L, length(hvg) == 3000L,
            !anyDuplicated(cells), !anyDuplicated(hvg))
  stamp('READ_FULL_FORMAL_COUNTS')
  counts <- as(readMM(file.path(inp, 'FORMAL_RAW_COUNTS.mtx')), 'dgCMatrix')
  dimnames(counts) <- list(genes, cells)
  obj <- CreateSeuratObject(counts=counts, min.cells=0, min.features=0,
                            project='scPAS_Result3_native_recovery_20260919')
  rm(counts)
  gc()
  stopifnot(identical(colnames(obj), cells))
  obj <- NormalizeData(obj, normalization.method='LogNormalize', scale.factor=10000, verbose=FALSE)

  bulk0 <- as.matrix(read.delim(file.path(inp, 'BULK_LOG2CPM.tsv'), row.names=1, check.names=FALSE))
  endpoints <- read.delim(file.path(inp, 'BASELINE_ENDPOINTS.tsv'), check.names=FALSE)
  coxsource <- Sys.getenv("RESULT6_PARTICIPANT_ENDPOINTS", unset="inputs/PARTICIPANT_ENDPOINTS.tsv")
  cox <- read.delim(coxsource, check.names=FALSE)
  as_bool <- function(x) tolower(as.character(x)) %in% c('true', 't', '1')
  cox <- cox[cox$endpoint == 'E_COX' & as_bool(cox$baseline) & as_bool(cox$eligible), ]
  stopifnot(!anyDuplicated(paste(cox$cohort, cox$participant_id)), all(cox$GSM %in% rownames(bulk0)))

  specs <- list(
    COX=list(endpoint='E_COX', family='cox', participant_n=426L),
    H2=list(endpoint='E_6M', family='binomial', participant_n=412L),
    H3=list(endpoint='E_12M', family='binomial', participant_n=377L)
  )
  expected_participants <- c(COX=426L, H2=412L, H3=377L)
  stopifnot(identical(vapply(specs, function(x) x$participant_n, integer(1)), expected_participants))

  write.table(
    data.frame(
      parameter=c('package', 'version', 'formal_cells', 'features', 'feature_selection',
                  'feature_selection_outcome', 'imputation', 'network_class', 'independent',
                  'alpha', 'permutation_times', 'FDR', 'source_formal', 'source_aux'),
      value=c('scPAS', version, length(cells), length(hvg),
              'outcome-free participant-batched Seurat-style HVG3000', 'FALSE', 'FALSE',
              'SC', 'TRUE', 'official default vector', '2000', '0.05', formal, aux)
    ),
    file.path(out, 'SCPAS_PARAMETERS.tsv'), sep='\t', row.names=FALSE, quote=FALSE
  )

  for (ep in names(specs)) {
    spec <- specs[[ep]]
    clinical <- if (ep == 'COX') cox else endpoints[endpoints$endpoint == spec$endpoint, ]
    stopifnot(nrow(clinical) == spec$participant_n,
              all(clinical$GSM %in% rownames(bulk0)),
              !anyDuplicated(paste(clinical$cohort, clinical$participant_id)))
    bulk <- t(bulk0[clinical$GSM, , drop=FALSE])
    phenotype <- if (ep == 'COX') {
      cbind(time=clinical$time_to_event_or_censor_days, status=clinical$y_true)
    } else {
      clinical$y_true
    }
    stamp(paste(ep, 'OFFICIAL_SCPAS_FIT'))
    fit <- scPAS(
      bulk_dataset=bulk, sc_dataset=obj, phenotype=phenotype, assay='RNA',
      tag=if (ep == 'COX') NULL else c('no_progression', 'progression'),
      nfeature=hvg, imputation=FALSE, alpha=NULL, network_class='SC',
      independent=TRUE, family=spec$family, permutation_times=2000,
      FDR.threshold=0.05
    )
    md <- fit@meta.data
    stopifnot(nrow(md) == 214402L, identical(rownames(md), cells),
              all(is.finite(md$scPAS_RS)), all(is.finite(md$scPAS_NRS)),
              all(is.finite(md$scPAS_FDR)))
    tab <- data.frame(
      cell_id=cells, scPAS_RS=md$scPAS_RS, scPAS_NRS=md$scPAS_NRS,
      scPAS_Pvalue=md$scPAS_Pvalue, scPAS_FDR=md$scPAS_FDR, scPAS=md$scPAS
    )
    target <- file.path(out, paste0('SCPAS_', ep, '_214402.tsv'))
    write.table(tab, target, sep='\t', row.names=FALSE, quote=FALSE)
    saveRDS(fit@misc$scPAS_para, file.path(out, paste0('SCPAS_', ep, '_PARAMETERS.rds')), compress=FALSE)
    write.table(
      as.data.frame(table(tab$scPAS)),
      file.path(out, paste0('SCPAS_', ep, '_LABEL_COUNTS.tsv')),
      sep='\t', row.names=FALSE, quote=FALSE
    )
    rm(fit, md, tab, bulk, phenotype)
    gc()
    stamp(paste(ep, 'COMPLETE_214402'))
  }
  stamp('SCPAS_RESULT3_NATIVE_RECOVERY_COMPLETE')
}

tryCatch(
  main(),
  error=function(e) {
    writeLines(conditionMessage(e), file.path(out, 'FORMAL_ERROR.txt'))
    stamp(paste('FAILED', conditionMessage(e)))
    stop(e)
  }
)
