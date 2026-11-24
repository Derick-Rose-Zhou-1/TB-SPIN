#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(DEGAS); library(Matrix)})

formal <- Sys.getenv("RESULT6_FORMAL_ROOT", unset="inputs/result6_formal_root")
v2 <- Sys.getenv("RESULT6_DEGAS_INPUT_ROOT", unset="inputs/result6_degas_input_root")
inp <- file.path(v2, '04_DEGAS', 'input')
recovery <- Sys.getenv('DEGAS_RECOVERY_OUTPUT', unset='')
if (!nzchar(recovery)) stop('DEGAS_RECOVERY_OUTPUT is required')
out <- file.path(recovery, '04_DEGAS')
if (dir.exists(out)) stop('refusing existing result directory: ', out)
dir.create(out, recursive=TRUE, showWarnings=FALSE)

status <- data.frame(
  endpoint=c('COX', 'H2', 'H3'),
  phenotype=c('E_COX', 'E_6M', 'E_12M'),
  state='PENDING',
  model_type=c('BlankCox', 'BlankClass', 'BlankClass'),
  native_binary_label=c(FALSE, TRUE, TRUE),
  detail='',
  stringsAsFactors=FALSE
)

write_status <- function() {
  write.table(status, file.path(out, 'ENDPOINT_STATUS.tsv'), sep='\t', row.names=FALSE, quote=FALSE)
}

set_status <- function(ep, state, detail='') {
  status[status$endpoint == ep, c('state', 'detail')] <<- list(state, detail)
  write_status()
}

stamp <- function(stage) {
  row <- data.frame(pid=Sys.getpid(), stage=stage, time=as.character(Sys.time()))
  write.table(row, file.path(out, 'CURRENT_STAGE.tsv'), sep='\t', row.names=FALSE, quote=FALSE)
  cat(format(Sys.time(), '%Y-%m-%d %H:%M:%S %z'), stage, '\n')
  flush.console()
}

run_endpoint <- function(ep, spec, clinical, bulk, common, scExp, cells) {
  set_status(ep, 'RUNNING')
  stamp(paste(ep, 'OFFICIAL', spec$model_type, 'TRAINING'))
  stopifnot(nrow(clinical) == spec$participant_n,
            !anyDuplicated(paste(clinical$cohort, clinical$participant_id)),
            !anyDuplicated(clinical$GSM),
            all(clinical$GSM %in% rownames(bulk)),
            all(clinical$y_true %in% c(0, 1)))
  b <- t(as.matrix(bulk[clinical$GSM, common, drop=FALSE]))
  patExp <- preprocessCounts(b)
  rownames(patExp) <- clinical$GSM
  colnames(patExp) <- common
  stopifnot(nrow(patExp) == nrow(clinical), ncol(patExp) == length(common),
            identical(rownames(patExp), clinical$GSM), all(is.finite(patExp)))

  if (ep == 'COX') {
    stopifnot(all(clinical$time_to_event_or_censor_days > 0))
    patLab <- cbind(time=clinical$time_to_event_or_censor_days, status=clinical$y_true)
  } else {
    patLab <- cbind(
      no_progression=as.integer(clinical$y_true == 0),
      progression=as.integer(clinical$y_true == 1)
    )
    stopifnot(all(rowSums(patLab) == 1L), all(colSums(patLab) >= 2L))
  }
  rownames(patLab) <- clinical$GSM

  initDEGAS()
  setPython(Sys.getenv("DEGAS_PYTHON", unset="python"))
  set_seed_term(20260919)
  stopifnot(checkForTF() == 0)
  tmp <- file.path(out, paste0('official_', tolower(ep), '_tmp'), '')
  fit <- runCCMTL(scExp, NULL, patExp, patLab, tmp, spec$model_type, 'DenseNet', 2)
  saveRDS(fit, file.path(out, paste0('DEGAS_', ep, '_OFFICIAL_FIT.rds')), compress=FALSE)

  stamp(paste(ep, 'OFFICIAL_CELL_PREDICTION'))
  if (ep == 'COX') {
    pred <- as.numeric(predClass(fit, scExp, 'pat'))
    stopifnot(length(pred) == length(cells), all(is.finite(pred)))
    output <- data.frame(
      cell_id=cells,
      degas_cox_association_score=pred,
      native_binary_label_available=FALSE
    )
  } else {
    prob <- as.matrix(predClass(fit, scExp, 'pat'))
    stopifnot(nrow(prob) == length(cells), ncol(prob) == 2L, all(is.finite(prob)))
    colnames(prob) <- c('no_progression', 'progression')
    onehot <- probtoOneHot(prob)
    stopifnot(all(rowSums(onehot) == 1L))
    native_label <- ifelse(onehot[, 'progression'] == 1L, 'DEGAS+', 'DEGAS-')
    output <- data.frame(
      cell_id=cells,
      probability_no_progression=prob[, 'no_progression'],
      probability_progression=prob[, 'progression'],
      native_label=native_label
    )
    counts <- as.data.frame(table(output$native_label))
    write.table(counts, file.path(out, paste0('DEGAS_', ep, '_LABEL_COUNTS.tsv')),
                sep='\t', row.names=FALSE, quote=FALSE)
  }
  stopifnot(nrow(output) == 214402L, !anyDuplicated(output$cell_id), identical(output$cell_id, cells))
  write.table(output, file.path(out, paste0('DEGAS_', ep, '_214402.tsv')),
              sep='\t', row.names=FALSE, quote=FALSE, na='NA')
  set_status(ep, 'COMPLETE_214402', if (ep == 'COX') 'continuous_only_no_native_binary_label' else 'native_BlankClass_argmax_label')
  rm(list=intersect(
    c('fit', 'pred', 'prob', 'onehot', 'native_label', 'output',
      'patExp', 'patLab', 'b', 'counts'),
    ls()
  ))
  gc()
}

main <- function() {
  write_status()
  version <- as.character(packageVersion('DEGAS'))
  stopifnot(identical(version, '1.0.0'))
  cells <- readLines(file.path(inp, 'DEGAS_CELLS.txt'))
  genes <- readLines(file.path(inp, 'DEGAS_HVG3000.txt'))
  stopifnot(length(cells) == 214402L, length(genes) == 3000L,
            !anyDuplicated(cells), !anyDuplicated(genes))
  stamp('READ_OUTCOME_FREE_HVG3000_FORMAL_COUNTS')
  x <- as(readMM(file.path(inp, 'DEGAS_SC_RAW_HVG3000.mtx')), 'dgCMatrix')
  dimnames(x) <- list(genes, cells)
  stopifnot(nrow(x) == 3000L, ncol(x) == 214402L, all(x@x >= 0), all(is.finite(x@x)))

  bulk <- read.delim(file.path(formal, '00_INPUT', 'BULK_LOG2CPM.tsv'), row.names=1, check.names=FALSE)
  endpoints <- read.delim(file.path(formal, '00_INPUT', 'BASELINE_ENDPOINTS.tsv'), check.names=FALSE)
  coxsource <- Sys.getenv("RESULT6_PARTICIPANT_ENDPOINTS", unset="inputs/PARTICIPANT_ENDPOINTS.tsv")
  cox <- read.delim(coxsource, check.names=FALSE)
  as_bool <- function(x) tolower(as.character(x)) %in% c('true', 't', '1')
  cox <- cox[cox$endpoint == 'E_COX' & as_bool(cox$baseline) & as_bool(cox$eligible), ]

  specs <- list(
    COX=list(endpoint='E_COX', model_type='BlankCox', participant_n=426L),
    H2=list(endpoint='E_6M', model_type='BlankClass', participant_n=412L),
    H3=list(endpoint='E_12M', model_type='BlankClass', participant_n=377L)
  )
  expected_participants <- c(COX=426L, H2=412L, H3=377L)
  stopifnot(identical(vapply(specs, function(x) x$participant_n, integer(1)), expected_participants))
  clinical_sets <- list(
    COX=cox,
    H2=endpoints[endpoints$endpoint == 'E_6M', ],
    H3=endpoints[endpoints$endpoint == 'E_12M', ]
  )
  for (ep in names(specs)) {
    clinical <- clinical_sets[[ep]]
    stopifnot(nrow(clinical) == specs[[ep]]$participant_n,
              !anyDuplicated(paste(clinical$cohort, clinical$participant_id)),
              !anyDuplicated(clinical$GSM), all(clinical$GSM %in% rownames(bulk)))
  }

  mapping <- do.call(rbind, lapply(names(specs), function(ep) {
    clinical <- clinical_sets[[ep]]
    data.frame(endpoint=ep, phenotype=specs[[ep]]$endpoint,
               cohort=clinical$cohort, participant_id=clinical$participant_id,
               GSM=clinical$GSM, y_true=clinical$y_true,
               time_to_event_or_censor_days=if (ep == 'COX') clinical$time_to_event_or_censor_days else NA_real_)
  }))
  write.table(mapping, file.path(out, 'PARTICIPANT_SAMPLE_MAPPING.tsv'), sep='\t', row.names=FALSE, quote=FALSE, na='NA')

  common <- genes[genes %in% colnames(bulk)]
  stopifnot(length(common) >= 2400L)
  x <- x[common, , drop=FALSE]
  stamp('OFFICIAL_DEGAS_PREPROCESS_COUNTS')
  scExp <- preprocessCounts(as.matrix(x))
  rm(x)
  gc()
  stopifnot(nrow(scExp) == length(cells), ncol(scExp) == length(common), all(is.finite(scExp)))
  rownames(scExp) <- cells
  colnames(scExp) <- common

  parameter_rows <- do.call(rbind, lapply(names(specs), function(ep) {
    data.frame(
      endpoint=ep,
      parameter=c('phenotype', 'model_type', 'architecture', 'FFdepth', 'training_steps',
                  'single_cell_batch_size', 'patient_batch_size', 'hidden_features',
                  'dropout_keep', 'lambda_L2', 'lambda_patient', 'lambda_MMD', 'seed',
                  'formal_cells', 'input_genes', 'feature_selection',
                  'outcome_used_for_feature_selection', 'binary_label_rule'),
      value=c(specs[[ep]]$endpoint, specs[[ep]]$model_type, 'DenseNet', '2', '2000',
              '200', '50', '50', '0.5', '3', '3', '3', '20260919',
              length(cells), length(common),
              'outcome-free Scanpy Seurat-style HVG3000 with participant batch',
              'FALSE', if (ep == 'COX') 'NONE_OFFICIAL_CONTINUOUS_ONLY' else 'DEGAS::probtoOneHot_argmax')
    )
  }))
  write.table(parameter_rows, file.path(out, 'DEGAS_PARAMETERS.tsv'), sep='\t', row.names=FALSE, quote=FALSE)

  provenance <- data.frame(
    endpoint=names(specs),
    phenotype=vapply(specs, function(x) x$endpoint, character(1)),
    model_type=vapply(specs, function(x) x$model_type, character(1)),
    official_package='DEGAS 1.0.0',
    tensorflow='2.3.1', protobuf='3.20.3',
    participant_n=expected_participants,
    formal_cell_n=214402L,
    phenotype_exact=TRUE, sample_ids_exact=TRUE, coverage_target_exact=TRUE,
    native_binary_label=c(FALSE, TRUE, TRUE),
    source_formal=formal,
    source_hvg=inp
  )
  write.table(provenance, file.path(out, 'PROVENANCE.tsv'), sep='\t', row.names=FALSE, quote=FALSE)

  for (ep in names(specs)) {
    tryCatch(
      run_endpoint(ep, specs[[ep]], clinical_sets[[ep]], bulk, common, scExp, cells),
      error=function(e) {
        message <- conditionMessage(e)
        writeLines(message, file.path(out, paste0('DEGAS_', ep, '_ERROR.txt')))
        set_status(ep, 'FAILED', message)
        stamp(paste(ep, 'FAILED', message))
      }
    )
  }
  if (any(status$state == 'FAILED')) stop('one or more DEGAS endpoints failed; inspect ENDPOINT_STATUS.tsv')
  stamp('DEGAS_H123_NATIVE_RECOVERY_COMPLETE')
}

tryCatch(
  main(),
  error=function(e) {
    writeLines(conditionMessage(e), file.path(out, 'FORMAL_ERROR.txt'))
    stamp(paste('FAILED', conditionMessage(e)))
    stop(e)
  }
)
