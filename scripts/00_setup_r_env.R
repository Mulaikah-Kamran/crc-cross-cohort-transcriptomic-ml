#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# 00_setup_r_env.R
# Purpose: install every R/Bioconductor package needed for Phase 1 data
#          acquisition. Run this ONCE, locally, before the extraction scripts.
# ---------------------------------------------------------------------------

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

pkgs <- c(
  "SummarizedExperiment",  # container FieldEffectCrc returns data in
  "FieldEffectCrc",        # Dampier et al. pooled CRC cohorts (A/B/C)
  "TCGAbiolinks",          # GDC/TCGA-COAD query + download
  "GEOquery",              # GSE196006 / GSE251845 retrieval
  "biomaRt",               # Ensembl ID <-> symbol / version cross-checks
  "arrow"                  # write parquet so Python can read directly
)

BiocManager::install(pkgs, update = FALSE, ask = FALSE)

# Sanity check
invisible(lapply(pkgs, function(p) {
  ok <- requireNamespace(p, quietly = TRUE)
  cat(sprintf("%-25s %s\n", p, ifelse(ok, "OK", "FAILED - install manually")))
}))
