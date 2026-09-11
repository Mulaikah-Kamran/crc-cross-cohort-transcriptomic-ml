#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# 01_extract_fieldeffectcrc.R
#
# Purpose: Load FieldEffectCrc's three cohorts (each a SummarizedExperiment),
#          and write out, per cohort:
#            - counts matrix   (genes x samples, raw counts)   -> parquet
#            - colData / metadata (samples x variables)        -> csv
#            - rowData / gene annotation (genes x annotation)  -> csv
#
# This gives Python everything it needs without touching R again.
#
# NOTE: Run this locally (or on any machine with internet access to
# Bioconductor) — it cannot run inside Claude's sandboxed environment,
# which does not have network access to Bioconductor's servers.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(FieldEffectCrc)
  library(SummarizedExperiment)
  library(arrow)
})

out_dir <- "data/raw/fieldeffectcrc"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

extract_cohort <- function(se, cohort_name) {
  message(sprintf("Extracting cohort %s: %d genes x %d samples",
                   cohort_name, nrow(se), ncol(se)))

  counts <- as.data.frame(assay(se, "counts"))

  counts$gene_id <- rownames(counts)
  col_meta <- as.data.frame(colData(se))
  col_meta$sample_id <- rownames(col_meta)
  row_meta <- as.data.frame(rowData(se))
  row_meta$gene_id <- rownames(row_meta)

  write_parquet(counts, file.path(out_dir, sprintf("cohort%s_counts.parquet", cohort_name)))
  write.csv(col_meta, file.path(out_dir, sprintf("cohort%s_colData.csv", cohort_name)), row.names = FALSE)
  write.csv(row_meta, file.path(out_dir, sprintf("cohort%s_rowData.csv", cohort_name)), row.names = FALSE)
}

# --- Cohort A: discovery -----------------------------------------------
# NOTE: the package exports these as objects named with spaces, not as
# conventionally-named functions like cohort_A(). Confirmed interactively
# against the installed package version (2024-10-31 build).
se_A <- `cohort A from Dampier et al.`()
extract_cohort(se_A, "A")

# --- Cohort B: matched-pair external validation -------------------------
se_B <- `cohort B from Dampier et al.`()
extract_cohort(se_B, "B")

# --- Cohort C: single-end, cross-protocol stress test --------------------
se_C <- `cohort C from Dampier et al.`()
extract_cohort(se_C, "C")

message("Done. Files written to: ", out_dir)
message("Column names confirmed against the installed package: sample type ",
        "is 'sampType', originating sub-study is 'study', and the counts ",
        "assay is named 'counts'. These are the exact values to pass into ",
        "the Phase 1 mini-audit script (cohortA_mini_audit.py).")
