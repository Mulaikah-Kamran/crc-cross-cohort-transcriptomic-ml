#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# 03_download_geo_eocrc_locrc.R
#
# Purpose: Download GSE196006 (EOCRC arm) and GSE251845 (LOCRC arm) from the
# same underlying study (Penn State Genome Sciences core, matched tumor/
# normal, STAR/hg38), and pool them into ONE external validation cohort as
# per Decision #5 in the locked spec.
#
# NOTE: Run this locally with internet access to NCBI GEO — not reachable
# from Claude's sandbox.
#
# IMPORTANT: GEO series don't always store raw counts in a uniform way.
# This script downloads the supplementary files and the sample metadata;
# you MUST inspect the downloaded supplementary file(s) manually the first
# time to confirm they contain a raw gene x sample count matrix (rather than,
# e.g., only normalized values or per-sample files that need concatenating).
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(GEOquery)
  library(arrow)
  library(dplyr)
})

out_dir <- "data/raw/geo_eocrc_locrc"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

accessions <- c("GSE196006", "GSE251845")

for (acc in accessions) {
  message(sprintf("--- Downloading %s ---", acc))

  # Sample-level metadata (phenotype/characteristics)
  gse <- getGEO(acc, GSEMatrix = TRUE, getGPL = FALSE)
  pdata <- pData(gse[[1]])
  pdata$geo_series <- acc
  write.csv(pdata, file.path(out_dir, sprintf("%s_sample_metadata.csv", acc)),
            row.names = FALSE)

  # Supplementary files (often contains the raw counts matrix, if deposited)
  getGEOSuppFiles(acc, baseDir = out_dir, makeDirectory = TRUE)
}

message("Done downloading raw files.")
message("NEXT STEP (manual): open data/raw/geo_eocrc_locrc/<ACC>/ and identify ",
        "the raw counts file for each series. Load both into R or Python, ",
        "confirm they share the same gene ID scheme (Ensembl/hg38 STAR output ",
        "per the source publication), tag each sample with its cohort ",
        "(EOCRC vs LOCRC) AND with 'pooled_geo_validation' as its cohort role, ",
        "then concatenate into one counts matrix + one metadata table before ",
        "writing to parquet/csv for the Python pipeline. This manual check is ",
        "deliberate: GEO supplementary file structure is not standardized ",
        "enough to safely automate blindly.")
