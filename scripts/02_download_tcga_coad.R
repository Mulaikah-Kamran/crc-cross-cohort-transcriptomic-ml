#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# 02_download_tcga_coad.R
#
# Purpose: Query and download TCGA-COAD RNA-seq (STAR - Counts, GDC harmonized,
# GRCh38), restricted to Primary Tumor + Solid Tissue Normal samples, and
# write a tidy counts matrix + clinical/sample metadata for Python.
#
# NOTE: Run this locally with internet access to the GDC API
# (https://api.gdc.cancer.gov) - not reachable from Claude's sandbox.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(TCGAbiolinks)
  library(SummarizedExperiment)
  library(arrow)
})

out_dir <- "data/raw/tcga_coad"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

query <- GDCquery(
  project = "TCGA-COAD",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts",
  sample.type = c("Primary Tumor", "Solid Tissue Normal")
)

GDCdownload(query, method = "api", files.per.chunk = 20)
data_se <- GDCprepare(query)

message(sprintf("TCGA-COAD: %d genes x %d samples", nrow(data_se), ncol(data_se)))
print(table(data_se$sample_type))

counts <- as.data.frame(assay(data_se, "unstranded"))
counts$gene_id <- rownames(counts)

col_meta <- as.data.frame(colData(data_se))

list_cols <- sapply(col_meta, is.list)
for (col in names(col_meta)[list_cols]) {
  col_meta[[col]] <- sapply(col_meta[[col]], function(x) paste(unlist(x), collapse = "; "))
}
col_meta$sample_id <- rownames(col_meta)

write_parquet(counts, file.path(out_dir, "tcga_coad_counts.parquet"))
write_parquet(col_meta, file.path(out_dir, "tcga_coad_colData.parquet"))

message("Done. Verify the reported tumor/normal N against the audit table ",
        "(expected roughly 444-481 tumor / 39-41 normal, GDC counts shift ",
        "slightly over time as data gets periodically re-harmonized).")
