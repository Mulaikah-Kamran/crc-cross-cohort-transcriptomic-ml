#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# 02_download_tcga_coad.R
#
# Purpose: Query and download TCGA-COAD RNA-seq (STAR - Counts, GDC harmonized,
# GRCh38), restricted to Primary Tumor + Solid Tissue Normal samples, and
# write a tidy counts matrix + clinical/sample metadata for Python.
#
# NOTE: Run this locally with internet access to the GDC API
# (https://api.gdc.cancer.gov) — not reachable from Claude's sandbox.
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
data_se <- GDCprepare(query)   # returns a SummarizedExperiment

message(sprintf("TCGA-COAD: %d genes x %d samples", nrow(data_se), ncol(data_se)))
print(table(data_se$sample_type))   # sanity check tumor vs normal counts — should be ~444 vs ~41

counts <- as.data.frame(assay(data_se, "unstranded"))  # confirm assay name; GDC STAR output
                                                        # has stranded/unstranded options
counts$gene_id <- rownames(counts)

col_meta <- as.data.frame(colData(data_se))
col_meta$sample_id <- rownames(col_meta)

write_parquet(counts, file.path(out_dir, "tcga_coad_counts.parquet"))
write.csv(col_meta, file.path(out_dir, "tcga_coad_colData.csv"), row.names = FALSE)

message("Done. IMPORTANT: verify the reported tumor/normal N against the audit ",
        "table (expected ~444 tumor / ~41 normal) — GDC counts can shift slightly ",
        "as data is periodically re-harmonized.")
