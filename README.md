# crc-cross-cohort-transcriptomic-ml

Bulk RNA-seq machine learning study on colorectal cancer, built around one question that gets skipped more often than it should: when you find genes that separate tumor from normal tissue, do they actually mean something, or did your model just learn to recognize which lab processed the sample?

## What this project is actually about

Telling colorectal tumor tissue apart from normal tissue using gene expression is not a hard problem. Tumor and normal tissue are wildly different biologically, and plenty of published studies report accuracy north of 95%. That part isn't the contribution here.

What this project asks instead: if you train a classifier on one cohort and test it on three other cohorts collected by different labs, on different sequencing machines, sometimes even using a different definition of "normal" tissue, which genes still hold up? And which feature-selection method, out of several reasonable choices, actually gives you genes worth trusting rather than an artifact of the cohort you happened to train on?

This grew directly out of an earlier project (an early-onset vs late-onset CRC classifier) where a dataset that looked great turned out to be picking up cohort and sequencing differences rather than biology. That failure is the whole reason this project is designed the way it is.

## The short version of the design

- **Discovery cohort:** FieldEffectCrc Cohort A, 834 samples, tumor vs healthy tissue.
- **Four separate external validation sets**, deliberately chosen to differ from the discovery cohort in different ways: a matched-pair cohort on the same sequencing protocol, a single-end sequencing cohort that stress-tests protocol sensitivity on purpose, TCGA-COAD (which uses adjacent-normal rather than healthy tissue as its baseline), and a pooled cohort from an independent lab.
- **Three feature-selection strategies** compared head to head: differential-expression-informed, purely data-driven, and pathway-level.
- Everything is built around patient-level, leakage-safe nested cross-validation. Feature selection never sees the test fold, and the external cohorts are frozen and touched exactly once.

The full reasoning behind every one of these choices, including the datasets we considered and rejected, lives in [`docs/P2.2_Project_Design_Specification.md`](docs/P2.2_Project_Design_Specification.md). That document is the single source of truth for scope. If something in the code ever seems to contradict it, the spec wins and the code is wrong.

## Where things stand right now

Phase 1 (data acquisition and a mandatory batch-effect audit) is in progress. See [`PHASE1_README.md`](PHASE1_README.md) for exact run instructions. Two things worth knowing:

1. The R scripts that pull data from Bioconductor, GDC, and GEO have to be run locally with normal internet access. They're written and documented but not yet run against the live data.
2. The Python analysis scripts (gene ID overlap check, and a PCA/UMAP audit for hidden batch structure in the discovery cohort) have already been built and tested against synthetic data with a deliberately planted batch effect, to confirm they actually catch the failure mode they're meant to catch before they ever touch real data. Figures from that test run are in `results/figures/cohortA_audit_synthetic_test_example/`.

## Repository layout

```
configs/      accession registry and pipeline parameters
data/         raw data is never committed; see data/README.md
docs/         the locked project design specification
notebooks/    exploratory work, kept separate from reusable code
scripts/      R scripts for pulling data from Bioconductor / GDC / GEO
src/          Python source code for the actual pipeline
results/      figures and tables produced by the pipeline
reports/      write-up drafts
```

## Setup

```bash
# R side, for data acquisition
conda env create -f environment-r.yml
conda activate p2.2-crc-r
Rscript scripts/00_setup_r_env.R

# Python side, for everything downstream
pip install -r requirements.txt
```

## Why "cross-cohort" and not just "colorectal cancer classifier"

Because the classifier part is the easy, already-done part. The interesting part, and the part this repo is actually structured to answer, is whether a signal found in one place survives contact with data it has never seen. Most of the design decisions in here exist to make that question answerable honestly rather than to make the accuracy number look good.
