"""
gene_overlap_check.py

Purpose
-------
Quantify how many genes are usable in common across all four cohorts before
any feature selection happens. A frozen model trained on Cohort A features
can only be applied to a validation cohort for the genes both share, so this
number directly bounds what the whole study can do.

What it does
------------
1. Loads each cohort's gene ID list (from the *_rowData.csv / *_counts files
   produced by the R extraction scripts).
2. Strips Ensembl version suffixes (e.g. ENSG00000141510.11 -> ENSG00000141510)
   since different pipelines annotate versions inconsistently even on the
   same genome build.
3. Reports pairwise and overall intersection sizes, and flags if the overlap
   is suspiciously small (a sign of a build/annotation mismatch that needs
   investigating before Phase 2).

Usage
-----
    python gene_overlap_check.py \
        --cohortA data/raw/fieldeffectcrc/cohortA_counts.parquet \
        --cohortB data/raw/fieldeffectcrc/cohortB_counts.parquet \
        --cohortC data/raw/fieldeffectcrc/cohortC_counts.parquet \
        --tcga    data/raw/tcga_coad/tcga_coad_counts.parquet \
        --geo     data/raw/geo_eocrc_locrc/pooled_counts.parquet
"""

import argparse
import re
import pandas as pd


def strip_ensembl_version(gene_id: str) -> str:
    """ENSG00000141510.11 -> ENSG00000141510. Leaves non-Ensembl IDs (e.g.
    gene symbols) untouched."""
    return re.sub(r"^(ENSG\d+)\.\d+$", r"\1", str(gene_id))


def load_gene_ids(path: str, gene_col: str = "gene_id") -> set:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, columns=[gene_col])
    else:
        df = pd.read_csv(path, usecols=[gene_col])
    ids = df[gene_col].map(strip_ensembl_version)
    return set(ids)


def overlap_report(cohort_gene_sets: dict) -> None:
    names = list(cohort_gene_sets.keys())
    sizes = {n: len(g) for n, g in cohort_gene_sets.items()}

    print("=== Per-cohort gene counts (post version-stripping) ===")
    for n in names:
        print(f"  {n:12s}: {sizes[n]:6d} genes")

    print("\n=== Pairwise overlap (with discovery cohort A) ===")
    if "cohortA" not in cohort_gene_sets:
        print("  [!] No cohort named 'cohortA' found — skipping pairwise-vs-discovery view")
    else:
        a = cohort_gene_sets["cohortA"]
        for n in names:
            if n == "cohortA":
                continue
            overlap = a & cohort_gene_sets[n]
            pct_of_a = 100 * len(overlap) / max(len(a), 1)
            print(f"  A ∩ {n:10s}: {len(overlap):6d} genes  ({pct_of_a:.1f}% of Cohort A's genes)")

    print("\n=== Overall intersection across ALL cohorts ===")
    all_common = set.intersection(*cohort_gene_sets.values())
    print(f"  Usable in every cohort: {len(all_common)} genes")

    if len(all_common) < 10000:
        print("\n  [!] WARNING: overall overlap is below ~10,000 genes. This is a "
              "genuine flag, not just a low number — check for gene ID / genome "
              "build / annotation-version mismatches before proceeding to Phase 2. "
              "Do not silently proceed with a small overlap without investigating why.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohortA", required=True)
    parser.add_argument("--cohortB", required=True)
    parser.add_argument("--cohortC", required=True)
    parser.add_argument("--tcga", required=True)
    parser.add_argument("--geo", required=True)
    args = parser.parse_args()

    cohort_gene_sets = {
        "cohortA": load_gene_ids(args.cohortA),
        "cohortB": load_gene_ids(args.cohortB),
        "cohortC": load_gene_ids(args.cohortC),
        "tcga_coad": load_gene_ids(args.tcga),
        "geo_pooled": load_gene_ids(args.geo),
    }
    overlap_report(cohort_gene_sets)


if __name__ == "__main__":
    main()
