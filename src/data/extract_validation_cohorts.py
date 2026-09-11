"""
extract_validation_cohorts.py

Builds Validation 2 (non-TCGA clinical sites), Validation 4 (Cohort B,
TCGA-free), and Validation 3 clean (Cohort C, TCGA-free) from raw data
already downloaded in Phase 1.

Validation 3 correction: the Phase 4 investigation found 135 of Cohort
C's 195 TCGA-origin patients are literally the same patients as the
TCGA-COAD discovery cohort - real leakage, not just a confound. Same
precedent as Cohort B: drop ALL TCGA-origin samples for one clean rule.

Usage
-----
    python extract_validation_cohorts.py \
        --cohortA-counts data/raw/fieldeffectcrc/cohortA_counts.parquet \
        --cohortA-coldata data/raw/fieldeffectcrc/cohortA_colData.csv \
        --cohortB-counts data/raw/fieldeffectcrc/cohortB_counts.parquet \
        --cohortB-coldata data/raw/fieldeffectcrc/cohortB_colData.csv \
        --out-dir data/raw/fieldeffectcrc
"""

import argparse
import pandas as pd

CLINICAL_SITES = {"HebeiMU", "KoreaAMC", "KoreaPNU", "Mayo"}


def extract_subset(counts_path: str, coldata_path: str, keep_mask_fn,
                    cohort_role: str, expected_n: int, out_prefix: str, out_dir: str):
    counts = pd.read_parquet(counts_path)
    coldata = pd.read_csv(coldata_path)

    subset_meta = coldata[keep_mask_fn(coldata)].copy()
    subset_meta["cohort_role"] = cohort_role

    sample_ids = subset_meta["sample_id"].tolist()
    available_ids = [s for s in sample_ids if s in counts.columns]
    missing_ids = set(sample_ids) - set(available_ids)
    if missing_ids:
        print(f"[!] WARNING: {len(missing_ids)} sample(s) in the filtered metadata "
              f"are not found in the counts matrix columns: {missing_ids}. "
              f"Proceeding with the {len(available_ids)} that are available, but "
              f"this mismatch shouldn't happen and is worth checking.")

    subset_counts = counts[["gene_id"] + available_ids]

    n_actual = len(subset_meta)
    print(f"\n{cohort_role}: {n_actual} samples (expected {expected_n})")
    if "sampType" in subset_meta.columns:
        print(subset_meta["sampType"].value_counts())
    if n_actual != expected_n:
        print(f"[!] MISMATCH: got {n_actual}, expected {expected_n}. "
              f"Stop and investigate before using this output.")

    subset_counts.to_parquet(f"{out_dir}/{out_prefix}_counts.parquet")
    subset_meta.to_csv(f"{out_dir}/{out_prefix}_colData.csv", index=False)
    print(f"Wrote {out_dir}/{out_prefix}_counts.parquet and _colData.csv")

    return subset_meta, subset_counts


def extract_cohortC_clean(counts_path: str, coldata_path: str, out_dir: str):
    extract_subset(
        counts_path=counts_path,
        coldata_path=coldata_path,
        keep_mask_fn=lambda df: df["study"] != "TCGA",
        cohort_role="validation_3_cohortC_clean",
        expected_n=80,
        out_prefix="validation3_cohortC_clean",
        out_dir=out_dir,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohortA-counts", required=True)
    parser.add_argument("--cohortA-coldata", required=True)
    parser.add_argument("--cohortB-counts", required=True)
    parser.add_argument("--cohortB-coldata", required=True)
    parser.add_argument("--cohortC-counts", required=False,
                         default="data/raw/fieldeffectcrc/cohortC_counts.parquet")
    parser.add_argument("--cohortC-coldata", required=False,
                         default="data/raw/fieldeffectcrc/cohortC_colData.csv")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    extract_subset(
        counts_path=args.__dict__["cohortA_counts"],
        coldata_path=args.__dict__["cohortA_coldata"],
        keep_mask_fn=lambda df: df["study"].isin(CLINICAL_SITES),
        cohort_role="validation_2_clinical_sites",
        expected_n=38,
        out_prefix="validation2_clinical_sites",
        out_dir=args.out_dir,
    )

    extract_subset(
        counts_path=args.__dict__["cohortB_counts"],
        coldata_path=args.__dict__["cohortB_coldata"],
        keep_mask_fn=lambda df: df["study"] != "TCGA",
        cohort_role="validation_4_cohortB_clean",
        expected_n=22,
        out_prefix="validation4_cohortB_clean",
        out_dir=args.out_dir,
    )

    extract_cohortC_clean(
        counts_path=args.__dict__["cohortC_counts"],
        coldata_path=args.__dict__["cohortC_coldata"],
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
