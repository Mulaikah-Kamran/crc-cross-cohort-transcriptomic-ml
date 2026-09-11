"""
extract_validation_cohorts.py

Purpose
-------
Builds the two remaining external validation cohorts from raw data we
already downloaded in Phase 1, no new downloads needed, just filtering.

Validation 2: the non-TCGA clinical sites (HebeiMU, KoreaAMC, KoreaPNU,
Mayo) subset of FieldEffectCrc Cohort A. Expected: 38 samples (21 CRC,
17 NAT), confirmed against the real Cohort A crosstab during the Cohort A
audit.

Validation 4: FieldEffectCrc Cohort B with ALL TCGA-origin patients
removed (not just the 3 that technically overlapped with the TCGA-COAD
discovery cohort - per the locked decision, all 4 are dropped for a
clean "zero TCGA outside discovery" rule). Expected: 22 samples
(11 matched patient pairs).

Both expected counts are asserted explicitly - if the real data doesn't
match, this stops and prints what it actually found rather than silently
producing a cohort that's a different size than what we designed around.

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
              f"Stop and investigate before using this output - do not assume "
              f"this is fine just because the script ran without crashing.")

    subset_counts.to_parquet(f"{out_dir}/{out_prefix}_counts.parquet")
    subset_meta.to_csv(f"{out_dir}/{out_prefix}_colData.csv", index=False)
    print(f"Wrote {out_dir}/{out_prefix}_counts.parquet and _colData.csv")

    return subset_meta, subset_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohortA-counts", required=True)
    parser.add_argument("--cohortA-coldata", required=True)
    parser.add_argument("--cohortB-counts", required=True)
    parser.add_argument("--cohortB-coldata", required=True)
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


if __name__ == "__main__":
    main()
