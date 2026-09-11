"""
check_cohortC_TCGA_overlap.py

The Phase 4 investigation found that 195 of Cohort C's 207 tumor samples
(94%) originate from TCGA - the same source as our discovery cohort.
This checks whether these are literally the same patients (leakage) or
genuinely different TCGA patients (a real confound, but not leakage).

Usage
-----
    python check_cohortC_TCGA_overlap.py \
        --cohortC-coldata data/raw/fieldeffectcrc/cohortC_colData.csv \
        --tcga-coldata data/raw/tcga_coad/tcga_coad_colData.parquet
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohortC-coldata", required=True)
    parser.add_argument("--tcga-coldata", required=True)
    args = parser.parse_args()

    cohortC = pd.read_csv(args.__dict__["cohortC_coldata"])
    tcga = pd.read_parquet(args.__dict__["tcga_coldata"])

    c_tcga_subset = cohortC[cohortC["study"] == "TCGA"]
    print(f"Cohort C TCGA-origin samples: {len(c_tcga_subset)}")
    print(c_tcga_subset["sampType"].value_counts())

    c_tcga_subs = c_tcga_subset["subId"].astype(str)
    print(f"\nSample of Cohort C's TCGA-origin subject IDs: {sorted(set(c_tcga_subs))[:10]}")

    patient_col = None
    for candidate in ["patient", "patient_id", "bcr_patient_barcode", "submitter_id"]:
        if candidate in tcga.columns:
            patient_col = candidate
            break

    if patient_col is None:
        print(f"[!] Could not find a patient identifier column in the TCGA colData. "
              f"Available columns: {list(tcga.columns)}")
        return

    tcga_patients = set(tcga[patient_col].astype(str))
    exact_overlap = set(c_tcga_subs) & tcga_patients

    substring_overlap = set()
    for sub in c_tcga_subs:
        for pat in tcga_patients:
            if sub in pat or pat in sub:
                substring_overlap.add(sub)
                break

    print(f"\nExact ID overlap with TCGA-COAD discovery patients: {len(exact_overlap)} "
          f"of {len(set(c_tcga_subs))} unique Cohort C TCGA-origin patients")
    print(f"Substring-match overlap (in case of ID format differences): "
          f"{len(substring_overlap)} of {len(set(c_tcga_subs))}")

    if exact_overlap or substring_overlap:
        overlap_count = len(exact_overlap | substring_overlap)
        pct = overlap_count / len(set(c_tcga_subs)) * 100
        print(f"\nVERDICT: {overlap_count} patients ({pct:.0f}% of Cohort C's TCGA-origin "
              f"patients) appear to be THE SAME patients as in the TCGA-COAD discovery "
              f"cohort. This is leakage, not just a confound - Cohort C's strong Phase 4 "
              f"performance is at least partly explained by the model having effectively "
              f"seen these patients (or their tumor's twin sample) during training. "
              f"These specific samples should be excluded from Cohort C before treating "
              f"it as a valid external validation result.")
    else:
        print(f"\nVERDICT: no patient ID overlap found. These are likely genuinely "
              f"different TCGA patients - not literal leakage, but Cohort C's tumor "
              f"class still shares TCGA's broad population/processing lineage while its "
              f"normal class does not (MtSinai-dominated), which is a real confound "
              f"worth stating plainly rather than presenting Cohort C's result as a "
              f"clean, unconfounded generalization test.")


if __name__ == "__main__":
    main()
