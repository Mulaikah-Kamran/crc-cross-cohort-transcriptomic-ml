"""
check_cohortB_overlap.py

Purpose
-------
Cohort A's audit revealed that its samples were pooled from several original
studies, including TCGA and four smaller clinical sites (HebeiMU, KoreaAMC,
KoreaPNU, Mayo). Those same original studies are now being used directly:
TCGA-COAD is the new discovery cohort, and the four clinical sites are being
used as a separate validation set.

FieldEffectCrc Cohort B (the 30 matched-pair samples) was curated by the same
package author, from the same pool of underlying studies. Before treating
Cohort B as an independent fourth validation set, we need to rule out that
its samples are actually a subset of patients already used in TCGA-COAD or
the clinical-sites validation set. Reusing the same patients across two roles
would quietly break independence, even without any code bug, it would just
be the same patients counted twice.

What this does
---------------
1. Prints Cohort B's study x sampType breakdown, this alone tells us which
   original studies contributed to Cohort B.
2. If any Cohort B samples come from TCGA, GTEx/BarcUVa (excluded already),
   or the four clinical sites, checks subject ID overlap directly against
   the corresponding cohort's colData already extracted.
3. Reports a clear verdict: safe to use Cohort B as-is, safe after dropping
   specific overlapping samples, or drop Cohort B entirely.

Usage
-----
    python check_cohortB_overlap.py \
        --cohortB-coldata data/raw/fieldeffectcrc/cohortB_colData.csv \
        --cohortA-coldata data/raw/fieldeffectcrc/cohortA_colData.csv \
        --tcga-coldata data/raw/tcga_coad/tcga_coad_colData.parquet
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohortB-coldata", required=True)
    parser.add_argument("--cohortA-coldata", required=True)
    parser.add_argument("--tcga-coldata", required=True)
    args = parser.parse_args()

    cohortB = pd.read_csv(args.__dict__["cohortB_coldata"])
    cohortA = pd.read_csv(args.__dict__["cohortA_coldata"])
    tcga = pd.read_parquet(args.__dict__["tcga_coldata"])

    print("=== Cohort B: study x sampType breakdown ===")
    print(pd.crosstab(cohortB["study"], cohortB["sampType"]))
    print()

    clinical_sites = {"HebeiMU", "KoreaAMC", "KoreaPNU", "Mayo"}
    b_studies = set(cohortB["study"].unique())

    overlaps_tcga = "TCGA" in b_studies
    overlaps_clinical = bool(b_studies & clinical_sites)

    if not overlaps_tcga and not overlaps_clinical:
        print("Cohort B draws from studies other than TCGA or the four "
              "clinical sites used elsewhere in the new architecture. "
              f"Cohort B's studies are: {sorted(b_studies)}")
        print("VERDICT: no overlap risk detected from study membership alone. "
              "Safe to keep Cohort B as an independent validation set, "
              "though it's still worth a quick subject-ID sanity check if "
              "any of these studies appear anywhere else in the pipeline.")
        return

    print(f"Cohort B contains samples from: {sorted(b_studies)}")
    if overlaps_tcga:
        print("[!] Cohort B includes TCGA-sourced samples - checking for "
              "subject ID overlap against the TCGA-COAD discovery cohort...")

        b_tcga_subs = cohortB.loc[cohortB["study"] == "TCGA", "subId"].astype(str)

        # TCGA patient barcodes are typically formatted like TCGA-AA-3660.
        # Cohort B's subId may or may not include the full barcode - check
        # both exact match and substring containment to be safe.
        patient_col = None
        for candidate in ["patient", "patient_id", "bcr_patient_barcode", "submitter_id"]:
            if candidate in tcga.columns:
                patient_col = candidate
                break

        if patient_col is None:
            print("[!] Could not find a patient identifier column in the TCGA "
                  f"colData. Available columns: {list(tcga.columns)}")
            print("STOP: cannot verify overlap automatically. Do not assume "
                  "no overlap, inspect the columns manually before proceeding.")
            return

        tcga_patients = set(tcga[patient_col].astype(str))
        exact_overlap = set(b_tcga_subs) & tcga_patients

        # substring check in case of ID format differences (e.g. short vs full barcode)
        substring_overlap = set()
        for sub in b_tcga_subs:
            for pat in tcga_patients:
                if sub in pat or pat in sub:
                    substring_overlap.add(sub)
                    break

        print(f"Cohort B TCGA-origin subject IDs: {sorted(set(b_tcga_subs))}")
        print(f"Exact ID overlap with TCGA-COAD patients: {sorted(exact_overlap)}")
        print(f"Substring-match overlap (format differences): {sorted(substring_overlap)}")

        if exact_overlap or substring_overlap:
            print("\nVERDICT: overlap detected. These specific Cohort B samples "
                  "must be dropped before using Cohort B as a validation set, "
                  "or drop Cohort B entirely if the overlap covers most of it.")
        else:
            print("\nVERDICT: Cohort B draws from the TCGA study, but no matching "
                  "patient IDs were found against the actual TCGA-COAD discovery "
                  "cohort samples. Likely safe, but given this used a substring "
                  "heuristic rather than a guaranteed ID scheme match, treat this "
                  "as provisional rather than certain.")

    if overlaps_clinical:
        overlapping_sites = b_studies & clinical_sites
        print(f"\n[!] Cohort B includes samples from clinical site(s): "
              f"{sorted(overlapping_sites)}, which are also used as a "
              "separate validation set. Checking subject ID overlap...")

        b_clinical_subs = cohortB.loc[
            cohortB["study"].isin(clinical_sites), "subId"
        ].astype(str)
        a_clinical_subs = cohortA.loc[
            cohortA["study"].isin(overlapping_sites), "subId"
        ].astype(str)

        overlap = set(b_clinical_subs) & set(a_clinical_subs)
        print(f"Cohort B clinical-site subject IDs: {sorted(set(b_clinical_subs))}")
        print(f"Overlapping subject IDs with the clinical-sites validation set: "
              f"{sorted(overlap)}")

        if overlap:
            print("\nVERDICT: overlap detected with the clinical-sites validation "
                  "set. Drop these specific patients from one of the two roles "
                  "before using both as independent validation sets.")
        else:
            print("\nVERDICT: no subject ID overlap found with the clinical-sites "
                  "validation set.")


if __name__ == "__main__":
    main()
