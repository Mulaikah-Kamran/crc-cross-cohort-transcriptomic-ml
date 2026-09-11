"""
pool_geo_eocrc_locrc.py

Purpose
-------
GSE196006 (EOCRC arm, ~21 patients) and GSE251845 (LOCRC arm, ~22 patients)
come from the same underlying study, protocol, and lab, but GEO split them
into two separate accessions and they don't label tumor/normal the same way.
This script pools them into ONE external validation cohort, per Decision #5
in the locked spec, and normalizes both into a single consistent format.

What's different between the two source files, and how this handles it
-------------------------------------------------------------------------
GSE196006 column names look like: X15.018_L7_G821_htseq.out
  - patient id is embedded (15.018)
  - tissue is a code: 0 = Normal, 7 = Tumor (confirmed directly against the
    series' own sample metadata, not guessed from the column name alone)

GSE251845 column names look like: 24C_htseq.out or 35c_htseq.out
  - patient id is the leading number (24, 35, ...)
  - tissue is a C/N suffix: C = Tumor, N = Normal
  - one sample in the real data (patient 35's tumor sample) has a lowercase
    'c' instead of 'C' - this is handled by case-insensitive matching, not
    by special-casing that one sample, so any similar typo elsewhere in the
    file would also be caught correctly.

Output
------
Two files in the same directory as the inputs:
  - pooled_counts.parquet   (genes x samples, inner-joined on gene ID)
  - pooled_metadata.csv     (one row per sample: patient id, cohort arm,
                              tissue label, and cohort_role tag)

Usage
-----
    python pool_geo_eocrc_locrc.py \
        --gse196006 data/raw/geo_eocrc_locrc/GSE196006/GSE196006_raw_counts.csv.gz \
        --gse251845 data/raw/geo_eocrc_locrc/GSE251845/GSE251845_htseq_raw_counts.csv.gz \
        --out-dir data/raw/geo_eocrc_locrc
"""

import argparse
import re
import sys

import pandas as pd


def parse_gse196006_col(col: str):
    """X15.018_L7_G821_htseq.out -> ('15.018', 'Tumor')"""
    m = re.match(r"X?(\d+\.\d+)_[A-Za-z](\d)_", col)
    if not m:
        return None, None
    patient_id, tissue_code = m.group(1), m.group(2)
    tissue = "Tumor" if tissue_code == "7" else "Normal"
    return patient_id, tissue


def parse_gse251845_col(col: str):
    """24C_htseq.out -> ('24', 'Tumor'); 35c_htseq.out -> ('35', 'Tumor')
    (case-insensitive on purpose - the real data has at least one lowercase
    'c' that would otherwise silently fail to match)"""
    m = re.match(r"(\d+)([A-Za-z])_", col)
    if not m:
        return None, None
    patient_id, suffix = m.group(1), m.group(2).upper()
    tissue = "Tumor" if suffix == "C" else "Normal"
    return patient_id, tissue


def load_and_annotate(path: str, series: str, parser) -> tuple:
    df = pd.read_csv(path, index_col=0)
    df.index.name = "gene_id"

    meta_rows = []
    renamed_cols = {}
    n_unparsed = 0
    for col in df.columns:
        patient_id, tissue = parser(col)
        if patient_id is None:
            n_unparsed += 1
            print(f"[!] WARNING: could not parse column '{col}' from {series} - "
                  f"dropping it rather than guessing.", file=sys.stderr)
            continue
        new_name = f"{series}_{patient_id}_{tissue}"
        renamed_cols[col] = new_name
        meta_rows.append({
            "sample_id": new_name,
            "patient_id": f"{series}_{patient_id}",
            "cohort_arm": "EOCRC" if series == "GSE196006" else "LOCRC",
            "tissue_label": tissue,
            "geo_series": series,
            "cohort_role": "pooled_geo_validation",
        })

    if n_unparsed > 0:
        print(f"[!] {n_unparsed} column(s) in {series} could not be parsed and "
              f"were dropped. Do not proceed without checking why - this should "
              f"be 0 for a clean run.", file=sys.stderr)

    df = df.rename(columns=renamed_cols)
    df = df[[c for c in renamed_cols.values()]]  # keep only successfully parsed columns
    meta = pd.DataFrame(meta_rows)
    return df, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gse196006", required=True)
    parser.add_argument("--gse251845", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    df_eocrc, meta_eocrc = load_and_annotate(args.gse196006, "GSE196006", parse_gse196006_col)
    df_locrc, meta_locrc = load_and_annotate(args.gse251845, "GSE251845", parse_gse251845_col)

    print(f"GSE196006 (EOCRC): {df_eocrc.shape[1]} samples, {df_eocrc.shape[0]} genes")
    print(f"GSE251845 (LOCRC): {df_locrc.shape[1]} samples, {df_locrc.shape[0]} genes")

    # Inner join on gene ID - only keep genes present in both, since a
    # pooled cohort with mismatched gene sets per sample would be nonsense.
    shared_genes = df_eocrc.index.intersection(df_locrc.index)
    n_dropped = min(len(df_eocrc.index), len(df_locrc.index)) - len(shared_genes)
    print(f"Genes shared between both series: {len(shared_genes)} "
          f"({n_dropped} genes present in only one series were dropped)")

    pooled_counts = pd.concat(
        [df_eocrc.loc[shared_genes], df_locrc.loc[shared_genes]], axis=1
    )
    pooled_meta = pd.concat([meta_eocrc, meta_locrc], ignore_index=True)

    print(f"\nPooled cohort: {pooled_counts.shape[1]} samples total "
          f"({pooled_meta['patient_id'].nunique()} unique patients)")
    print(pooled_meta.groupby(["cohort_arm", "tissue_label"]).size())

    out_counts = pooled_counts.reset_index()
    out_counts.to_parquet(f"{args.out_dir}/pooled_counts.parquet")
    pooled_meta.to_csv(f"{args.out_dir}/pooled_metadata.csv", index=False)

    print(f"\nWrote pooled_counts.parquet and pooled_metadata.csv to {args.out_dir}")
    print("Sanity check before moving on: patient_id count above should be "
          "roughly 21 (EOCRC) + 22 (LOCRC) = 43, with 2 samples (Tumor + "
          "Normal) per patient, ~86 total samples. If it doesn't match, "
          "stop and investigate rather than proceeding.")


if __name__ == "__main__":
    main()
