"""
field_effect_check.py

Hypothesis H5 (exploratory): does the frozen model assign genuinely
healthy tissue an even lower tumor-probability than adjacent-normal
tissue, consistent with a healthy -> adjacent-normal -> tumor field
effect gradient?

CAVEAT: the "healthy" samples (BarcUVa/GTEx) are exactly the ones the
Cohort A audit found completely confounded with data source. This
cannot distinguish genuine biology from those sources simply looking
technically different. Exploratory only, not confirmatory.

Usage
-----
    python field_effect_check.py \
        --cohortA-counts data/raw/fieldeffectcrc/cohortA_counts.parquet \
        --cohortA-coldata data/raw/fieldeffectcrc/cohortA_colData.csv \
        --models-dir results/models \
        --out results/tables/phase6_field_effect_check.csv
"""

import argparse
import glob
import re
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src/feature_selection")
import sklearn_transformers  # noqa: F401


def log2_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    return np.log2(cpm + 1)


def strip_ensembl_version(gene_id: str) -> str:
    return re.sub(r"^(ENSG\d+)\.\d+$", r"\1", str(gene_id))


def align_to_training_genes(log_expr, train_gene_ids, train_gene_means):
    aligned = log_expr.reindex(index=train_gene_ids)
    n_present = aligned.notna().any(axis=1).sum()
    coverage = n_present / len(train_gene_ids)
    for gene in aligned.index[aligned.isna().all(axis=1)]:
        aligned.loc[gene] = train_gene_means.get(gene, 0.0)
    return aligned, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohortA-counts", required=True)
    parser.add_argument("--cohortA-coldata", required=True)
    parser.add_argument("--models-dir", default="results/models")
    parser.add_argument("--out", default="results/tables/phase6_field_effect_check.csv")
    args = parser.parse_args()

    with open(f"{args.models_dir}/training_gene_order.txt") as f:
        train_gene_ids = f.read().splitlines()
    train_gene_means = pd.read_parquet(f"{args.models_dir}/training_gene_means.parquet")["mean_log2cpm"]

    counts = pd.read_parquet(args.cohortA_counts)
    gene_col = "gene_id" if "gene_id" in counts.columns else counts.columns[0]
    counts = counts.set_index(gene_col)
    counts.index = counts.index.map(strip_ensembl_version)
    counts = counts.groupby(counts.index).sum()

    coldata = pd.read_csv(args.cohortA_coldata).set_index("sample_id")
    common = coldata.index.intersection(counts.columns)
    counts, coldata = counts[common], coldata.loc[common]

    print(f"Cohort A (full, all 3 tissue types): {len(coldata)} samples")
    print(coldata["sampType"].value_counts())

    log_expr = log2_cpm(counts)
    aligned, coverage = align_to_training_genes(log_expr, train_gene_ids, train_gene_means)
    print(f"Gene coverage against training set: {coverage:.1%}")
    X = aligned.T.values

    frozen_paths = sorted(glob.glob(f"{args.models_dir}/frozen_*.joblib"))

    rows = []
    for model_path in frozen_paths:
        model_name = model_path.split("frozen_")[-1].replace(".joblib", "")
        pipeline = joblib.load(model_path)
        y_proba = pipeline.predict_proba(X)[:, 1]

        scores = pd.Series(y_proba, index=coldata.index)
        mean_hlt = scores[coldata["sampType"] == "HLT"].mean()
        mean_nat = scores[coldata["sampType"] == "NAT"].mean()
        mean_crc = scores[coldata["sampType"] == "CRC"].mean()

        gradient_correct = mean_hlt < mean_nat < mean_crc

        print(f"{model_name:30s} HLT={mean_hlt:.3f}  NAT={mean_nat:.3f}  CRC={mean_crc:.3f}  "
              f"gradient {'CONFIRMED' if gradient_correct else 'not confirmed'}")

        rows.append({
            "model": model_name,
            "mean_predicted_tumor_prob_HLT": round(mean_hlt, 4),
            "mean_predicted_tumor_prob_NAT": round(mean_nat, 4),
            "mean_predicted_tumor_prob_CRC": round(mean_crc, 4),
            "gradient_HLT_lt_NAT_lt_CRC": gradient_correct,
        })

    result = pd.DataFrame(rows)
    result.to_csv(args.out, index=False)

    n_confirmed = result["gradient_HLT_lt_NAT_lt_CRC"].sum()
    print(f"\n{n_confirmed} of {len(result)} models show the expected HLT < NAT < CRC gradient")
    print(f"Saved to {args.out}")

    print("\nCAVEAT (do not omit when reporting this): every HLT sample comes from only two "
          "sources (BarcUVa, GTEx), the exact confound the Cohort A audit found. Exploratory "
          "and hypothesis-generating only.")


if __name__ == "__main__":
    main()
