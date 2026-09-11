"""
run_phase4_external_validation.py

Applies all 9 frozen pipelines (from freeze_models.py) to all 4 external
validation cohorts, all now confirmed free of TCGA-origin patients.

Usage
-----
    python run_phase4_external_validation.py \
        --models-dir results/models \
        --out results/tables/phase4_external_validation_results.csv
"""

import argparse
import glob
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, "src/feature_selection")
from sklearn_transformers import load_gmt_file  # noqa: F401


def log2_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    return np.log2(cpm + 1)


def strip_ensembl_version(gene_id: str) -> str:
    return re.sub(r"^(ENSG\d+)\.\d+$", r"\1", str(gene_id))


def align_to_training_genes(log_expr: pd.DataFrame, train_gene_ids: list,
                             train_gene_means: pd.Series) -> tuple:
    aligned = log_expr.reindex(index=train_gene_ids)
    n_present = aligned.notna().any(axis=1).sum()
    coverage = n_present / len(train_gene_ids)
    for gene in aligned.index[aligned.isna().all(axis=1)]:
        aligned.loc[gene] = train_gene_means.get(gene, 0.0)
    return aligned, coverage


COHORT_CONFIGS = [
    {
        "name": "validation1_geo_pooled",
        "counts": "data/raw/geo_eocrc_locrc/pooled_counts.parquet",
        "coldata": "data/raw/geo_eocrc_locrc/pooled_metadata.csv",
        "label_col": "tissue_label", "positive": "Tumor", "negative": "Normal",
        "exclude_values": [],
    },
    {
        "name": "validation2_clinical_sites",
        "counts": "data/raw/fieldeffectcrc/validation2_clinical_sites_counts.parquet",
        "coldata": "data/raw/fieldeffectcrc/validation2_clinical_sites_colData.csv",
        "label_col": "sampType", "positive": "CRC", "negative": "NAT",
        "exclude_values": [],
    },
    {
        "name": "validation3_cohortC_clean",
        "counts": "data/raw/fieldeffectcrc/validation3_cohortC_clean_counts.parquet",
        "coldata": "data/raw/fieldeffectcrc/validation3_cohortC_clean_colData.csv",
        "label_col": "sampType", "positive": "CRC", "negative": "NAT",
        "exclude_values": ["HLT"],
    },
    {
        "name": "validation4_cohortB_clean",
        "counts": "data/raw/fieldeffectcrc/validation4_cohortB_clean_counts.parquet",
        "coldata": "data/raw/fieldeffectcrc/validation4_cohortB_clean_colData.csv",
        "label_col": "sampType", "positive": "CRC", "negative": "NAT",
        "exclude_values": [],
    },
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default="results/models")
    parser.add_argument("--out", default="results/tables/phase4_external_validation_results.csv")
    args = parser.parse_args()

    with open(f"{args.models_dir}/training_gene_order.txt") as f:
        train_gene_ids = f.read().splitlines()
    train_gene_means = pd.read_parquet(f"{args.models_dir}/training_gene_means.parquet")["mean_log2cpm"]

    frozen_paths = sorted(glob.glob(f"{args.models_dir}/frozen_*.joblib"))
    print(f"Found {len(frozen_paths)} frozen models, {len(train_gene_ids)} training genes.")

    results = []
    for cohort_cfg in COHORT_CONFIGS:
        print(f"\n=== {cohort_cfg['name']} ===")
        counts = pd.read_parquet(cohort_cfg["counts"])
        gene_col = "gene_id" if "gene_id" in counts.columns else counts.columns[0]
        counts = counts.set_index(gene_col)
        counts.index = counts.index.map(strip_ensembl_version)
        counts = counts.groupby(counts.index).sum()

        coldata = pd.read_csv(cohort_cfg["coldata"])
        sample_id_col = "sample_id" if "sample_id" in coldata.columns else coldata.columns[0]
        coldata = coldata.set_index(sample_id_col)

        n_before = len(coldata)
        coldata = coldata[~coldata[cohort_cfg["label_col"]].isin(cohort_cfg["exclude_values"])]
        n_excluded = n_before - len(coldata)
        if n_excluded > 0:
            print(f"  Excluded {n_excluded} samples with label(s) {cohort_cfg['exclude_values']} "
                  f"(outside the tumor-vs-NAT endpoint, not part of this evaluation).")

        common_samples = coldata.index.intersection(counts.columns)
        counts = counts[common_samples]
        coldata = coldata.loc[common_samples]

        y = (coldata[cohort_cfg["label_col"]] == cohort_cfg["positive"]).astype(int)
        print(f"  {len(y)} samples: {y.sum()} positive ({cohort_cfg['positive']}), "
              f"{len(y) - y.sum()} negative ({cohort_cfg['negative']})")

        log_expr = log2_cpm(counts)
        aligned, coverage = align_to_training_genes(log_expr, train_gene_ids, train_gene_means)
        print(f"  Gene coverage against training set: {coverage:.1%}")

        X = aligned.T.values

        for model_path in frozen_paths:
            model_name_full = model_path.split("frozen_")[-1].replace(".joblib", "")
            pipeline = joblib.load(model_path)

            try:
                y_proba = pipeline.predict_proba(X)[:, 1]
                roc_auc = roc_auc_score(y.values, y_proba)
                pr_auc = average_precision_score(y.values, y_proba)
                error = ""
            except Exception as e:
                roc_auc, pr_auc = None, None
                error = str(e)

            results.append({
                "cohort": cohort_cfg["name"], "model": model_name_full,
                "n_samples": len(y), "gene_coverage": round(coverage, 4),
                "roc_auc": roc_auc, "pr_auc": pr_auc, "error": error,
            })
            status = f"ROC-AUC={roc_auc:.3f} PR-AUC={pr_auc:.3f}" if error == "" else f"FAILED: {error}"
            print(f"    {model_name_full:30s} {status}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(results_df)} cohort x model results to {args.out}")

    print("\n=== Summary: mean ROC-AUC by strategy (averaged across models and cohorts) ===")
    results_df["strategy"] = results_df["model"].str.extract(r"^(de_informed|data_driven|pathway)_")
    summary = results_df.dropna(subset=["roc_auc"]).groupby(["strategy", "cohort"])["roc_auc"].mean().unstack()
    print(summary)


if __name__ == "__main__":
    main()
