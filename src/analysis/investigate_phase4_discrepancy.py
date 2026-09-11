"""
investigate_phase4_discrepancy.py

Three specific questions about the Phase 4 results, checked directly
against real data rather than guessed at.

Usage
-----
    python investigate_phase4_discrepancy.py --models-dir results/models
"""

import argparse
import glob
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src/feature_selection")


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


def get_selected_genes(pipeline, gene_ids: list, strategy: str) -> list:
    select_step = pipeline.named_steps["select"]

    if strategy == "de_informed":
        prefilter = pipeline.named_steps["prefilter"]
        prefilter_idx = prefilter.get_support(indices=True)
        selectk_idx = select_step.get_support(indices=True)
        original_idx = prefilter_idx[selectk_idx]
        return [gene_ids[i] for i in original_idx]

    if strategy == "data_driven":
        return [gene_ids[i] for i in select_step.selected_idx_]

    if strategy == "pathway":
        used = set()
        for genes in select_step.usable_pathways_.values():
            used.update(genes)
        return list(used)

    raise ValueError(strategy)


def bootstrap_auc_ci(y_true: np.ndarray, y_proba: np.ndarray, n_boot: int = 1000,
                      random_state: int = 42) -> tuple:
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        y_b, p_b = y_true[idx], y_proba[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, p_b))
    aucs = np.array(aucs)
    return aucs.mean(), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


COHORT_CONFIGS = [
    {"name": "validation2_clinical_sites",
     "counts": "data/raw/fieldeffectcrc/validation2_clinical_sites_counts.parquet",
     "coldata": "data/raw/fieldeffectcrc/validation2_clinical_sites_colData.csv",
     "label_col": "sampType", "positive": "CRC", "exclude_values": []},
    {"name": "validation3_cohortC_clean",
     "counts": "data/raw/fieldeffectcrc/validation3_cohortC_clean_counts.parquet",
     "coldata": "data/raw/fieldeffectcrc/validation3_cohortC_clean_colData.csv",
     "label_col": "sampType", "positive": "CRC", "exclude_values": ["HLT"]},
    {"name": "validation4_cohortB_clean",
     "counts": "data/raw/fieldeffectcrc/validation4_cohortB_clean_counts.parquet",
     "coldata": "data/raw/fieldeffectcrc/validation4_cohortB_clean_colData.csv",
     "label_col": "sampType", "positive": "CRC", "exclude_values": []},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default="results/models")
    args = parser.parse_args()

    print("=" * 70)
    print("QUESTION 1: Cohort C study composition vs Validation 2/4's sites")
    print("=" * 70)
    coldata_C = pd.read_csv("data/raw/fieldeffectcrc/cohortC_colData.csv")
    print("\nCohort C (Validation 3) - study x sampType:")
    print(pd.crosstab(coldata_C["study"], coldata_C["sampType"]))

    coldata_val2 = pd.read_csv("data/raw/fieldeffectcrc/validation2_clinical_sites_colData.csv")
    print("\nValidation 2 (clinical sites) - study x sampType, for comparison:")
    print(pd.crosstab(coldata_val2["study"], coldata_val2["sampType"]))

    with open(f"{args.models_dir}/training_gene_order.txt") as f:
        train_gene_ids = f.read().splitlines()
    train_gene_means = pd.read_parquet(f"{args.models_dir}/training_gene_means.parquet")["mean_log2cpm"]
    frozen_paths = sorted(glob.glob(f"{args.models_dir}/frozen_*.joblib"))

    print("\n" + "=" * 70)
    print("QUESTION 2: Real per-strategy gene coverage (not just overall 57.2%)")
    print("=" * 70)

    for cohort_cfg in COHORT_CONFIGS:
        counts = pd.read_parquet(cohort_cfg["counts"])
        gene_col = "gene_id" if "gene_id" in counts.columns else counts.columns[0]
        counts = counts.set_index(gene_col)
        counts.index = counts.index.map(strip_ensembl_version)
        counts = counts.groupby(counts.index).sum()
        cohort_genes_available = set(counts.index)

        print(f"\n{cohort_cfg['name']}:")
        for model_path in frozen_paths:
            model_name = model_path.split("frozen_")[-1].replace(".joblib", "")
            strategy = re.match(r"(de_informed|data_driven|pathway)_", model_name).group(1)
            pipeline = joblib.load(model_path)
            selected = get_selected_genes(pipeline, train_gene_ids, strategy)
            n_present = sum(1 for g in selected if g in cohort_genes_available)
            real_coverage = n_present / len(selected)
            print(f"  {model_name:30s} {n_present}/{len(selected)} selected genes present "
                  f"({real_coverage:.1%} real coverage, not imputed)")

    print("\n" + "=" * 70)
    print("QUESTION 3: Bootstrap 95% CIs - how much is small-sample noise?")
    print("=" * 70)

    for cohort_cfg in COHORT_CONFIGS:
        counts = pd.read_parquet(cohort_cfg["counts"])
        gene_col = "gene_id" if "gene_id" in counts.columns else counts.columns[0]
        counts = counts.set_index(gene_col)
        counts.index = counts.index.map(strip_ensembl_version)
        counts = counts.groupby(counts.index).sum()

        coldata = pd.read_csv(cohort_cfg["coldata"])
        sample_id_col = "sample_id" if "sample_id" in coldata.columns else coldata.columns[0]
        coldata = coldata.set_index(sample_id_col)
        coldata = coldata[~coldata[cohort_cfg["label_col"]].isin(cohort_cfg["exclude_values"])]

        common = coldata.index.intersection(counts.columns)
        counts, coldata = counts[common], coldata.loc[common]
        y = (coldata[cohort_cfg["label_col"]] == cohort_cfg["positive"]).astype(int).values

        log_expr = log2_cpm(counts)
        aligned, _ = align_to_training_genes(log_expr, train_gene_ids, train_gene_means)
        X = aligned.T.values

        print(f"\n{cohort_cfg['name']} (N={len(y)}):")
        for model_path in frozen_paths:
            model_name = model_path.split("frozen_")[-1].replace(".joblib", "")
            pipeline = joblib.load(model_path)
            y_proba = pipeline.predict_proba(X)[:, 1]
            point_auc = roc_auc_score(y, y_proba)
            mean_boot, lo, hi = bootstrap_auc_ci(y, y_proba)
            print(f"  {model_name:30s} point={point_auc:.3f}  "
                  f"bootstrap 95% CI=[{lo:.3f}, {hi:.3f}]  (width={hi-lo:.3f})")


if __name__ == "__main__":
    main()
