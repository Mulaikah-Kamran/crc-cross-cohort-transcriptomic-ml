"""
stability_crosscheck.py

Secondary cross-check: does the stable DE-informed/data-driven consensus
gene set (from feature_stability.py) show consistent direction and
retain explanatory power in Validation 1 (GEO pooled) and Validation 3
(Cohort C, cleaned)?

Usage
-----
    python stability_crosscheck.py \
        --tcga-counts data/raw/tcga_coad/tcga_coad_counts.parquet \
        --tcga-coldata data/raw/tcga_coad/tcga_coad_colData.parquet \
        --stability-dir results/tables \
        --out results/tables/phase5_crosscheck_results.csv
"""

import argparse
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def log2_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    return np.log2(cpm + 1)


def strip_ensembl_version(gene_id: str) -> str:
    return re.sub(r"^(ENSG\d+)\.\d+$", r"\1", str(gene_id))


def compute_log2fc(log_expr: pd.DataFrame, y: pd.Series, stable_genes: list) -> pd.Series:
    available = [g for g in stable_genes if g in log_expr.index]
    pos = log_expr.loc[available, y == 1].mean(axis=1)
    neg = log_expr.loc[available, y == 0].mean(axis=1)
    return pos - neg


def load_cohort(counts_path, coldata_path, label_col, positive, exclude_values=None):
    exclude_values = exclude_values or []
    counts = pd.read_parquet(counts_path)
    gene_col = "gene_id" if "gene_id" in counts.columns else counts.columns[0]
    counts = counts.set_index(gene_col)
    counts.index = counts.index.map(strip_ensembl_version)
    counts = counts.groupby(counts.index).sum()

    coldata = pd.read_csv(coldata_path) if coldata_path.endswith(".csv") else pd.read_parquet(coldata_path)
    sample_id_col = "sample_id" if "sample_id" in coldata.columns else coldata.columns[0]
    coldata = coldata.set_index(sample_id_col)
    coldata = coldata[~coldata[label_col].isin(exclude_values)]

    common = coldata.index.intersection(counts.columns)
    counts, coldata = counts[common], coldata.loc[common]
    y = (coldata[label_col] == positive).astype(int)
    return counts, y


def evaluate_gene_subset(train_counts, train_y, test_counts, test_y, genes: list):
    train_log = log2_cpm(train_counts)
    test_log = log2_cpm(test_counts)

    available = [g for g in genes if g in train_log.index and g in test_log.index]
    if len(available) < 5:
        return None, len(available)

    X_train = train_log.loc[available].T.values
    X_test = test_log.loc[available].T.values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train_s, train_y.values)
    y_proba = clf.predict_proba(X_test_s)[:, 1]
    return roc_auc_score(test_y.values, y_proba), len(available)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcga-counts", required=True)
    parser.add_argument("--tcga-coldata", required=True)
    parser.add_argument("--stability-dir", default="results/tables")
    parser.add_argument("--out", default="results/tables/phase5_crosscheck_results.csv")
    args = parser.parse_args()

    tcga_counts, tcga_y = load_cohort(
        args.tcga_counts, args.tcga_coldata, "sample_type", "Primary Tumor"
    )
    tcga_log = log2_cpm(tcga_counts)
    print(f"TCGA-COAD (discovery): {tcga_counts.shape[1]} samples")

    val_cohorts = {
        "validation1_geo_pooled": dict(
            counts="data/raw/geo_eocrc_locrc/pooled_counts.parquet",
            coldata="data/raw/geo_eocrc_locrc/pooled_metadata.csv",
            label_col="tissue_label", positive="Tumor",
        ),
        "validation3_cohortC_clean": dict(
            counts="data/raw/fieldeffectcrc/validation3_cohortC_clean_counts.parquet",
            coldata="data/raw/fieldeffectcrc/validation3_cohortC_clean_colData.csv",
            label_col="sampType", positive="CRC", exclude_values=["HLT"],
        ),
    }

    results = []
    for strategy_name, stability_file in [
        ("de_informed", f"{args.stability_dir}/stability_de_informed.csv"),
        ("data_driven", f"{args.stability_dir}/stability_data_driven.csv"),
    ]:
        freq = pd.read_csv(stability_file, index_col=0)
        stable_genes = freq[freq.iloc[:, 0] >= 0.70].index.tolist()
        print(f"\n=== {strategy_name}: {len(stable_genes)} stable genes ===")

        tcga_log2fc = compute_log2fc(tcga_log, tcga_y, stable_genes)

        for val_name, cfg in val_cohorts.items():
            val_counts, val_y = load_cohort(
                cfg["counts"], cfg["coldata"], cfg["label_col"], cfg["positive"],
                cfg.get("exclude_values"),
            )
            val_log = log2_cpm(val_counts)
            val_log2fc = compute_log2fc(val_log, val_y, stable_genes)

            common_genes = tcga_log2fc.index.intersection(val_log2fc.index)
            concordant = (np.sign(tcga_log2fc.loc[common_genes]) ==
                          np.sign(val_log2fc.loc[common_genes])).sum()
            concordance_pct = concordant / len(common_genes) if len(common_genes) > 0 else None

            auc, n_genes_used = evaluate_gene_subset(
                tcga_counts, tcga_y, val_counts, val_y, stable_genes
            )

            print(f"  vs {val_name}: {len(common_genes)} genes comparable, "
                  f"{concordant} concordant direction ({concordance_pct:.1%}), "
                  f"stable-gene-only ROC-AUC = {auc if auc else 'N/A'} (using {n_genes_used} genes)")

            results.append({
                "strategy": strategy_name, "validation_cohort": val_name,
                "n_stable_genes": len(stable_genes),
                "n_genes_comparable": len(common_genes),
                "n_concordant_direction": int(concordant),
                "concordance_pct": round(concordance_pct, 4) if concordance_pct is not None else None,
                "stable_gene_subset_auc": round(auc, 4) if auc is not None else None,
                "n_genes_in_auc_model": n_genes_used,
            })

    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
