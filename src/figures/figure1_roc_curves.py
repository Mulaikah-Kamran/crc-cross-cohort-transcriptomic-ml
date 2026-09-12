"""
figure1_roc_curves.py

Figure 1: ROC curves comparing the three feature-selection strategies on
Validation 1 and Validation 3, using Elastic Net as the representative
classifier per strategy.

Usage
-----
    python figure1_roc_curves.py \
        --models-dir results/models \
        --out results/figures/final/figure1_roc_curves.png
"""

import argparse
import re
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

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
    for gene in aligned.index[aligned.isna().all(axis=1)]:
        aligned.loc[gene] = train_gene_means.get(gene, 0.0)
    return aligned


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


STRATEGY_COLORS = {"de_informed": "#d62728", "data_driven": "#1f77b4", "pathway": "#2ca02c"}
STRATEGY_LABELS = {"de_informed": "DE-informed", "data_driven": "Data-driven", "pathway": "Pathway"}


def plot_panel(ax, title, train_gene_ids, train_gene_means, models_dir, counts, y):
    log_expr = log2_cpm(counts)
    aligned = align_to_training_genes(log_expr, train_gene_ids, train_gene_means)
    X = aligned.T.values

    for strategy in ["de_informed", "data_driven", "pathway"]:
        pipeline = joblib.load(f"{models_dir}/frozen_{strategy}_elastic_net.joblib")
        y_proba = pipeline.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        ax.plot(fpr, tpr, color=STRATEGY_COLORS[strategy], linewidth=2,
                 label=f"{STRATEGY_LABELS[strategy]} (AUC = {auc:.2f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default="results/models")
    parser.add_argument("--out", default="results/figures/final/figure1_roc_curves.png")
    args = parser.parse_args()

    with open(f"{args.models_dir}/training_gene_order.txt") as f:
        train_gene_ids = f.read().splitlines()
    train_gene_means = pd.read_parquet(f"{args.models_dir}/training_gene_means.parquet")["mean_log2cpm"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    counts1, y1 = load_cohort(
        "data/raw/geo_eocrc_locrc/pooled_counts.parquet",
        "data/raw/geo_eocrc_locrc/pooled_metadata.csv",
        "tissue_label", "Tumor",
    )
    plot_panel(axes[0], "Validation 1: Independent Lab (GEO)", train_gene_ids, train_gene_means,
               args.models_dir, counts1, y1)

    counts3, y3 = load_cohort(
        "data/raw/fieldeffectcrc/validation3_cohortC_clean_counts.parquet",
        "data/raw/fieldeffectcrc/validation3_cohortC_clean_colData.csv",
        "sampType", "CRC", exclude_values=["HLT"],
    )
    plot_panel(axes[1], "Validation 3: Cross-Protocol Cohort (Cleaned)", train_gene_ids, train_gene_means,
               args.models_dir, counts3, y3)

    fig.suptitle("External Validation: ROC Curves by Feature-Selection Strategy",
                  fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
