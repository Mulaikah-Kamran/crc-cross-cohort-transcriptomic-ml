"""
figure2_field_effect.py

Figure 2: distribution of predicted tumor-probability scores across
Healthy, Tumor-Adjacent-Normal, and Tumor tissue, using Elastic Net per
strategy.

Usage
-----
    python figure2_field_effect.py \
        --cohortA-counts data/raw/fieldeffectcrc/cohortA_counts.parquet \
        --cohortA-coldata data/raw/fieldeffectcrc/cohortA_colData.csv \
        --models-dir results/models \
        --out results/figures/final/figure2_field_effect.png
"""

import argparse
import re
import sys

import joblib
import matplotlib.pyplot as plt
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
    for gene in aligned.index[aligned.isna().all(axis=1)]:
        aligned.loc[gene] = train_gene_means.get(gene, 0.0)
    return aligned


STRATEGY_LABELS = {"de_informed": "DE-informed", "data_driven": "Data-driven", "pathway": "Pathway"}
GROUP_ORDER = ["HLT", "NAT", "CRC"]
GROUP_LABELS = {"HLT": "Healthy", "NAT": "Adjacent-\nNormal", "CRC": "Tumor"}
GROUP_COLORS = {"HLT": "#2ca02c", "NAT": "#ff7f0e", "CRC": "#d62728"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohortA-counts", required=True)
    parser.add_argument("--cohortA-coldata", required=True)
    parser.add_argument("--models-dir", default="results/models")
    parser.add_argument("--out", default="results/figures/final/figure2_field_effect.png")
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

    log_expr = log2_cpm(counts)
    aligned = align_to_training_genes(log_expr, train_gene_ids, train_gene_means)
    X = aligned.T.values

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    for ax, strategy in zip(axes, ["de_informed", "data_driven", "pathway"]):
        pipeline = joblib.load(f"{args.models_dir}/frozen_{strategy}_elastic_net.joblib")
        scores = pd.Series(pipeline.predict_proba(X)[:, 1], index=coldata.index)

        data = [scores[coldata["sampType"] == g].values for g in GROUP_ORDER]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        ax.set_xticks(range(1, len(GROUP_ORDER) + 1))
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
        for patch, g in zip(bp["boxes"], GROUP_ORDER):
            patch.set_facecolor(GROUP_COLORS[g])
            patch.set_alpha(0.7)

        for i, g in enumerate(GROUP_ORDER):
            y = scores[coldata["sampType"] == g].values
            x = np.random.RandomState(0).normal(i + 1, 0.05, size=len(y))
            ax.scatter(x, y, s=6, color="black", alpha=0.25, zorder=3)

        ax.set_title(STRATEGY_LABELS[strategy], fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        if ax is axes[0]:
            ax.set_ylabel("Predicted Tumor Probability")

    fig.suptitle("Field Effect: Predicted Tumor Probability Across Tissue States",
                  fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
