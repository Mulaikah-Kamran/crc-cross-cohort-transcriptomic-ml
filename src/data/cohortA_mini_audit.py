"""
cohortA_mini_audit.py

Purpose
-------
Cohort A (FieldEffectCrc, n=834) is itself pooled from multiple original
sources (GDC / SRA-via-dbGaP / SRA-public / BarcUVa-Seq, per its own
metadata). Before trusting internal nested-CV performance on Cohort A, we
must check whether samples cluster by originating sub-study/site rather
than by biology (tumor/HLT/NAT) — i.e. rule out a smaller version of the
exact cohort-confounding problem from P2.1.

This is a QC gate, not a modeling step: no feature selection or scaling
parameters computed here are reused downstream.

What it does
------------
1. Loads Cohort A's counts matrix + colData (metadata).
2. log2(CPM + 1) transforms counts (standard, leakage-free QC-only step —
   this transform is NOT the one used inside the actual nested CV pipeline).
3. Runs PCA (top 2 PCs) and UMAP on the top N most-variable genes.
4. Produces two side-by-side-style plots for each embedding: colored by
   biological label (tumor/HLT/NAT), and colored by originating
   sub-study/site metadata.
5. Prints a simple diagnostic: whether PC1/PC2 (or UMAP dims) correlate
   more strongly with the biological label or with the sub-study/site label
   (via an R^2 from a one-way ANOVA-style fit) as a numeric sanity check
   alongside the visual inspection.

Usage
-----
    python cohortA_mini_audit.py \
        --counts data/raw/fieldeffectcrc/cohortA_counts.parquet \
        --coldata data/raw/fieldeffectcrc/cohortA_colData.csv \
        --study-col study \
        --biology-col sampType \
        --out-dir results/figures/cohortA_audit

NOTE ON COLUMN NAMES: `--study-col` / `--biology-col` default to the names
documented in the FieldEffectCrc manual ('study', 'sampType') but MUST be
verified against the actual colData.csv column names after extraction —
this was flagged explicitly in 01_extract_fieldeffectcrc.R.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def log2_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    """genes x samples counts -> genes x samples log2(CPM+1). QC-only transform."""
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    return np.log2(cpm + 1)


def top_variable_genes(log_expr: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    variances = log_expr.var(axis=1)
    top_genes = variances.sort_values(ascending=False).head(n).index
    return log_expr.loc[top_genes]


def label_association_strength(embedding: np.ndarray, labels: pd.Series) -> float:
    """
    Quick numeric proxy for 'how much does this categorical variable explain
    the embedding coordinates' — mean R^2 across embedding dimensions from a
    one-way ANOVA (group means vs. grand mean). Not a formal test, just a
    fast triage signal to accompany the plots.
    """
    labels = labels.astype(str).fillna("NA")
    r2s = []
    for dim in range(embedding.shape[1]):
        y = embedding[:, dim]
        grand_mean = y.mean()
        ss_tot = ((y - grand_mean) ** 2).sum()
        ss_between = 0.0
        for lvl in labels.unique():
            mask = (labels == lvl).values
            if mask.sum() < 2:
                continue
            group_mean = y[mask].mean()
            ss_between += mask.sum() * (group_mean - grand_mean) ** 2
        r2 = ss_between / ss_tot if ss_tot > 0 else 0.0
        r2s.append(r2)
    return float(np.mean(r2s))


def plot_embedding(embedding: np.ndarray, labels: pd.Series, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = labels.astype(str).fillna("NA")
    for lvl in sorted(labels.unique()):
        mask = (labels == lvl).values
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=12, alpha=0.7, label=lvl)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(fontsize=7, markerscale=1.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--coldata", required=True)
    parser.add_argument("--study-col", default="study")
    parser.add_argument("--biology-col", default="sampType")
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--out-dir", default="results/figures/cohortA_audit")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    counts = pd.read_parquet(args.counts).set_index("gene_id")
    coldata = pd.read_csv(args.coldata).set_index("sample_id")
    counts = counts[coldata.index.intersection(counts.columns)]  # align sample order

    log_expr = log2_cpm(counts)
    top_expr = top_variable_genes(log_expr, n=args.n_top_genes)

    X = StandardScaler().fit_transform(top_expr.T.values)  # samples x genes

    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X)
    print(f"PCA explained variance ratio (PC1, PC2): {pca.explained_variance_ratio_}")

    for label_col, label_name in [(args.biology_col, "biology"), (args.study_col, "substudy")]:
        if label_col not in coldata.columns:
            print(f"[!] Column '{label_col}' not found in colData — check actual column names.")
            continue
        r2 = label_association_strength(pcs, coldata[label_col])
        print(f"PCA vs {label_name} ('{label_col}'): mean R^2 across PC1/PC2 = {r2:.3f}")
        plot_embedding(pcs, coldata[label_col], f"Cohort A — PCA colored by {label_name}",
                        os.path.join(args.out_dir, f"pca_by_{label_name}.png"))

    if HAS_UMAP:
        reducer = umap.UMAP(random_state=0)
        emb = reducer.fit_transform(X)
        for label_col, label_name in [(args.biology_col, "biology"), (args.study_col, "substudy")]:
            if label_col not in coldata.columns:
                continue
            r2 = label_association_strength(emb, coldata[label_col])
            print(f"UMAP vs {label_name} ('{label_col}'): mean R^2 across dims = {r2:.3f}")
            plot_embedding(emb, coldata[label_col], f"Cohort A — UMAP colored by {label_name}",
                            os.path.join(args.out_dir, f"umap_by_{label_name}.png"))
    else:
        print("[!] umap-learn not installed — skipping UMAP (PCA results above are still valid).")

    print(
        "\nHOW TO READ THIS: if the R^2 (and visual clustering) for 'substudy' is "
        "comparable to or higher than for 'biology', that is the P2.1-style red "
        "flag — samples are separating by data source, not tumor/normal status. "
        "In that case, STOP before Phase 2 and revisit the discovery-cohort "
        "choice per the Go/No-Go condition in the locked spec."
    )


if __name__ == "__main__":
    main()
