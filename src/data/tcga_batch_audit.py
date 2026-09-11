"""
tcga_batch_audit.py

Purpose
-------
TCGA-COAD is a single harmonized GDC pipeline, not a pool of five outside
studies the way FieldEffectCrc Cohort A was, so this isn't expected to find
anything close to that scale of confound. But TCGA is itself assembled from
many collection sites and processed across many sequencing plates over
several years, so it's worth the same kind of check before trusting the
discovery cohort's internal CV splits, rather than assuming a "harmonized"
label means no structure exists.

TCGA sample barcodes encode this directly. A full aliquot barcode looks like:

    TCGA-A6-2684-01A-01R-0826-07
     │    │   │    │    │   │   │
     │    │   │    │    │   │   └─ sequencing/plate center
     │    │   │    │    │   └───── plate
     │    │   │    │    └───────── portion/analyte
     │    │   │    └────────────── sample type + vial
     │    │   └─────────────────── participant
     │    └─────────────────────── Tissue Source Site (TSS) - the collecting institution
     └──────────────────────────── project

This script parses TSS and plate out of the barcode, then runs the same
PCA/UMAP diagnostic used on Cohort A: does biology (tumor vs. tumor-adjacent
normal) or a technical variable (TSS, plate) explain more of the variance?

Usage
-----
    python tcga_batch_audit.py \
        --counts data/raw/tcga_coad/tcga_coad_counts.parquet \
        --coldata data/raw/tcga_coad/tcga_coad_colData.parquet \
        --barcode-col barcode \
        --biology-col sample_type \
        --out-dir results/figures/tcga_batch_audit
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


def parse_tcga_barcode(barcode: str):
    """TCGA-A6-2684-01A-01R-0826-07 -> ('A6', '0826'). Returns (None, None)
    if the barcode doesn't have enough segments to parse safely."""
    parts = str(barcode).split("-")
    if len(parts) < 6:
        return None, None
    tss = parts[1]
    plate = parts[5]
    return tss, plate


def log2_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    return np.log2(cpm + 1)


def top_variable_genes(log_expr: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    variances = log_expr.var(axis=1)
    top_genes = variances.sort_values(ascending=False).head(n).index
    return log_expr.loc[top_genes]


def label_association_strength(embedding: np.ndarray, labels: pd.Series) -> float:
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


def plot_embedding(embedding: np.ndarray, labels: pd.Series, title: str, out_path: str,
                    legend: bool = True):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = labels.astype(str).fillna("NA")
    n_levels = labels.nunique()
    for lvl in sorted(labels.unique()):
        mask = (labels == lvl).values
        # TSS/plate can have dozens of levels - skip the legend in that case,
        # it becomes unreadable and isn't the point of the plot anyway
        label_kw = {"label": lvl} if legend and n_levels <= 15 else {}
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=10, alpha=0.6, **label_kw)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    if legend and n_levels <= 15:
        ax.legend(fontsize=7, markerscale=1.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--coldata", required=True)
    parser.add_argument("--barcode-col", default="barcode")
    parser.add_argument("--biology-col", default="sample_type")
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--out-dir", default="results/figures/tcga_batch_audit")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    counts = pd.read_parquet(args.counts).set_index("gene_id")
    coldata = pd.read_parquet(args.coldata).set_index("sample_id")
    counts = counts[coldata.index.intersection(counts.columns)]

    if args.barcode_col not in coldata.columns:
        print(f"[!] Column '{args.barcode_col}' not found. Available columns "
              f"containing 'barcode': "
              f"{[c for c in coldata.columns if 'barcode' in c.lower()]}")
        return

    tss_list, plate_list = [], []
    for bc in coldata[args.barcode_col]:
        tss, plate = parse_tcga_barcode(bc)
        tss_list.append(tss)
        plate_list.append(plate)
    coldata["tss"] = tss_list
    coldata["plate"] = plate_list

    n_unparsed = coldata["tss"].isna().sum()
    if n_unparsed > 0:
        print(f"[!] {n_unparsed} barcode(s) could not be parsed and will show "
              f"as 'NA' in the plots - check the barcode format if this is "
              f"more than a handful.")

    print(f"Unique TSS (collection sites): {coldata['tss'].nunique()}")
    print(f"Unique plates: {coldata['plate'].nunique()}")
    print()

    log_expr = log2_cpm(counts)
    top_expr = top_variable_genes(log_expr, n=args.n_top_genes)
    X = StandardScaler().fit_transform(top_expr.T.values)

    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X)
    print(f"PCA explained variance ratio (PC1, PC2): {pca.explained_variance_ratio_}")

    for label_col, label_name in [(args.biology_col, "biology"), ("tss", "TSS"), ("plate", "plate")]:
        if label_col not in coldata.columns:
            print(f"[!] Column '{label_col}' not found - skipping.")
            continue
        r2 = label_association_strength(pcs, coldata[label_col])
        print(f"PCA vs {label_name} ('{label_col}'): mean R^2 across PC1/PC2 = {r2:.3f}")
        plot_embedding(pcs, coldata[label_col], f"TCGA-COAD — PCA colored by {label_name}",
                        os.path.join(args.out_dir, f"pca_by_{label_name}.png"))

    if HAS_UMAP:
        reducer = umap.UMAP(random_state=0)
        emb = reducer.fit_transform(X)
        for label_col, label_name in [(args.biology_col, "biology"), ("tss", "TSS"), ("plate", "plate")]:
            if label_col not in coldata.columns:
                continue
            r2 = label_association_strength(emb, coldata[label_col])
            print(f"UMAP vs {label_name} ('{label_col}'): mean R^2 across dims = {r2:.3f}")
            plot_embedding(emb, coldata[label_col], f"TCGA-COAD — UMAP colored by {label_name}",
                            os.path.join(args.out_dir, f"umap_by_{label_name}.png"))
    else:
        print("[!] umap-learn not installed - skipping UMAP.")

    print(
        "\nHOW TO READ THIS: TCGA-COAD is expected to show some structure by "
        "TSS or plate, since it's assembled from many collection sites over "
        "several years - some non-zero R^2 for these is normal and doesn't "
        "invalidate the cohort. The threshold that matters is the same as "
        "before: if TSS or plate R^2 rivals or exceeds the biology R^2, that's "
        "a real problem worth stopping for, not a curiosity. A biology R^2 "
        "that clearly leads is what clears this cohort for Phase 2."
    )


if __name__ == "__main__":
    main()
