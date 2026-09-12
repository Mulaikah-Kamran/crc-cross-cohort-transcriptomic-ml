"""
hypergeometric_enrichment.py

Phase 6: generic over-representation test (hypergeometric test) with
Benjamini-Hochberg FDR correction, run against MSigDB GMT files.

Usage
-----
    python hypergeometric_enrichment.py \
        --gene-list results/tables/stability_de_informed.csv \
        --gene-list-threshold 0.70 \
        --background data/raw/tcga_coad/tcga_coad_counts.parquet \
        --gmt data/raw/msigdb/c5_go_bp.Hs.ensembl.gmt \
        --gmt-name GO_BP \
        --out results/tables/enrichment_de_informed_GO_BP.csv
"""

import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


def strip_ensembl_version(gene_id: str) -> str:
    return re.sub(r"^(ENSG\d+)\.\d+$", r"\1", str(gene_id))


def load_gmt_file(path: str) -> dict:
    gene_sets = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            gene_sets[parts[0]] = parts[2:]
    return gene_sets


def get_background_universe(counts_path: str) -> set:
    counts = pd.read_parquet(counts_path).set_index("gene_id")
    counts.index = counts.index.map(strip_ensembl_version)
    counts = counts.groupby(counts.index).sum()
    variances = counts.var(axis=1)
    return set(variances[variances > 0].index)


def hypergeometric_enrichment(gene_set: set, background: set, pathway_gene_sets: dict,
                               min_overlap: int = 2) -> pd.DataFrame:
    gene_set = gene_set & background
    N = len(background)
    n = len(gene_set)

    rows = []
    for pathway, genes in pathway_gene_sets.items():
        pathway_in_background = set(genes) & background
        K = len(pathway_in_background)
        if K == 0:
            continue

        overlap = gene_set & pathway_in_background
        k = len(overlap)
        if k < min_overlap:
            continue

        p_value = hypergeom.sf(k - 1, N, K, n)

        rows.append({
            "pathway": pathway,
            "k_overlap": k,
            "K_pathway_size_in_background": K,
            "n_gene_set_size": n,
            "N_background_size": N,
            "overlap_genes": ";".join(sorted(overlap)),
            "p_value": p_value,
        })

    if not rows:
        return pd.DataFrame(columns=["pathway", "k_overlap", "K_pathway_size_in_background",
                                      "n_gene_set_size", "N_background_size",
                                      "overlap_genes", "p_value", "padj"])

    result = pd.DataFrame(rows)
    _, padj, _, _ = multipletests(result["p_value"], method="fdr_bh")
    result["padj"] = padj
    return result.sort_values("padj").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-list", required=True)
    parser.add_argument("--gene-list-threshold", type=float, default=0.70)
    parser.add_argument("--background", required=True)
    parser.add_argument("--gmt", required=True)
    parser.add_argument("--gmt-name", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    freq = pd.read_csv(args.gene_list, index_col=0)
    gene_set = set(freq[freq.iloc[:, 0] >= args.gene_list_threshold].index)
    print(f"Gene set of interest: {len(gene_set)} genes (>= {args.gene_list_threshold:.0%} threshold)")

    background = get_background_universe(args.background)
    print(f"Background universe: {len(background)} non-constant TCGA-COAD genes")

    pathway_gene_sets = load_gmt_file(args.gmt)
    print(f"Loaded {len(pathway_gene_sets)} {args.gmt_name} gene sets")

    result = hypergeometric_enrichment(gene_set, background, pathway_gene_sets)
    result["collection"] = args.gmt_name
    result.to_csv(args.out, index=False)

    n_significant = (result["padj"] < 0.05).sum()
    print(f"\n{len(result)} pathways tested (after skipping near-zero-overlap ones), "
          f"{n_significant} significant at padj<0.05")
    print(f"Saved to {args.out}")

    if n_significant > 0:
        print("\nTop significant pathways:")
        print(result[result["padj"] < 0.05].head(10)[["pathway", "k_overlap", "padj"]].to_string(index=False))


if __name__ == "__main__":
    main()
