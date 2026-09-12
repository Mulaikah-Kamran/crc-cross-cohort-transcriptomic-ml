"""
tissue_composition_check.py

Quantifies whether the stable consensus gene sets overlap curated
immune/stromal/epithelial marker genes, per the design spec's Section 21
tissue-composition requirement.

Usage
-----
    python tissue_composition_check.py \
        --stability-de results/tables/stability_de_informed.csv \
        --stability-dd results/tables/stability_data_driven.csv \
        --background data/raw/tcga_coad/tcga_coad_counts.parquet \
        --marker-gmt data/raw/markers/tissue_composition_markers.gmt \
        --out results/tables/phase6_tissue_composition_check.csv
"""

import argparse
import sys

import pandas as pd

sys.path.insert(0, "src/analysis")
from hypergeometric_enrichment import (
    load_gmt_file, get_background_universe, hypergeometric_enrichment
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stability-de", required=True)
    parser.add_argument("--stability-dd", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--marker-gmt", required=True)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--out", default="results/tables/phase6_tissue_composition_check.csv")
    args = parser.parse_args()

    background = get_background_universe(args.background)
    marker_sets = load_gmt_file(args.marker_gmt)
    print(f"Loaded {len(marker_sets)} marker categories: {list(marker_sets.keys())}")
    for name, genes in marker_sets.items():
        in_bg = set(genes) & background
        print(f"  {name}: {len(genes)} curated genes, {len(in_bg)} present in the background universe")

    rows = []
    for strategy, path in [("de_informed", args.stability_de), ("data_driven", args.stability_dd)]:
        freq = pd.read_csv(path, index_col=0)
        stable_genes = set(freq[freq.iloc[:, 0] >= args.threshold].index)
        n_stable = len(stable_genes)
        print(f"\n=== {strategy}: {n_stable} stable genes ===")

        result = hypergeometric_enrichment(stable_genes, background, marker_sets, min_overlap=1)

        for _, r in result.iterrows():
            pct_of_stable = r["k_overlap"] / n_stable
            print(f"  {r['pathway']:30s} {r['k_overlap']}/{n_stable} stable genes "
                  f"({pct_of_stable:.1%}), padj={r['padj']:.4f}")
            rows.append({
                "strategy": strategy,
                "marker_category": r["pathway"],
                "n_stable_genes": n_stable,
                "n_overlap": r["k_overlap"],
                "pct_of_stable_genes": round(pct_of_stable, 4),
                "hypergeometric_padj": r["padj"],
                "overlap_genes": r["overlap_genes"],
            })

        tested_categories = set(result["pathway"]) if len(result) > 0 else set()
        for cat in marker_sets:
            if cat not in tested_categories:
                print(f"  {cat:30s} 0/{n_stable} stable genes (0.0%) - no overlap at all")
                rows.append({
                    "strategy": strategy, "marker_category": cat,
                    "n_stable_genes": n_stable, "n_overlap": 0,
                    "pct_of_stable_genes": 0.0, "hypergeometric_padj": None,
                    "overlap_genes": "",
                })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")

    print("\nREMINDER: any significant overlap here means the stable gene set's predictive "
          "signal is at least partly associated with tissue/cellular composition, not that "
          "the classifier discovered tumor-cell-intrinsic biomarkers for that lineage.")


if __name__ == "__main__":
    main()
