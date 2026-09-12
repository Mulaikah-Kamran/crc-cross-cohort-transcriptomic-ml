"""
figure3_biomarkers.py

Figure 3: top stable genes per strategy, ranked by Elastic Net
coefficient magnitude, colored by direction, with the four genes that
mark the BEST4+ colonocyte lineage (OTOP2, BEST4, CA7, GUCA2A) called
out distinctly.

Usage
-----
    python figure3_biomarkers.py \
        --biomarkers results/tables/top_biomarkers_symbols_merged.csv \
        --out results/figures/final/figure3_biomarkers.png
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

BEST4_LINEAGE_GENES = {"OTOP2", "BEST4", "CA7", "GUCA2A"}
STRATEGY_LABELS = {"de_informed": "DE-informed", "data_driven": "Data-driven"}


def plot_panel(ax, df, strategy):
    sub = df[df["strategy"] == strategy].copy()
    sub = sub.reindex(sub["elastic_net_coefficient"].abs().sort_values().index)

    colors = []
    edge_colors = []
    for _, row in sub.iterrows():
        base_color = "#d62728" if row["elastic_net_coefficient"] > 0 else "#1f77b4"
        colors.append(base_color)
        edge_colors.append("gold" if row["gene_symbol"] in BEST4_LINEAGE_GENES else "none")

    bars = ax.barh(sub["gene_symbol"], sub["elastic_net_coefficient"],
                     color=colors, edgecolor=edge_colors, linewidth=2.5)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(STRATEGY_LABELS[strategy], fontsize=12, fontweight="bold")
    ax.set_xlabel("Elastic Net Coefficient\n(negative = down in tumor, positive = up in tumor)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biomarkers", required=True)
    parser.add_argument("--out", default="results/figures/final/figure3_biomarkers.png")
    args = parser.parse_args()

    df = pd.read_csv(args.biomarkers)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plot_panel(axes[0], df, "de_informed")
    plot_panel(axes[1], df, "data_driven")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="Up in tumor"),
        Patch(facecolor="#1f77b4", label="Down in tumor"),
        Patch(facecolor="white", edgecolor="gold", linewidth=2.5, label="BEST4+ colonocyte marker"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.06))

    fig.suptitle("Top 10 Stable Biomarker Genes per Strategy", fontsize=13, fontweight="bold", y=1.12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
