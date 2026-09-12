"""
top_biomarkers_table.py

Pulls the top 10 genes per strategy (by |Elastic Net coefficient|) and
computes each gene's ACTUAL log2 fold change (tumor vs normal, in
TCGA-COAD) directly from the data, cross-checked against the model
coefficient's implied direction.

Usage
-----
    python top_biomarkers_table.py \
        --weights results/tables/phase5_model_feature_weights.csv \
        --tcga-counts data/raw/tcga_coad/tcga_coad_counts.parquet \
        --tcga-coldata data/raw/tcga_coad/tcga_coad_colData.parquet \
        --top-n 10 \
        --out results/tables/top_biomarkers_ensembl.csv
"""

import argparse
import re

import numpy as np
import pandas as pd


def log2_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    return np.log2(cpm + 1)


def strip_ensembl_version(gene_id: str) -> str:
    return re.sub(r"^(ENSG\d+)\.\d+$", r"\1", str(gene_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--tcga-counts", required=True)
    parser.add_argument("--tcga-coldata", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--out", default="results/tables/top_biomarkers_ensembl.csv")
    args = parser.parse_args()

    weights = pd.read_csv(args.weights)

    counts = pd.read_parquet(args.tcga_counts).set_index("gene_id")
    coldata = pd.read_parquet(args.tcga_coldata).set_index("sample_id")
    counts = counts[coldata.index.intersection(counts.columns)]
    counts.index = counts.index.map(strip_ensembl_version)
    counts = counts.groupby(counts.index).sum()

    y = (coldata.loc[counts.columns, "sample_type"].to_numpy(dtype=object) == "Primary Tumor")
    log_expr = log2_cpm(counts)

    rows = []
    for strategy in weights["strategy"].unique():
        sub = weights[weights["strategy"] == strategy].copy()
        sub["abs_coef"] = sub["elastic_net_coefficient"].abs()
        top = sub.sort_values("abs_coef", ascending=False).head(args.top_n)

        for _, r in top.iterrows():
            gene = r["gene_id"]
            if gene not in log_expr.index:
                log2fc = None
            else:
                tumor_mean = log_expr.loc[gene, y].mean()
                normal_mean = log_expr.loc[gene, ~y].mean()
                log2fc = tumor_mean - normal_mean

            direction_from_data = None
            direction_from_coef = None
            if log2fc is not None:
                direction_from_data = "UP in tumor" if log2fc > 0 else "DOWN in tumor"
            if pd.notna(r["elastic_net_coefficient"]):
                direction_from_coef = "UP in tumor" if r["elastic_net_coefficient"] > 0 else "DOWN in tumor"

            agree = (direction_from_data == direction_from_coef) if (direction_from_data and direction_from_coef) else None

            rows.append({
                "strategy": strategy,
                "ensembl_id": gene,
                "elastic_net_coefficient": r["elastic_net_coefficient"],
                "random_forest_gini_importance": r["random_forest_gini_importance"],
                "log2fc_tcga_coad": round(log2fc, 3) if log2fc is not None else None,
                "direction_from_actual_data": direction_from_data,
                "direction_from_model_coefficient": direction_from_coef,
                "directions_agree": agree,
            })

    result = pd.DataFrame(rows)
    result.to_csv(args.out, index=False)

    n_disagree = (result["directions_agree"] == False).sum()
    print(f"Saved {len(result)} genes ({args.top_n} per strategy) to {args.out}")
    if n_disagree > 0:
        print(f"\n[!] {n_disagree} gene(s) show a MISMATCH between the model coefficient's "
              f"implied direction and the actual measured log2FC direction.")
    else:
        print("\nAll genes: model coefficient direction agrees with actual measured log2FC direction.")

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
