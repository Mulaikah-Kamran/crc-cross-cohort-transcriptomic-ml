"""
extract_feature_weights.py

Pulls per-gene weights from the already-frozen Elastic Net and Random
Forest models (no retraining), restricted to each strategy's stable
consensus gene set from Phase 5's bootstrap analysis.

Usage
-----
    python extract_feature_weights.py \
        --models-dir results/models \
        --stability-dir results/tables \
        --out results/tables/phase5_model_feature_weights.csv
"""

import argparse
import re
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src/feature_selection")
import sklearn_transformers  # noqa: F401


def get_selected_genes_and_order(pipeline, gene_ids: list, strategy: str):
    select_step = pipeline.named_steps["select"]

    if strategy == "de_informed":
        prefilter = pipeline.named_steps["prefilter"]
        prefilter_idx = prefilter.get_support(indices=True)
        selectk_idx = select_step.get_support(indices=True)
        original_idx = prefilter_idx[selectk_idx]
        return [gene_ids[i] for i in original_idx]

    if strategy == "data_driven":
        return [gene_ids[i] for i in select_step.selected_idx_]

    raise ValueError(f"Unsupported strategy for gene-level weights: {strategy}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default="results/models")
    parser.add_argument("--stability-dir", default="results/tables")
    parser.add_argument("--out", default="results/tables/phase5_model_feature_weights.csv")
    args = parser.parse_args()

    with open(f"{args.models_dir}/training_gene_order.txt") as f:
        train_gene_ids = f.read().splitlines()

    rows = []
    for strategy in ["de_informed", "data_driven"]:
        stability_freq = pd.read_csv(f"{args.stability_dir}/stability_{strategy}.csv", index_col=0)
        stable_genes = set(stability_freq[stability_freq.iloc[:, 0] >= 0.70].index)
        print(f"{strategy}: {len(stable_genes)} stable genes to look up")

        en_pipeline = joblib.load(f"{args.models_dir}/frozen_{strategy}_elastic_net.joblib")
        en_genes = get_selected_genes_and_order(en_pipeline, train_gene_ids, strategy)
        en_coefs = en_pipeline.named_steps["clf"].coef_[0]
        en_weights = dict(zip(en_genes, en_coefs))

        rf_pipeline = joblib.load(f"{args.models_dir}/frozen_{strategy}_random_forest.joblib")
        rf_genes = get_selected_genes_and_order(rf_pipeline, train_gene_ids, strategy)
        rf_importances = rf_pipeline.named_steps["clf"].feature_importances_
        rf_weights = dict(zip(rf_genes, rf_importances))

        for gene in stable_genes:
            rows.append({
                "strategy": strategy,
                "gene_id": gene,
                "elastic_net_coefficient": en_weights.get(gene),
                "random_forest_gini_importance": rf_weights.get(gene),
            })

    result = pd.DataFrame(rows)
    result["_sort_key"] = result["elastic_net_coefficient"].abs()
    result = result.sort_values(["strategy", "_sort_key"], ascending=[True, False]).drop(columns="_sort_key")

    result.to_csv(args.out, index=False)
    print(f"\nSaved {len(result)} gene-level weight rows to {args.out}")

    n_missing = result["elastic_net_coefficient"].isna().sum()
    if n_missing > 0:
        print(f"\n[!] {n_missing} stable gene(s) have no weight because they weren't in that "
              f"specific frozen model's exact top-200 selection. This is expected, not a bug.")

    print("\nTop 5 by |Elastic Net coefficient|, per strategy:")
    for strategy in ["de_informed", "data_driven"]:
        print(f"\n{strategy}:")
        print(result[result["strategy"] == strategy].head(5).to_string(index=False))

    print("\nCAVEAT: Random Forest Gini importance is known to be biased toward "
          "correlated features. Gene expression data is heavily co-expressed, "
          "so RF importance rankings here should be read as suggestive, not "
          "definitive - a permutation-importance or SHAP analysis would be "
          "more robust but is deliberately out of scope for this lightweight pass.")


if __name__ == "__main__":
    main()
