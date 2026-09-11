"""
freeze_models.py

Fits all 9 (3 feature-selection strategies x 3 models) pipelines on the
FULL TCGA-COAD discovery cohort and saves them for Phase 4 external
validation. All 9 are kept, not just an internal-CV "winner" - Phase 3's
internal CV couldn't meaningfully distinguish between them (all near
ceiling), and the actual research question is which one generalizes,
which internal CV on the training cohort itself cannot answer.

Usage
-----
    python freeze_models.py \
        --counts data/raw/tcga_coad/tcga_coad_counts.parquet \
        --coldata data/raw/tcga_coad/tcga_coad_colData.parquet \
        --hallmark-gmt data/raw/msigdb/h.all.Hs.ensembl.gmt \
        --n-genes 200 --n-inner-splits 3 \
        --out-dir results/models
"""

import argparse
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, "src/validation")
sys.path.insert(0, "src/feature_selection")
sys.path.insert(0, "src/models")

from sklearn_transformers import TopVarianceSelector, HallmarkPathwayScorer, load_gmt_file
from model_configs import get_models_and_grids


def log2_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    return np.log2(cpm + 1)


def strip_ensembl_version(gene_id: str) -> str:
    import re
    return re.sub(r"^(ENSG\d+)\.\d+$", r"\1", str(gene_id))


def build_pipeline(strategy: str, model_estimator, n_genes: int, gene_sets: dict, gene_ids: list):
    if strategy == "de_informed":
        return Pipeline([
            ("prefilter", VarianceThreshold(threshold=0.0)),
            ("select", SelectKBest(score_func=f_classif, k=n_genes)),
            ("scale", StandardScaler()),
            ("clf", model_estimator),
        ])
    if strategy == "data_driven":
        return Pipeline([("select", TopVarianceSelector(k=n_genes)),
                          ("scale", StandardScaler()), ("clf", model_estimator)])
    if strategy == "pathway":
        return Pipeline([("select", HallmarkPathwayScorer(gene_sets=gene_sets, gene_ids=gene_ids)),
                          ("clf", model_estimator)])
    raise ValueError(f"Unknown strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--coldata", required=True)
    parser.add_argument("--hallmark-gmt", required=True)
    parser.add_argument("--n-genes", type=int, default=200)
    parser.add_argument("--n-inner-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", default="results/models")
    args = parser.parse_args()

    counts = pd.read_parquet(args.counts).set_index("gene_id")
    coldata = pd.read_parquet(args.coldata).set_index("sample_id")
    counts = counts[coldata.index.intersection(counts.columns)]

    counts.index = counts.index.map(strip_ensembl_version)
    counts = counts.groupby(counts.index).sum()

    coldata["patient_id"] = coldata["barcode"].str.split("-").str[:3].str.join("-")

    log_expr = log2_cpm(counts)
    X_all = log_expr.T

    POSITIVE_LABEL = "Primary Tumor"
    y_all_raw = coldata.loc[X_all.index, "sample_type"]
    y_all = (y_all_raw.to_numpy(dtype=object) == POSITIVE_LABEL).astype(int)

    patient_ids_all = coldata.loc[X_all.index, "patient_id"]
    gene_ids = list(X_all.columns)

    gene_sets = load_gmt_file(args.hallmark_gmt)
    print(f"Loaded {len(gene_sets)} pathways from {args.hallmark_gmt}")
    print(f"Freezing on {X_all.shape[0]} samples, {X_all.shape[1]} genes "
          f"({y_all.sum()} tumor, {len(y_all) - y_all.sum()} normal)")

    with open(f"{args.out_dir}/training_gene_order.txt", "w") as f:
        f.write("\n".join(gene_ids))
    train_gene_means = X_all.mean(axis=0)
    train_gene_means.to_frame("mean_log2cpm").to_parquet(f"{args.out_dir}/training_gene_means.parquet")
    print(f"Saved training gene order ({len(gene_ids)} genes) and training gene means.")

    inner_cv = StratifiedGroupKFold(n_splits=args.n_inner_splits, shuffle=True,
                                     random_state=args.random_state)
    models_and_grids = get_models_and_grids(random_state=args.random_state)
    strategies = ["de_informed", "data_driven", "pathway"]

    frozen_summary = []
    for strategy in strategies:
        for model_name, (estimator, grid) in models_and_grids.items():
            print(f"\nFreezing strategy={strategy} model={model_name}...")
            pipeline = build_pipeline(strategy, estimator, args.n_genes, gene_sets, gene_ids)
            search = GridSearchCV(pipeline, grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1)
            search.fit(X_all.values, y_all, groups=patient_ids_all.to_numpy(dtype=object))

            out_path = f"{args.out_dir}/frozen_{strategy}_{model_name}.joblib"
            joblib.dump(search.best_estimator_, out_path)

            print(f"  best params: {search.best_params_}")
            print(f"  best inner CV ROC-AUC: {search.best_score_:.4f}")
            print(f"  saved to {out_path}")

            frozen_summary.append({
                "strategy": strategy, "model": model_name,
                "best_params": search.best_params_,
                "inner_cv_roc_auc": search.best_score_,
                "path": out_path,
            })

    pd.DataFrame(frozen_summary).to_csv(f"{args.out_dir}/frozen_models_summary.csv", index=False)
    print(f"\nAll 9 models frozen. Summary saved to {args.out_dir}/frozen_models_summary.csv")


if __name__ == "__main__":
    main()
