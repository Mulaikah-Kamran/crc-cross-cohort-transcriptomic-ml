"""
run_nested_cv.py

Runs the actual Phase 3 experiment: for each outer fold (from Phase 2's
patient-grouped repeated CV), for each of 3 feature-selection strategies,
for each of 3 models, fit a GridSearchCV pipeline on the outer training
data (with its own inner CV for hyperparameter tuning), evaluate once on
the outer test fold, and log ROC-AUC and PR-AUC.

Why log2-CPM is computed once, upfront, outside the CV loop
-----------------------------------------------------------------
CPM and log2 are per-sample transforms - each sample's value depends only
on that sample's own library size, never on any other sample. There is
nothing to "fit," so doing this once before splitting is not a leakage
risk. What DOES need to be fit per-fold is gene selection and
standardization, which is exactly what putting them inside a Pipeline
handles automatically: GridSearchCV fits the whole pipeline fresh on each
training fold it sees, so the selection and scaling steps never see the
validation data they're being evaluated on.

Usage
-----
    python run_nested_cv.py \
        --counts data/raw/tcga_coad/tcga_coad_counts.parquet \
        --coldata data/raw/tcga_coad/tcga_coad_colData.parquet \
        --hallmark-gmt path/to/h.all.v2024.1.Hs.symbols.gmt \
        --n-splits 5 --n-repeats 5 \
        --out results/tables/phase3_nested_cv_results.csv
"""

import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, "src/validation")
sys.path.insert(0, "src/feature_selection")
sys.path.insert(0, "src/models")

from cv_splits import generate_repeated_grouped_splits, verify_no_patient_leakage
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
        selector = TopVarianceSelector(k=n_genes)
        return Pipeline([("select", selector), ("scale", StandardScaler()), ("clf", model_estimator)])

    if strategy == "pathway":
        scorer = HallmarkPathwayScorer(gene_sets=gene_sets, gene_ids=gene_ids)
        return Pipeline([("select", scorer), ("clf", model_estimator)])

    raise ValueError(f"Unknown strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--coldata", required=True)
    parser.add_argument("--hallmark-gmt", required=True)
    parser.add_argument("--n-genes", type=int, default=200)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--n-inner-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out", default="results/tables/phase3_nested_cv_results.csv")
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
    y_all = pd.Series(y_all, index=X_all.index)

    patient_ids_all = coldata.loc[X_all.index, "patient_id"]
    gene_ids = list(X_all.columns)

    gene_sets = load_gmt_file(args.hallmark_gmt)
    print(f"Loaded {len(gene_sets)} pathways from {args.hallmark_gmt}")

    outer_folds = generate_repeated_grouped_splits(
        sample_ids=X_all.index.to_series(), labels=y_all, patient_ids=patient_ids_all,
        n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.random_state,
    )
    assert verify_no_patient_leakage(outer_folds, patient_ids_all, X_all.index.to_series()), \
        "Outer fold patient leakage detected - stopping rather than running a compromised experiment."
    print(f"Generated and verified {len(outer_folds)} leakage-free outer folds.")

    models_and_grids = get_models_and_grids(random_state=args.random_state)
    strategies = ["de_informed", "data_driven", "pathway"]

    results = []
    header_written = False

    for outer in outer_folds:
        X_train, X_test = X_all.loc[outer.train_sample_ids], X_all.loc[outer.test_sample_ids]
        y_train, y_test = y_all.loc[outer.train_sample_ids], y_all.loc[outer.test_sample_ids]
        groups_train = patient_ids_all.loc[outer.train_sample_ids]

        inner_cv = StratifiedGroupKFold(n_splits=args.n_inner_splits, shuffle=True,
                                         random_state=args.random_state)

        for strategy in strategies:
            for model_name, (estimator, grid) in models_and_grids.items():
                t0 = time.time()
                row = {"repeat": outer.repeat, "fold": outer.fold,
                       "strategy": strategy, "model": model_name}
                try:
                    pipeline = build_pipeline(strategy, estimator, args.n_genes, gene_sets, gene_ids)

                    search = GridSearchCV(pipeline, grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1)
                    search.fit(X_train.values, y_train.to_numpy(dtype=int),
                               groups=groups_train.to_numpy(dtype=object))

                    y_proba = search.best_estimator_.predict_proba(X_test.values)[:, 1]
                    y_test_binary = y_test.to_numpy(dtype=int)

                    row["roc_auc"] = roc_auc_score(y_test_binary, y_proba)
                    row["pr_auc"] = average_precision_score(y_test_binary, y_proba)
                    row["best_params"] = search.best_params_
                    row["error"] = ""
                    row["seconds"] = round(time.time() - t0, 1)

                    print(f"repeat={outer.repeat} fold={outer.fold} strategy={strategy:12s} "
                          f"model={model_name:14s} ROC-AUC={row['roc_auc']:.3f} "
                          f"PR-AUC={row['pr_auc']:.3f} ({row['seconds']:.1f}s)")

                except Exception as e:
                    row["roc_auc"] = None
                    row["pr_auc"] = None
                    row["best_params"] = None
                    row["error"] = str(e)
                    row["seconds"] = round(time.time() - t0, 1)
                    print(f"[!] FAILED: repeat={outer.repeat} fold={outer.fold} "
                          f"strategy={strategy} model={model_name}: {e}")

                results.append(row)
                pd.DataFrame([row]).to_csv(args.out, mode="a", header=not header_written, index=False)
                header_written = True

    results_df = pd.DataFrame(results)
    print(f"\nCompleted {len(results_df)} fold/strategy/model combinations, "
          f"saved incrementally to {args.out}")

    n_failed = results_df["error"].astype(bool).sum()
    if n_failed > 0:
        print(f"\n[!] {n_failed} combinations failed - see the 'error' column in "
              f"{args.out} before trusting the summary below, since it's computed "
              f"only from the successful rows.")

    print("\n=== Summary: mean ROC-AUC / PR-AUC by strategy x model, across successful outer folds ===")
    summary = results_df.dropna(subset=["roc_auc"]).groupby(["strategy", "model"])[["roc_auc", "pr_auc"]].agg(["mean", "std"])
    print(summary)


if __name__ == "__main__":
    main()
