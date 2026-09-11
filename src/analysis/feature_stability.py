"""
feature_stability.py

Phase 5: bootstrap stability analysis on the TCGA-COAD discovery cohort.
Pre-specified BEFORE any results were seen: B=200 patient-level bootstrap
resamples, 70% selection-frequency threshold for a feature to count as
"stable".

Usage
-----
    python feature_stability.py \
        --counts data/raw/tcga_coad/tcga_coad_counts.parquet \
        --coldata data/raw/tcga_coad/tcga_coad_colData.parquet \
        --hallmark-gmt data/raw/msigdb/h.all.Hs.ensembl.gmt \
        --n-genes 200 --n-boot 200 --threshold 0.70 \
        --out-dir results/tables
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

warnings.filterwarnings("ignore")

sys.path.insert(0, "src/feature_selection")
from sklearn_transformers import TopVarianceSelector, HallmarkPathwayScorer, load_gmt_file


def log2_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    return np.log2(cpm + 1)


def strip_ensembl_version(gene_id: str) -> str:
    import re
    return re.sub(r"^(ENSG\d+)\.\d+$", r"\1", str(gene_id))


def patient_level_bootstrap_indices(patient_ids: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    unique_patients = np.unique(patient_ids)
    resampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
    indices = []
    for patient in resampled_patients:
        indices.extend(np.where(patient_ids == patient)[0])
    return np.array(indices)


def run_stability_de_informed(X, y, patient_ids, gene_ids, n_genes, n_boot, random_state):
    rng = np.random.RandomState(random_state)
    counts = {g: 0 for g in gene_ids}

    for b in range(n_boot):
        idx = patient_level_bootstrap_indices(patient_ids, rng)
        X_b, y_b = X[idx], y[idx]

        prefilter = VarianceThreshold(threshold=0.0)
        X_filtered = prefilter.fit_transform(X_b)
        prefilter_idx = prefilter.get_support(indices=True)

        selector = SelectKBest(score_func=f_classif, k=min(n_genes, X_filtered.shape[1]))
        selector.fit(X_filtered, y_b)
        selectk_idx = selector.get_support(indices=True)

        original_idx = prefilter_idx[selectk_idx]
        for i in original_idx:
            counts[gene_ids[i]] += 1

        if (b + 1) % 50 == 0:
            print(f"  de_informed: {b+1}/{n_boot} resamples done")

    return {g: c / n_boot for g, c in counts.items()}


def run_stability_data_driven(X, y, patient_ids, gene_ids, n_genes, n_boot, random_state):
    rng = np.random.RandomState(random_state)
    counts = {g: 0 for g in gene_ids}

    for b in range(n_boot):
        idx = patient_level_bootstrap_indices(patient_ids, rng)
        X_b = X[idx]

        selector = TopVarianceSelector(k=n_genes)
        selector.fit(X_b)
        for i in selector.selected_idx_:
            counts[gene_ids[i]] += 1

        if (b + 1) % 50 == 0:
            print(f"  data_driven: {b+1}/{n_boot} resamples done")

    return {g: c / n_boot for g, c in counts.items()}


def run_stability_pathway(X, y, patient_ids, gene_ids, gene_sets, n_boot, random_state, alpha=0.05):
    rng = np.random.RandomState(random_state)

    scorer = HallmarkPathwayScorer(gene_sets=gene_sets, gene_ids=gene_ids)
    scorer.fit(X)
    usable_pathways = scorer.usable_pathways_
    sig_counts = {pw: 0 for pw in usable_pathways}

    for b in range(n_boot):
        idx = patient_level_bootstrap_indices(patient_ids, rng)
        X_b, y_b = X[idx], y[idx]

        scorer_b = HallmarkPathwayScorer(gene_sets=gene_sets, gene_ids=gene_ids)
        scorer_b.fit(X_b)
        pathway_scores = scorer_b.transform(X_b)
        pathway_names = scorer_b.pathway_names_

        for j, pw in enumerate(pathway_names):
            if pw not in sig_counts:
                continue
            pos_scores = pathway_scores[y_b == 1, j]
            neg_scores = pathway_scores[y_b == 0, j]
            if len(pos_scores) < 2 or len(neg_scores) < 2:
                continue
            _, p_val = stats.ttest_ind(pos_scores, neg_scores, equal_var=False)
            if p_val < alpha:
                sig_counts[pw] += 1

        if (b + 1) % 50 == 0:
            print(f"  pathway: {b+1}/{n_boot} resamples done")

    return {pw: c / n_boot for pw, c in sig_counts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--coldata", required=True)
    parser.add_argument("--hallmark-gmt", required=True)
    parser.add_argument("--n-genes", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", default="results/tables")
    args = parser.parse_args()

    counts = pd.read_parquet(args.counts).set_index("gene_id")
    coldata = pd.read_parquet(args.coldata).set_index("sample_id")
    counts = counts[coldata.index.intersection(counts.columns)]
    counts.index = counts.index.map(strip_ensembl_version)
    counts = counts.groupby(counts.index).sum()

    coldata["patient_id"] = coldata["barcode"].str.split("-").str[:3].str.join("-")

    log_expr = log2_cpm(counts)
    X_all = log_expr.T
    gene_ids = list(X_all.columns)

    POSITIVE_LABEL = "Primary Tumor"
    y_all = (coldata.loc[X_all.index, "sample_type"].to_numpy(dtype=object) == POSITIVE_LABEL).astype(int)
    patient_ids_all = coldata.loc[X_all.index, "patient_id"].to_numpy(dtype=object)

    X = X_all.values

    gene_sets = load_gmt_file(args.hallmark_gmt)
    print(f"Loaded {len(gene_sets)} pathways.")
    print(f"Running B={args.n_boot} patient-level bootstrap resamples, "
          f"threshold={args.threshold:.0%}, on {X_all.shape[0]} samples, "
          f"{len(np.unique(patient_ids_all))} unique patients.\n")

    print("Strategy A (DE-informed)...")
    freq_de = run_stability_de_informed(X, y_all, patient_ids_all, gene_ids,
                                         args.n_genes, args.n_boot, args.random_state)

    print("\nStrategy B (data-driven)...")
    freq_dd = run_stability_data_driven(X, y_all, patient_ids_all, gene_ids,
                                         args.n_genes, args.n_boot, args.random_state)

    print("\nStrategy C (pathway)...")
    freq_pw = run_stability_pathway(X, y_all, patient_ids_all, gene_ids, gene_sets,
                                     args.n_boot, args.random_state)

    consensus_de = {g: f for g, f in freq_de.items() if f >= args.threshold}
    consensus_dd = {g: f for g, f in freq_dd.items() if f >= args.threshold}
    consensus_pw = {p: f for p, f in freq_pw.items() if f >= args.threshold}

    print(f"\n=== Consensus features (>= {args.threshold:.0%} selection frequency) ===")
    print(f"DE-informed: {len(consensus_de)} stable genes")
    print(f"Data-driven: {len(consensus_dd)} stable genes")
    print(f"Pathway:     {len(consensus_pw)} stable pathways")

    gene_overlap = set(consensus_de) & set(consensus_dd)
    print(f"\nGene-level overlap between DE-informed and data-driven consensus sets: "
          f"{len(gene_overlap)} genes "
          f"({len(gene_overlap)/max(len(set(consensus_de)|set(consensus_dd)),1):.1%} of their union)")

    pd.Series(freq_de).sort_values(ascending=False).to_csv(f"{args.out_dir}/stability_de_informed.csv",
                                                             header=["selection_frequency"])
    pd.Series(freq_dd).sort_values(ascending=False).to_csv(f"{args.out_dir}/stability_data_driven.csv",
                                                             header=["selection_frequency"])
    pd.Series(freq_pw).sort_values(ascending=False).to_csv(f"{args.out_dir}/stability_pathway.csv",
                                                             header=["significant_association_frequency"])

    print(f"\nFull frequency tables saved to {args.out_dir}/stability_*.csv")
    if gene_overlap:
        print(f"\nOverlapping stable genes (DE-informed & data-driven): {sorted(gene_overlap)}")


if __name__ == "__main__":
    main()
