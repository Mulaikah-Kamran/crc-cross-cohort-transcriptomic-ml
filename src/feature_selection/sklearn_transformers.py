"""
sklearn_transformers.py

Two small, standard-pattern scikit-learn transformers (fit/transform,
subclassing BaseEstimator + TransformerMixin so they drop straight into a
Pipeline and get fit correctly inside each CV fold automatically), plus
the .gmt file loader used to load MSigDB gene sets for the pathway
transformer.

Strategy A (t-test / ANOVA F-test) needs no custom code at all - it's just
sklearn's own SelectKBest(f_classif, k=N), since for two classes the F-test
and a two-sample t-test are the same thing (F = t^2). Import it directly
from sklearn.feature_selection where it's used.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def load_gmt_file(path: str) -> dict:
    """Parses a standard .gmt gene set file (tab-separated: pathway name,
    description, gene1, gene2, ...) into {pathway_name: [genes]}."""
    gene_sets = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pathway_name = parts[0]
            genes = parts[2:]
            gene_sets[pathway_name] = genes
    return gene_sets


class TopVarianceSelector(BaseEstimator, TransformerMixin):
    """Strategy B: keeps the k genes with the highest variance in the
    training fold. Purely unsupervised - never looks at y at all, which is
    the point of this strategy (see fit signature: y is accepted and
    ignored only so it fits the standard sklearn Transformer interface,
    which passes y to fit() even when the transformer doesn't need it)."""

    def __init__(self, k: int = 200):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X)
        variances = X.var(axis=0)
        self.selected_idx_ = np.argsort(variances)[::-1][: self.k]
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.selected_idx_]


class HallmarkPathwayScorer(BaseEstimator, TransformerMixin):
    """Strategy C: the combined z-score method (Lee et al. 2008). Each
    gene's mean/std is learned in fit() from whatever fold is passed in -
    when this sits inside a Pipeline inside GridSearchCV inside an outer
    CV loop, that's automatically the correct inner-training-fold-only
    data, scikit-learn handles the "fit only on train" part structurally,
    which is the whole reason to use Pipeline instead of custom code here.

    gene_ids must be provided at construction time and match the column
    order of whatever X arrives at fit()/transform() - X is expected to be
    log2-CPM values, genes as columns, samples as rows (the standard
    scikit-learn samples x features orientation).
    """

    def __init__(self, gene_sets: dict, gene_ids: list, min_genes_per_pathway: int = 5):
        self.gene_sets = gene_sets
        self.gene_ids = gene_ids
        self.min_genes_per_pathway = min_genes_per_pathway

    def fit(self, X, y=None):
        X = pd.DataFrame(np.asarray(X), columns=self.gene_ids)

        all_pathway_genes = set()
        for genes in self.gene_sets.values():
            all_pathway_genes.update(genes)
        relevant_genes = [g for g in self.gene_ids if g in all_pathway_genes]

        self.gene_means_ = X[relevant_genes].mean(axis=0)
        self.gene_stds_ = X[relevant_genes].std(axis=0).replace(0, 1.0)

        self.usable_pathways_ = {
            pw: [g for g in genes if g in relevant_genes]
            for pw, genes in self.gene_sets.items()
        }
        self.usable_pathways_ = {
            pw: genes for pw, genes in self.usable_pathways_.items()
            if len(genes) >= self.min_genes_per_pathway
        }
        self.pathway_names_ = list(self.usable_pathways_.keys())
        return self

    def transform(self, X):
        X = pd.DataFrame(np.asarray(X), columns=self.gene_ids)
        relevant_genes = self.gene_means_.index
        z = X[relevant_genes].sub(self.gene_means_, axis=1).div(self.gene_stds_, axis=1)

        scores = {pw: z[genes].mean(axis=1) for pw, genes in self.usable_pathways_.items()}
        return pd.DataFrame(scores)[self.pathway_names_].values
