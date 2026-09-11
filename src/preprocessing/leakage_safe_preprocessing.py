"""
leakage_safe_preprocessing.py

Purpose
-------
Makes the "fit only on the training fold" rule structural rather than a
convention someone has to remember to follow correctly every time. This is
the exact mistake the design spec explicitly warns against: running feature
selection or scaling on the full dataset before splitting into folds leaks
information from the test fold into training, and the resulting performance
numbers end up optimistic in a way that won't reproduce externally.

What's stateless (safe to apply anywhere, no fitting needed)
---------------------------------------------------------------
- CPM (counts-per-million) normalization: computed per-sample from that
  sample's own library size, doesn't depend on any other sample.
- log2(x + 1) transform: applied elementwise, same story.

What must be fit on the training fold only, then reused unchanged
----------------------------------------------------------------------
- Which genes pass the variance filter (top-N most variable genes) - this
  depends on the training fold's own gene variances.
- Per-gene mean and standard deviation for standardization - computed from
  the training fold, then applied to both train and test using those exact
  same fitted values (not recomputed on the test fold).

Usage
-----
    pre = LeakageSafePreprocessor(n_top_genes=2000)
    pre.fit(train_counts)                  # genes x samples, raw counts
    X_train = pre.transform(train_counts)
    X_test = pre.transform(test_counts)    # uses train-fitted genes + stats
"""

import numpy as np
import pandas as pd


class LeakageSafePreprocessor:
    def __init__(self, n_top_genes: int = 2000):
        self.n_top_genes = n_top_genes
        self._selected_genes = None
        self._gene_means = None
        self._gene_stds = None
        self._is_fit = False

    @staticmethod
    def _log2_cpm(counts: pd.DataFrame) -> pd.DataFrame:
        lib_sizes = counts.sum(axis=0)
        cpm = counts.div(lib_sizes, axis=1) * 1e6
        return np.log2(cpm + 1)

    def fit(self, train_counts: pd.DataFrame) -> "LeakageSafePreprocessor":
        """train_counts: genes x samples, raw counts, TRAINING FOLD ONLY."""
        log_expr = self._log2_cpm(train_counts)

        variances = log_expr.var(axis=1)
        self._selected_genes = variances.sort_values(ascending=False).head(self.n_top_genes).index

        filtered = log_expr.loc[self._selected_genes]
        self._gene_means = filtered.mean(axis=1)
        self._gene_stds = filtered.std(axis=1).replace(0, 1.0)  # avoid div-by-zero on constant genes

        self._is_fit = True
        return self

    def transform(self, counts: pd.DataFrame) -> pd.DataFrame:
        """counts: genes x samples, raw counts. Can be the training fold
        (to get its own transformed values) or a held-out fold / external
        cohort - the gene list and standardization stats used are always
        the ones learned in fit(), never recomputed here."""
        if not self._is_fit:
            raise RuntimeError("Call fit() on the training fold before transform().")

        log_expr = self._log2_cpm(counts)

        missing_genes = self._selected_genes.difference(log_expr.index)
        if len(missing_genes) > 0:
            print(f"[!] {len(missing_genes)} of {len(self._selected_genes)} fitted genes "
                  f"are absent from this data and will be dropped rather than imputed. "
                  f"This is expected when transforming an external cohort with a "
                  f"different gene annotation, but check the count is small.")

        available_genes = self._selected_genes.intersection(log_expr.index)
        filtered = log_expr.loc[available_genes]

        standardized = filtered.sub(self._gene_means.loc[available_genes], axis=0)
        standardized = standardized.div(self._gene_stds.loc[available_genes], axis=0)

        return standardized

    def fit_transform(self, train_counts: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_counts)
        return self.transform(train_counts)

    @property
    def selected_genes(self):
        if not self._is_fit:
            raise RuntimeError("Not fit yet.")
        return self._selected_genes
