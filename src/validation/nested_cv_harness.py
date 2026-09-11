"""
nested_cv_harness.py

Purpose
-------
Wires together the three Phase 2 pieces (outer folds, inner folds, and the
leakage-safe preprocessor) into the actual nested structure the design spec
calls for: an outer loop for honest performance estimation, and an inner
loop, built ONLY from the outer training fold's own patients, for feature
selection and hyperparameter tuning.

The rule this file exists to enforce structurally
---------------------------------------------------
Nothing that touches an outer test fold's samples may ever influence any
fitted parameter, not the preprocessing stats, not which genes get kept,
not which hyperparameters get chosen. Concretely:

  outer_train, outer_test = one outer fold
      │
      ├── inner_train, inner_val = built ONLY from outer_train's patients
      │       │
      │       └── LeakageSafePreprocessor.fit(inner_train)   <- fitting happens here
      │           .transform(inner_train), .transform(inner_val)
      │           (Phase 3 will plug feature selection + hyperparameter
      │            search in at this point, using only inner_train/inner_val)
      │
      └── once inner CV has picked a configuration: refit the
          preprocessor on the FULL outer_train, then transform
          outer_test using those outer_train-fitted stats - this is
          the only point at which outer_test is touched, and it is
          only ever transformed, never fit on.

Usage
-----
    from cv_splits import generate_repeated_grouped_splits
    from nested_cv_harness import generate_inner_splits, run_nested_fold_scaffold

    outer_assignments = generate_repeated_grouped_splits(...)

    for outer in outer_assignments:
        inner_splits = generate_inner_splits(
            outer_train_sample_ids=outer.train_sample_ids,
            sample_ids=coldata["sample_id"], labels=coldata["sample_type"],
            patient_ids=coldata["patient_id"], n_inner_splits=3, random_state=42,
        )
        # Phase 3 plugs in here: for each inner split, fit preprocessor +
        # feature selector + model on inner_train, evaluate on inner_val,
        # pick the best configuration, then refit on all of outer_train
        # and evaluate once on outer_test.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from leakage_safe_preprocessing import LeakageSafePreprocessor


@dataclass
class InnerFoldAssignment:
    inner_fold: int
    train_sample_ids: list
    val_sample_ids: list


def generate_inner_splits(
    outer_train_sample_ids: list,
    sample_ids: pd.Series,
    labels: pd.Series,
    patient_ids: pd.Series,
    n_inner_splits: int = 3,
    random_state: int = 42,
) -> list:
    """Builds inner CV splits using ONLY the samples in outer_train_sample_ids.
    This is the structural guarantee that outer test samples can never leak
    in: they're simply not part of the pool this function draws from."""
    id_to_label = dict(zip(sample_ids, labels))
    id_to_patient = dict(zip(sample_ids, patient_ids))

    outer_train_ids = pd.Series(outer_train_sample_ids)
    outer_train_labels = outer_train_ids.map(id_to_label)
    outer_train_patients = outer_train_ids.map(id_to_patient)

    splitter = StratifiedGroupKFold(
        n_splits=n_inner_splits, shuffle=True, random_state=random_state
    )

    inner_assignments = []
    for inner_fold, (train_idx, val_idx) in enumerate(
        splitter.split(X=outer_train_ids, y=outer_train_labels, groups=outer_train_patients)
    ):
        inner_assignments.append(
            InnerFoldAssignment(
                inner_fold=inner_fold,
                train_sample_ids=outer_train_ids.iloc[train_idx].tolist(),
                val_sample_ids=outer_train_ids.iloc[val_idx].tolist(),
            )
        )
    return inner_assignments


def verify_outer_test_never_in_inner_splits(
    inner_assignments: list, outer_test_sample_ids: list
) -> bool:
    """Independent structural check: confirms no outer test sample ever
    appears in any inner train or val set. Re-derived from scratch rather
    than trusted, same principle as the outer-fold leakage check."""
    outer_test_set = set(outer_test_sample_ids)
    clean = True
    for inner in inner_assignments:
        overlap_train = outer_test_set & set(inner.train_sample_ids)
        overlap_val = outer_test_set & set(inner.val_sample_ids)
        if overlap_train or overlap_val:
            print(f"[!] LEAKAGE at inner fold {inner.inner_fold}: outer test "
                  f"samples found in inner train {overlap_train} or "
                  f"inner val {overlap_val}")
            clean = False
    return clean


def demonstrate_leakage_safe_fit_transform(
    inner: InnerFoldAssignment,
    counts: pd.DataFrame,
    n_top_genes: int = 2000,
) -> dict:
    """Runs one inner fold's preprocessing exactly the way Phase 3 will use
    it: fit strictly on inner_train, transform both sides using those
    fitted stats. Returns the transformed data plus the fitted gene list,
    mainly so callers (and tests) can confirm the fit/transform split is
    actually happening as described rather than assumed."""
    preprocessor = LeakageSafePreprocessor(n_top_genes=n_top_genes)

    train_counts = counts[inner.train_sample_ids]
    val_counts = counts[inner.val_sample_ids]

    preprocessor.fit(train_counts)
    X_train = preprocessor.transform(train_counts)
    X_val = preprocessor.transform(val_counts)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "fitted_genes": preprocessor.selected_genes,
        "preprocessor": preprocessor,
    }
