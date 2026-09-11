"""
cv_splits.py

Purpose
-------
Generate the discovery cohort's outer CV splits: 5-fold, repeated 5 times
(25 total fold assignments), stratified by class (tumor vs tumor-adjacent
normal), with patient identity enforced as the grouping unit so that a
patient contributing both a tumor and a matched-normal sample never has
those two samples split across train and test within the same fold.

Why grouping matters here specifically
---------------------------------------
The TCGA batch audit found that a handful of patients in TCGA-COAD
contribute both a tumor and a matched-normal sample. If those ever ended
up in different folds, the model could pick up on patient-specific
technical or biological quirks from the training sample and get an
artificially easy time recognizing the "twin" sample in the test fold.
This is the same principle as the patient-level leakage rule stated
throughout the design spec, just applied concretely to the discovery
cohort's actual structure.

Why repeated rather than single-pass
--------------------------------------
TCGA-COAD's normal class is thin (~41 samples across ~522 total). A single
5-fold split leaves roughly 8 normal samples per outer fold, which makes
performance estimates and feature-selection stability sensitive to exactly
which samples happened to land where. Repeating the split with different
random seeds and averaging results reduces that sensitivity without
touching the external validation cohorts at all.

Usage
-----
    from cv_splits import generate_repeated_grouped_splits

    splits = generate_repeated_grouped_splits(
        sample_ids=coldata["sample_id"],
        labels=coldata["sample_type"],
        patient_ids=coldata["patient_id"],
        n_splits=5,
        n_repeats=5,
        random_state=42,
    )
    # splits is a list of (repeat_idx, fold_idx, train_sample_ids, test_sample_ids)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass
class FoldAssignment:
    repeat: int
    fold: int
    train_sample_ids: list
    test_sample_ids: list


def generate_repeated_grouped_splits(
    sample_ids: pd.Series,
    labels: pd.Series,
    patient_ids: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> list:
    """Returns a list of FoldAssignment, one per (repeat, fold) combination.

    StratifiedGroupKFold does the actual work: it keeps class proportions
    similar across folds while guaranteeing no group (patient) appears in
    both the train and test side of any single fold. Repeating it n_repeats
    times with different random states gives independent fold arrangements
    to average over, which is the point, a single arrangement is one noisy
    draw.
    """
    sample_ids = pd.Series(sample_ids).reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    patient_ids = pd.Series(patient_ids).reset_index(drop=True)

    assert len(sample_ids) == len(labels) == len(patient_ids), \
        "sample_ids, labels, and patient_ids must be the same length and aligned"

    assignments = []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state + repeat
        )
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X=sample_ids, y=labels, groups=patient_ids)
        ):
            assignments.append(
                FoldAssignment(
                    repeat=repeat,
                    fold=fold,
                    train_sample_ids=sample_ids.iloc[train_idx].tolist(),
                    test_sample_ids=sample_ids.iloc[test_idx].tolist(),
                )
            )
    return assignments


def verify_no_patient_leakage(assignments: list, patient_ids: pd.Series, sample_ids: pd.Series) -> bool:
    """Independent check that no patient appears on both sides of any fold.
    This is deliberately re-derived from scratch rather than trusting
    StratifiedGroupKFold blindly, since a leakage bug here would silently
    undermine every result downstream."""
    id_to_patient = dict(zip(sample_ids, patient_ids))
    all_clean = True
    for a in assignments:
        train_patients = {id_to_patient[s] for s in a.train_sample_ids}
        test_patients = {id_to_patient[s] for s in a.test_sample_ids}
        overlap = train_patients & test_patients
        if overlap:
            print(f"[!] LEAKAGE at repeat {a.repeat}, fold {a.fold}: "
                  f"patient(s) {overlap} appear in both train and test")
            all_clean = False
    return all_clean


def summarize_fold_class_balance(assignments: list, sample_ids: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """Quick per-fold class count table, useful as a first sanity check
    before the deeper plate-level diagnostic in fold_diagnostics.py."""
    id_to_label = dict(zip(sample_ids, labels))
    rows = []
    for a in assignments:
        test_labels = [id_to_label[s] for s in a.test_sample_ids]
        counts = pd.Series(test_labels).value_counts().to_dict()
        rows.append({"repeat": a.repeat, "fold": a.fold, **counts})
    return pd.DataFrame(rows).fillna(0)
