"""
fold_diagnostics.py

Purpose
-------
The TCGA-COAD batch audit found that although biology explains more
variance than plate overall, all 41 normal samples in the discovery
cohort come from only 6 of 28 plates. That's a manageable batch effect at
the whole-cohort level, but it becomes a real problem if a single CV fold
happens to concentrate most of those 6 plates, since that fold's test set
would then be dominated by a couple of plates rather than reflecting the
cohort's actual diversity.

This script checks every generated fold for exactly that risk: are normal
samples and plates reasonably spread across folds, or is any single fold
an outlier worth knowing about before trusting its results.

Usage
-----
    from fold_diagnostics import check_fold_plate_balance

    check_fold_plate_balance(
        assignments=assignments,               # from cv_splits.py
        sample_ids=coldata["sample_id"],
        labels=coldata["sample_type"],
        plates=coldata["plate"],
    )
"""

import pandas as pd


def check_fold_plate_balance(assignments: list, sample_ids: pd.Series,
                              labels: pd.Series, plates: pd.Series,
                              normal_label: str = "Solid Tissue Normal",
                              concentration_threshold: float = 0.5) -> pd.DataFrame:
    """For each fold, checks what fraction of that fold's test-side normal
    samples come from a single plate. Flags any fold where one plate
    accounts for more than `concentration_threshold` of the normal samples
    in that fold's test set - the default of 0.5 means "more than half of
    this fold's normal samples share one plate," which would make that
    fold's normal-class evaluation really a single-plate evaluation in
    disguise.
    """
    id_to_label = dict(zip(sample_ids, labels))
    id_to_plate = dict(zip(sample_ids, plates))

    rows = []
    for a in assignments:
        test_labels = [id_to_label[s] for s in a.test_sample_ids]
        test_plates = [id_to_plate[s] for s in a.test_sample_ids]

        normal_plates = [p for lbl, p in zip(test_labels, test_plates) if lbl == normal_label]
        n_normal = len(normal_plates)

        if n_normal == 0:
            rows.append({
                "repeat": a.repeat, "fold": a.fold, "n_normal_in_fold": 0,
                "top_plate_share": None, "flag": "NO NORMAL SAMPLES IN FOLD - investigate"
            })
            continue

        plate_counts = pd.Series(normal_plates).value_counts()
        top_plate_share = plate_counts.iloc[0] / n_normal

        flag = ""
        if top_plate_share > concentration_threshold:
            flag = (f"HIGH CONCENTRATION - {plate_counts.index[0]} accounts for "
                     f"{top_plate_share:.0%} of this fold's normal samples")

        rows.append({
            "repeat": a.repeat, "fold": a.fold, "n_normal_in_fold": n_normal,
            "top_plate_share": round(top_plate_share, 2), "flag": flag,
        })

    result = pd.DataFrame(rows)

    n_flagged = (result["flag"] != "").sum()
    print(f"Checked {len(result)} folds. {n_flagged} flagged for plate concentration "
          f"above {concentration_threshold:.0%}.")
    if n_flagged > 0:
        print("\nFlagged folds:")
        print(result[result["flag"] != ""].to_string(index=False))
        print("\nA flagged fold isn't necessarily fatal, but its results should be "
              "interpreted knowing the normal class in that fold is less diverse "
              "than the cohort as a whole. Worth checking whether flags cluster on "
              "particular repeats (bad luck in one random draw) or persist across "
              "repeats (a structural issue worth addressing before trusting the CV "
              "results at all).")
    else:
        print("No folds flagged - normal-sample plate distribution looks reasonable "
              "across all fold assignments.")

    return result
