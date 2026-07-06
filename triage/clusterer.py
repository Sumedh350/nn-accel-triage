"""Failure clusterer: rule-based label assignment + DBSCAN outlier detection.

Input:  list of feature dicts from feature_extractor.extract_features()
Output: same list with two new fields per record:
          cluster_label  str   — rule-based bucket
          outlier        bool  — True when DBSCAN assigns cluster_id == -1
"""

from __future__ import annotations

from typing import Any


_BUCKET_MAP: dict[str, int] = {"none": 0, "small": 1, "medium": 2, "large": 3}


def assign_rule_label(feat: dict[str, Any]) -> str:
    """Return the rule-based cluster label for one feature dict.

    Rules are evaluated in priority order; the first match wins.
    """
    if feat["has_reset_symptom"]:
        return "reset_fault"
    if feat["has_overflow_symptom"]:
        return "overflow_fault"
    if feat["error_magnitude_bucket"] == "medium" and feat["mismatch_rate"] == 1.0:
        return "off_by_one"
    if (
        feat["error_magnitude_bucket"] == "large"
        and not feat["has_overflow_symptom"]
        and not feat["has_reset_symptom"]
    ):
        return "sign_error"
    if feat["status"] == "pass":
        return "clean_pass"
    return "uncategorized"


def _to_numeric_vector(feat: dict[str, Any]) -> list[float]:
    """Return a 5-element normalized vector for DBSCAN."""
    return [
        float(feat["mismatch_rate"]),
        min(feat["max_abs_error"] / 65536.0, 1.0),
        _BUCKET_MAP.get(feat["error_magnitude_bucket"], 0) / 3.0,
        float(feat["has_reset_symptom"]),
        float(feat["has_overflow_symptom"]),
    ]


def cluster(
    features: list[dict[str, Any]],
    eps: float = 0.5,
    min_samples: int = 2,
) -> list[dict[str, Any]]:
    """Assign cluster_label and outlier flag to each feature dict.

    Returns a new list of dicts; originals are not modified.
    """
    labeled: list[dict[str, Any]] = [
        {**feat, "cluster_label": assign_rule_label(feat)} for feat in features
    ]

    if len(labeled) < 2:
        for record in labeled:
            record["outlier"] = False
        return labeled

    import numpy as np
    from sklearn.cluster import DBSCAN

    X = np.array([_to_numeric_vector(f) for f in features], dtype=float)
    db_labels = DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_

    for record, db_label in zip(labeled, db_labels):
        record["outlier"] = bool(db_label == -1)

    return labeled
