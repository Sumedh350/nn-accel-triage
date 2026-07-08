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
    # Faults whose test vectors happen to produce no observable difference
    # (e.g. ACC_W wide enough for actual values, or quant pipeline clamps identically).
    if feat["status"] == "pass" and feat["variant"] != "golden":
        return "latent_fault"
    if feat["has_reset_symptom"]:
        return "reset_fault"
    if feat["has_overflow_symptom"]:
        return "overflow_fault"
    # Subtract fault: actual == -expected at first mismatch (large errors, rate=1.0).
    if (
        feat["error_magnitude_bucket"] == "large"
        and not feat["has_overflow_symptom"]
        and not feat["has_reset_symptom"]
        and feat.get("has_subtract_symptom", False)
    ):
        return "arithmetic_error"
    if (
        feat["error_magnitude_bucket"] == "large"
        and not feat["has_overflow_symptom"]
        and not feat["has_reset_symptom"]
    ):
        return "sign_error"
    # Medium-error boundary bugs: full-mismatch (loop terminates early) and
    # partial-mismatch (OOB spatial write affects one row).
    if (
        feat["error_magnitude_bucket"] == "medium"
        and feat["mismatch_rate"] > 0.0
        and not feat["has_overflow_symptom"]
        and not feat["has_reset_symptom"]
    ):
        return "off_by_one"
    # quant_unit small-error faults: distinguish by max_abs_error and mismatch_rate.
    # Saturation (no clamp): values wrap 8-bit instead of clamping — max error 225–255.
    if (
        feat["dut"] == "quant_unit"
        and feat["status"] == "fail"
        and feat["error_magnitude_bucket"] == "small"
        and feat["max_abs_error"] >= 225
    ):
        return "saturation_error"
    # Zero-point (unsigned zp): only channels with negative zp are affected — partial rows.
    if (
        feat["dut"] == "quant_unit"
        and feat["status"] == "fail"
        and feat["error_magnitude_bucket"] == "small"
        and feat["mismatch_rate"] <= 0.5
    ):
        return "zero_point_error"
    # Shift (fixed shift=1): affects all channels with small per-element error.
    if (
        feat["dut"] == "quant_unit"
        and feat["status"] == "fail"
        and feat["error_magnitude_bucket"] == "small"
    ):
        return "shift_error"
    if feat["status"] == "pass":
        return "clean_pass"
    return "uncategorized"


def _to_numeric_vector(feat: dict[str, Any]) -> list[float]:
    """Return a 6-element normalized vector for DBSCAN."""
    return [
        float(feat["mismatch_rate"]),
        min(feat["max_abs_error"] / 65536.0, 1.0),
        _BUCKET_MAP.get(feat["error_magnitude_bucket"], 0) / 3.0,
        float(feat["has_reset_symptom"]),
        float(feat["has_overflow_symptom"]),
        float(feat.get("has_subtract_symptom", False)),
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
