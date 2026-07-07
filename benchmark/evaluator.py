"""Benchmark evaluator: score triage agent cluster_label predictions against ground truth.

Public API:
    evaluate(clustered_features, reports, ground_truth_path) -> dict
    format_report(metrics) -> str
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any


CLUSTER_TO_GT: dict[str, str] = {
    "clean_pass":     "no_fault",
    "overflow_fault": "accumulator_overflow",
    "sign_error":     "sign_extension_error",
    "off_by_one":     "loop_boundary_error",
    "reset_fault":    "reset_polarity_error",
    "uncategorized":  "unknown",
}

_CONFIDENCE_SCORE: dict[str, float] = {
    "high":   1.0,
    "medium": 0.5,
    "low":    0.0,
}


def evaluate(
    clustered_features: list[dict[str, Any]],
    reports: dict[str, dict[str, Any]],
    ground_truth_path: str | Path,
) -> dict[str, Any]:
    """Compare cluster_label predictions against ground truth labels.

    Args:
        clustered_features: output of clusterer.cluster() — each dict has
            "variant" and "cluster_label" fields.
        reports: output of triage_agent.triage() — maps cluster_label → report
            dict; each report has a "confidence" key.
        ground_truth_path: path to ground_truth.json.

    Returns:
        {
            "per_label": {label: {"precision", "recall", "f1", "support"}},
            "overall_accuracy": float,
            "mean_confidence": float,
            "total_records": int,
        }
    """
    ground_truth: dict[str, dict[str, str]] = json.loads(
        Path(ground_truth_path).read_text()
    )

    # Build list of (true_label, predicted_label) pairs, skipping unknown variants.
    pairs: list[tuple[str, str]] = []
    for feat in clustered_features:
        variant = feat.get("variant", "")
        if variant not in ground_truth:
            warnings.warn(f"variant {variant!r} not in ground truth — skipped")
            continue
        true_label = ground_truth[variant]["label"]
        predicted_label = CLUSTER_TO_GT.get(feat["cluster_label"], "unknown")
        pairs.append((true_label, predicted_label))

    total = len(pairs)

    # Confusion matrix: confusion[true][predicted] = count
    confusion: dict[str, dict[str, int]] = {}
    all_gt_labels: set[str] = {gt["label"] for gt in ground_truth.values()}
    for true_label, predicted_label in pairs:
        confusion.setdefault(true_label, {})
        confusion[true_label][predicted_label] = (
            confusion[true_label].get(predicted_label, 0) + 1
        )

    # Per-label precision / recall / F1
    per_label: dict[str, dict[str, float | int]] = {}
    correct = 0
    for label in sorted(all_gt_labels):
        tp = confusion.get(label, {}).get(label, 0)
        correct += tp

        # FP: other true classes predicted as this label
        fp = sum(
            confusion.get(true, {}).get(label, 0)
            for true in confusion
            if true != label
        )
        # FN: this true class predicted as something else
        fn = sum(
            count
            for pred, count in confusion.get(label, {}).items()
            if pred != label
        )
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    overall_accuracy = correct / total if total > 0 else 0.0

    # Mean confidence from reports
    confidence_scores = [
        _CONFIDENCE_SCORE.get(report.get("confidence", "low"), 0.0)
        for report in reports.values()
    ]
    mean_confidence = (
        sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    )

    return {
        "per_label": per_label,
        "overall_accuracy": overall_accuracy,
        "mean_confidence": mean_confidence,
        "total_records": total,
    }


def format_report(metrics: dict[str, Any]) -> str:
    """Return a human-readable summary of evaluate() output."""
    lines: list[str] = []
    lines.append(f"{'Label':<26} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Support':>8}")
    lines.append("-" * 65)
    for label, vals in sorted(metrics["per_label"].items()):
        lines.append(
            f"{label:<26} {vals['precision']:>9.3f} {vals['recall']:>9.3f}"
            f" {vals['f1']:>9.3f} {vals['support']:>8}"
        )
    lines.append("-" * 65)
    lines.append(f"overall accuracy: {metrics['overall_accuracy']:.3f}"
                 f"   total records: {metrics['total_records']}")
    lines.append(f"mean confidence:  {metrics['mean_confidence']:.3f}")
    return "\n".join(lines)
