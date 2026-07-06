"""Tests for triage/clusterer.py."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from clusterer import assign_rule_label, cluster
from feature_extractor import load_and_extract

_DB_PATH = Path(__file__).parent.parent / "regression_db.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_feat(**overrides) -> dict:
    base = {
        "run_id": "test-run",
        "dut": "mac_array",
        "variant": "golden",
        "test_name": "synthetic",
        "status": "fail",
        "config_n": 4,
        "config_data_type": 0,
        "config_acc_w": 32,
        "mismatch_rate": 0.5,
        "max_abs_error": 100,
        "error_magnitude_bucket": "small",
        "has_reset_symptom": False,
        "has_overflow_symptom": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Rule-based label tests
# ---------------------------------------------------------------------------

def test_reset_fault():
    feat = make_feat(has_reset_symptom=True)
    assert assign_rule_label(feat) == "reset_fault"


def test_overflow_fault():
    feat = make_feat(has_overflow_symptom=True, has_reset_symptom=False)
    assert assign_rule_label(feat) == "overflow_fault"


def test_off_by_one():
    feat = make_feat(
        error_magnitude_bucket="medium",
        mismatch_rate=1.0,
        has_reset_symptom=False,
        has_overflow_symptom=False,
    )
    assert assign_rule_label(feat) == "off_by_one"


def test_sign_error():
    feat = make_feat(
        error_magnitude_bucket="large",
        has_overflow_symptom=False,
        has_reset_symptom=False,
    )
    assert assign_rule_label(feat) == "sign_error"


def test_clean_pass():
    feat = make_feat(
        status="pass",
        has_reset_symptom=False,
        has_overflow_symptom=False,
        error_magnitude_bucket="none",
        mismatch_rate=0.0,
        max_abs_error=0,
    )
    assert assign_rule_label(feat) == "clean_pass"


def test_uncategorized():
    feat = make_feat(
        status="fail",
        error_magnitude_bucket="small",
        mismatch_rate=0.5,
        has_reset_symptom=False,
        has_overflow_symptom=False,
    )
    assert assign_rule_label(feat) == "uncategorized"


# ---------------------------------------------------------------------------
# Priority-order tests
# ---------------------------------------------------------------------------

def test_reset_beats_overflow():
    feat = make_feat(has_reset_symptom=True, has_overflow_symptom=True)
    assert assign_rule_label(feat) == "reset_fault"


def test_overflow_beats_off_by_one():
    feat = make_feat(
        has_overflow_symptom=True,
        error_magnitude_bucket="medium",
        mismatch_rate=1.0,
    )
    assert assign_rule_label(feat) == "overflow_fault"


# ---------------------------------------------------------------------------
# cluster() returns enriched dicts
# ---------------------------------------------------------------------------

def test_cluster_adds_fields():
    feats = [make_feat(), make_feat(status="pass", mismatch_rate=0.0, max_abs_error=0,
                                    error_magnitude_bucket="none")]
    result = cluster(feats)
    for r in result:
        assert "cluster_label" in r
        assert "outlier" in r
        assert isinstance(r["outlier"], bool)


def test_cluster_does_not_mutate_input():
    feat = make_feat()
    original_keys = set(feat.keys())
    cluster([feat, make_feat()])
    assert set(feat.keys()) == original_keys


# ---------------------------------------------------------------------------
# Real regression_db coverage test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _DB_PATH.exists(), reason="regression_db.jsonl not found")
def test_all_real_records_labeled():
    features = load_and_extract(_DB_PATH)
    assert len(features) == 27

    result = cluster(features)

    labels = [r["cluster_label"] for r in result]
    assert "uncategorized" not in labels, (
        f"Unexpected uncategorized records: "
        f"{[r for r in result if r['cluster_label'] == 'uncategorized']}"
    )

    counts = Counter(labels)
    assert counts["clean_pass"] == 15
    assert counts["reset_fault"] == 3
    assert counts["overflow_fault"] == 3
    assert counts["off_by_one"] == 3
    assert counts["sign_error"] == 3


# ---------------------------------------------------------------------------
# DBSCAN outlier detection tests
# ---------------------------------------------------------------------------

def test_outlier_flag_present():
    feats = [make_feat() for _ in range(3)]
    result = cluster(feats)
    for r in result:
        assert "outlier" in r
        assert isinstance(r["outlier"], bool)


def test_dbscan_detects_outlier():
    # 5 near-identical "pass" records forming a tight cluster
    cluster_feats = [
        make_feat(
            status="pass",
            mismatch_rate=0.0,
            max_abs_error=0,
            error_magnitude_bucket="none",
            has_reset_symptom=False,
            has_overflow_symptom=False,
        )
        for _ in range(5)
    ]
    # 1 singleton with extreme values, far from the cluster
    outlier_feat = make_feat(
        status="fail",
        mismatch_rate=1.0,
        max_abs_error=65535,
        error_magnitude_bucket="large",
        has_reset_symptom=False,
        has_overflow_symptom=False,
    )
    feats = cluster_feats + [outlier_feat]
    result = cluster(feats, eps=0.3, min_samples=2)

    outlier_flags = [r["outlier"] for r in result]
    # The singleton (last record) should be flagged
    assert outlier_flags[-1] is True
    # The tight cluster should not be flagged
    assert all(f is False for f in outlier_flags[:5])


def test_no_outlier_on_uniform_data():
    feats = [
        make_feat(mismatch_rate=0.0, max_abs_error=0, error_magnitude_bucket="none")
        for _ in range(5)
    ]
    result = cluster(feats, eps=0.5, min_samples=2)
    assert all(r["outlier"] is False for r in result)


def test_single_record_no_crash():
    result = cluster([make_feat()])
    assert len(result) == 1
    assert result[0]["outlier"] is False
    assert "cluster_label" in result[0]
