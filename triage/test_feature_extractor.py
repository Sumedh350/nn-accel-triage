"""Tests for triage/feature_extractor.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from feature_extractor import extract_features, load_and_extract

DB_PATH = Path(__file__).parent.parent / "regression_db.jsonl"

REQUIRED_KEYS = {
    "run_id", "dut", "variant", "test_name", "status",
    "config_n", "config_data_type", "config_acc_w",
    "mismatch_rate", "max_abs_error", "error_magnitude_bucket",
    "has_reset_symptom", "has_overflow_symptom",
}


def _make_record(
    *,
    status: str = "fail",
    variant: str = "fault_reset",
    mismatch_count: int = 16,
    total_elements: int = 16,
    max_abs_error: int = 32946,
    first_actual: int = 0,
    data_type: int | None = 0,
    acc_w: int = 32,
    n: int = 4,
) -> dict:
    config: dict = {"n": n, "acc_w": acc_w}
    if data_type is not None:
        config["data_type"] = data_type

    if status == "pass":
        details = None
    else:
        details = {
            "total_elements": total_elements,
            "mismatch_count": mismatch_count,
            "max_abs_error": max_abs_error,
            "first_mismatch": {"row": 0, "col": 0, "expected": 13549, "actual": first_actual},
        }

    return {
        "run_id": "00000000-0000-0000-0000-000000000000",
        "timestamp": "2026-07-06T00:00:00.000000Z",
        "dut": "mac_array",
        "variant": variant,
        "config": config,
        "test_name": "synthetic_test",
        "status": status,
        "mismatch_details": details,
    }


def test_pass_record() -> None:
    feat = extract_features(_make_record(status="pass", variant="golden"))
    assert feat["mismatch_rate"] == 0.0
    assert feat["max_abs_error"] == 0
    assert feat["error_magnitude_bucket"] == "none"
    assert feat["has_reset_symptom"] is False
    assert feat["has_overflow_symptom"] is False
    assert feat["status"] == "pass"
    assert feat["variant"] == "golden"
    assert feat["config_n"] == 4
    assert feat["config_data_type"] == 0
    assert feat["config_acc_w"] == 32


def test_fail_small_error() -> None:
    feat = extract_features(
        _make_record(
            variant="fault_off_by_one",
            mismatch_count=1,
            total_elements=16,
            max_abs_error=100,
            first_actual=200,
        )
    )
    assert feat["mismatch_rate"] == pytest.approx(1 / 16)
    assert feat["max_abs_error"] == 100
    assert feat["error_magnitude_bucket"] == "small"
    assert feat["has_reset_symptom"] is False
    assert feat["has_overflow_symptom"] is False


def test_fail_medium_error() -> None:
    feat = extract_features(
        _make_record(
            variant="fault_off_by_one",
            mismatch_count=16,
            total_elements=16,
            max_abs_error=13000,
            first_actual=2278,
        )
    )
    assert feat["mismatch_rate"] == 1.0
    assert feat["error_magnitude_bucket"] == "medium"
    assert feat["has_reset_symptom"] is False
    assert feat["has_overflow_symptom"] is False


def test_fail_large_no_symptom() -> None:
    feat = extract_features(
        _make_record(
            variant="fault_wrong_sign",
            mismatch_count=12,
            total_elements=16,
            max_abs_error=50000,
            first_actual=5000,
        )
    )
    assert feat["error_magnitude_bucket"] == "large"
    assert feat["has_reset_symptom"] is False
    assert feat["has_overflow_symptom"] is False


def test_reset_symptom() -> None:
    feat = extract_features(
        _make_record(
            variant="fault_reset",
            mismatch_count=16,
            total_elements=16,
            max_abs_error=32946,
            first_actual=0,
        )
    )
    assert feat["has_reset_symptom"] is True
    assert feat["has_overflow_symptom"] is False
    assert feat["error_magnitude_bucket"] == "large"


def test_overflow_symptom() -> None:
    feat = extract_features(
        _make_record(
            variant="fault_acc_overflow",
            mismatch_count=1,
            total_elements=16,
            max_abs_error=65536,
            first_actual=32590,
        )
    )
    assert feat["has_overflow_symptom"] is True
    assert feat["has_reset_symptom"] is False
    assert feat["mismatch_rate"] == pytest.approx(1 / 16)


def test_bucket_boundaries() -> None:
    def bucket(e: int) -> str:
        return extract_features(
            _make_record(mismatch_count=1, total_elements=1, max_abs_error=e, first_actual=1)
        )["error_magnitude_bucket"]

    assert bucket(255) == "small"
    assert bucket(256) == "medium"
    assert bucket(32767) == "medium"
    assert bucket(32768) == "large"


def test_load_regression_db() -> None:
    features = load_and_extract(DB_PATH)
    assert len(features) >= 84  # 27 original + 57 new fault variants

    for feat in features:
        assert REQUIRED_KEYS <= feat.keys(), f"Missing keys in {feat['run_id']}"
        assert isinstance(feat["mismatch_rate"], float)
        assert isinstance(feat["max_abs_error"], int)
        assert isinstance(feat["has_reset_symptom"], bool)
        assert isinstance(feat["has_overflow_symptom"], bool)
        assert feat["error_magnitude_bucket"] in ("none", "small", "medium", "large")

    passing = [f for f in features if f["status"] == "pass"]
    failing = [f for f in features if f["status"] == "fail"]
    assert len(passing) >= 15   # golden + latent acc_w faults
    assert len(failing) >= 12

    for feat in passing:
        assert feat["mismatch_rate"] == 0.0
        assert feat["max_abs_error"] == 0
        assert feat["error_magnitude_bucket"] == "none"

    for feat in failing:
        assert feat["mismatch_rate"] > 0.0
