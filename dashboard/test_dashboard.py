"""pytest tests for dashboard/dashboard.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard.dashboard import generate_dashboard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PASS_RECORD = {
    "run_id": "aaaa",
    "timestamp": "2026-01-01T00:00:00+00:00",
    "dut": "mac_array",
    "variant": "golden",
    "config": {"n": 4, "data_type": 0, "acc_w": 32},
    "test_name": "random_n4_int8_seed1",
    "status": "pass",
    "mismatch_details": None,
}

_FAIL_RECORD = {
    "run_id": "bbbb",
    "timestamp": "2026-01-01T00:00:00+00:00",
    "dut": "mac_array",
    "variant": "fault_acc_overflow",
    "config": {"n": 4, "data_type": 0, "acc_w": 32},
    "test_name": "random_n4_int8_seed1",
    "status": "fail",
    "mismatch_details": {
        "total_elements": 16,
        "mismatch_count": 1,
        "max_abs_error": 65536,
        "first_mismatch": {"row": 0, "col": 3, "expected": -32946, "actual": 32590},
    },
}

_BENCHMARK = {
    "stage_times": {
        "load": 0.0005,
        "extract_features": 0.00003,
        "cluster": 0.064,
        "triage": 31.5,
        "evaluate": 0.0007,
    },
    "total_ai_time": 31.565,
    "manual_baseline_seconds": 10800.0,
    "speedup_factor": 342.6,
    "eval_metrics": {
        "per_label": {
            "accumulator_overflow": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
        },
        "overall_accuracy": 1.0,
        "mean_confidence": 1.0,
        "total_records": 2,
    },
    "timestamp": "2026-07-07T13:00:00+00:00",
}


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "regression_db.jsonl"
    db.write_text(
        json.dumps(_PASS_RECORD) + "\n" + json.dumps(_FAIL_RECORD) + "\n",
        encoding="utf-8",
    )
    return db


@pytest.fixture()
def tmp_reports(tmp_path: Path) -> Path:
    rdir = tmp_path / "reports"
    rdir.mkdir()
    (rdir / "benchmark_20260707.json").write_text(
        json.dumps(_BENCHMARK), encoding="utf-8"
    )
    return rdir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generates_without_errors(tmp_db: Path, tmp_reports: Path, tmp_path: Path) -> None:
    out = generate_dashboard(
        db_path=tmp_db,
        reports_dir=tmp_reports,
        output_path=tmp_path / "dashboard.html",
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_valid_html(tmp_db: Path, tmp_reports: Path, tmp_path: Path) -> None:
    out = generate_dashboard(
        db_path=tmp_db,
        reports_dir=tmp_reports,
        output_path=tmp_path / "dashboard.html",
    )
    content = out.read_text(encoding="utf-8")
    assert "<html" in content
    assert "<table" in content
    assert "Cluster Summary" in content
    assert "Benchmark Timing" in content
    assert "Regression DB Summary" in content
    assert "Per-Cluster Triage Reports" in content


def test_cluster_labels_appear(tmp_db: Path, tmp_reports: Path, tmp_path: Path) -> None:
    out = generate_dashboard(
        db_path=tmp_db,
        reports_dir=tmp_reports,
        output_path=tmp_path / "dashboard.html",
    )
    content = out.read_text(encoding="utf-8")
    # At least one cluster label from the pipeline must appear
    assert any(
        label in content
        for label in ("overflow_fault", "clean_pass", "reset_fault", "sign_error", "uncategorized")
    )


def test_speedup_appears(tmp_db: Path, tmp_reports: Path, tmp_path: Path) -> None:
    out = generate_dashboard(
        db_path=tmp_db,
        reports_dir=tmp_reports,
        output_path=tmp_path / "dashboard.html",
    )
    content = out.read_text(encoding="utf-8")
    assert "342" in content
