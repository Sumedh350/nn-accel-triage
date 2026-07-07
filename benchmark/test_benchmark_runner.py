"""pytest tests for benchmark/benchmark_runner.py — mocks the Claude API."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import anthropic
import pytest

from benchmark.benchmark_runner import MANUAL_SECONDS_PER_CLUSTER, run_benchmark

REQUIRED_STAGE_KEYS = {"load", "extract_features", "cluster", "triage", "evaluate"}
REQUIRED_EVAL_KEYS = {"per_label", "overall_accuracy", "mean_confidence", "total_records"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(tmp_path: Path) -> Path:
    """Minimal JSONL: 1 clean pass + 1 overflow failure."""
    records = [
        {
            "run_id": "aaa",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "dut": "mac_array",
            "variant": "golden",
            "config": {"n": 4, "data_type": 0, "acc_w": 32},
            "test_name": "test_1",
            "status": "pass",
            "mismatch_details": None,
        },
        {
            "run_id": "bbb",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "dut": "mac_array",
            "variant": "fault_acc_overflow",
            "config": {"n": 4, "data_type": 0, "acc_w": 32},
            "test_name": "test_1",
            "status": "fail",
            "mismatch_details": {
                "total_elements": 16,
                "mismatch_count": 1,
                "max_abs_error": 65536,
                "first_mismatch": {"row": 0, "col": 0, "expected": -100, "actual": 100},
            },
        },
    ]
    p = tmp_path / "regression_db.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


def _make_gt(tmp_path: Path) -> Path:
    gt = {
        "golden": {"label": "no_fault", "description": ""},
        "fault_acc_overflow": {"label": "accumulator_overflow", "description": ""},
        "fault_wrong_sign": {"label": "sign_extension_error", "description": ""},
        "fault_off_by_one": {"label": "loop_boundary_error", "description": ""},
        "fault_reset": {"label": "reset_polarity_error", "description": ""},
    }
    p = tmp_path / "ground_truth.json"
    p.write_text(json.dumps(gt))
    return p


def _make_mock_client(confidence: str = "high") -> MagicMock:
    client = MagicMock(spec=anthropic.Anthropic)
    payload = {
        "likely_cause": "Accumulator width is too narrow, causing silent overflow.",
        "confidence": confidence,
        "recommended_debug_steps": ["Inspect ACC_W parameter", "Run overflow vectors"],
        "affected_configs": [],
    }
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=json.dumps(payload))]
    client.messages.create.return_value = mock_msg
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def benchmark_result(tmp_path: Path) -> dict:
    db = _make_db(tmp_path)
    gt = _make_gt(tmp_path)
    return run_benchmark(db_path=db, ground_truth_path=gt, client=_make_mock_client())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stage_times_keys_present(benchmark_result: dict) -> None:
    assert REQUIRED_STAGE_KEYS <= set(benchmark_result["stage_times"].keys())


def test_speedup_factor_gt_one(benchmark_result: dict) -> None:
    # Mocked API is near-instant; 1 failure cluster = 2700 s manual → speedup ≫ 1.
    assert benchmark_result["speedup_factor"] > 1.0


def test_eval_metrics_keys_present(benchmark_result: dict) -> None:
    assert REQUIRED_EVAL_KEYS <= set(benchmark_result["eval_metrics"].keys())


def test_result_json_serializable(benchmark_result: dict) -> None:
    serialized = json.dumps(benchmark_result)
    reloaded = json.loads(serialized)
    assert reloaded["speedup_factor"] == pytest.approx(benchmark_result["speedup_factor"])


def test_benchmark_writes_regression_db_compatible(tmp_path: Path) -> None:
    """run_benchmark(append_to_db=True) appends a valid JSON entry to the db file."""
    db = _make_db(tmp_path)
    gt = _make_gt(tmp_path)
    original_count = len(db.read_text().strip().splitlines())

    run_benchmark(
        db_path=db,
        ground_truth_path=gt,
        client=_make_mock_client(),
        append_to_db=True,
    )

    lines = db.read_text().strip().splitlines()
    assert len(lines) == original_count + 1
    entry = json.loads(lines[-1])
    assert entry["record_type"] == "benchmark_run"
    assert "speedup_factor" in entry
