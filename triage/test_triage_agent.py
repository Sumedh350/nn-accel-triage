"""Tests for triage/triage_agent.py — no real API calls."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import anthropic
import pytest

from triage_agent import triage


REQUIRED_FIELDS = {"likely_cause", "confidence", "recommended_debug_steps", "affected_configs"}
VALID_CONFIDENCE = {"high", "medium", "low"}


def make_feat(**overrides) -> dict:
    base = {
        "run_id": "test-run",
        "dut": "mac_array",
        "variant": "fault_acc_overflow",
        "test_name": "random_s1",
        "status": "fail",
        "config_n": 4,
        "config_data_type": 0,
        "config_acc_w": 32,
        "mismatch_rate": 0.75,
        "max_abs_error": 65536,
        "error_magnitude_bucket": "large",
        "has_reset_symptom": False,
        "has_overflow_symptom": True,
        "cluster_label": "overflow_fault",
        "outlier": False,
    }
    base.update(overrides)
    return base


def make_mock_client(report: dict | None = None) -> MagicMock:
    client = MagicMock(spec=anthropic.Anthropic)
    payload = report or {
        "likely_cause": "Accumulator width is too narrow, causing truncation.",
        "confidence": "high",
        "recommended_debug_steps": ["Inspect ACC_W parameter", "Run overflow test vectors"],
        "affected_configs": [],
    }
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=json.dumps(payload))]
    client.messages.create.return_value = mock_msg
    return client


# ---------------------------------------------------------------------------
# Grouping and API call count
# ---------------------------------------------------------------------------

def test_grouping_calls_api_per_cluster():
    feats = [
        make_feat(cluster_label="overflow_fault"),
        make_feat(cluster_label="overflow_fault"),
        make_feat(cluster_label="reset_fault", has_reset_symptom=True,
                  has_overflow_symptom=False, variant="fault_reset",
                  error_magnitude_bucket="large"),
        make_feat(cluster_label="sign_error", has_overflow_symptom=False,
                  variant="fault_wrong_sign", error_magnitude_bucket="large"),
    ]
    client = make_mock_client()
    result = triage(feats, client=client)
    assert client.messages.create.call_count == 3
    assert set(result.keys()) == {"overflow_fault", "reset_fault", "sign_error"}


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

def test_report_has_required_fields():
    feats = [make_feat()]
    client = make_mock_client()
    result = triage(feats, client=client)
    report = result["overflow_fault"]
    assert REQUIRED_FIELDS <= set(report.keys())


def test_confidence_is_valid_value():
    feats = [make_feat()]
    client = make_mock_client()
    result = triage(feats, client=client)
    assert result["overflow_fault"]["confidence"] in VALID_CONFIDENCE


def test_debug_steps_is_list():
    feats = [make_feat()]
    client = make_mock_client()
    result = triage(feats, client=client)
    steps = result["overflow_fault"]["recommended_debug_steps"]
    assert isinstance(steps, list)
    assert all(isinstance(s, str) for s in steps)


def test_affected_configs_populated():
    feats = [make_feat(dut="mac_array", config_n=4, variant="fault_acc_overflow")]
    client = make_mock_client()
    result = triage(feats, client=client)
    assert len(result["overflow_fault"]["affected_configs"]) > 0


# ---------------------------------------------------------------------------
# Skipping clean_pass
# ---------------------------------------------------------------------------

def test_clean_pass_not_in_output():
    feats = [
        make_feat(
            status="pass",
            cluster_label="clean_pass",
            mismatch_rate=0.0,
            max_abs_error=0,
            has_reset_symptom=False,
            has_overflow_symptom=False,
            error_magnitude_bucket="none",
            variant="golden",
        )
    ]
    client = make_mock_client()
    result = triage(feats, client=client)
    assert "clean_pass" not in result
    assert client.messages.create.call_count == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty_dict():
    result = triage([])
    assert result == {}


def test_api_error_returns_fallback():
    feats = [make_feat()]
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = anthropic.AnthropicError("connection refused")
    result = triage(feats, client=client)
    report = result["overflow_fault"]
    assert REQUIRED_FIELDS <= set(report.keys())
    assert report["confidence"] == "low"
    assert "API error" in report["likely_cause"]
    assert isinstance(report["recommended_debug_steps"], list)
