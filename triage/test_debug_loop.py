"""Tests for triage/debug_loop.py — no real API calls."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import anthropic
import pytest

from debug_loop import DebugLoop

REQUIRED_FIELDS = {"likely_cause", "confidence", "recommended_debug_steps", "affected_configs"}


def make_record(**overrides) -> dict:
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


def _mock_msg(report: dict) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(report))]
    return msg


def make_mock_client(reports: list[dict]) -> MagicMock:
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = [_mock_msg(r) for r in reports]
    return client


def _report(confidence: str = "low") -> dict:
    return {
        "likely_cause": "Accumulator too narrow.",
        "confidence": confidence,
        "recommended_debug_steps": ["Check ACC_W"],
        "affected_configs": [],
    }


# ---------------------------------------------------------------------------
# Convergence behaviour
# ---------------------------------------------------------------------------

def test_converges_immediately_on_high_confidence():
    client = make_mock_client([_report("high")])
    result = DebugLoop().run("overflow_fault", [make_record()], client=client)
    assert result["converged"] is True
    assert result["turns"] == 1
    assert len(result["history"]) == 1
    assert client.messages.create.call_count == 1


def test_runs_max_turns_when_confidence_stays_low():
    max_turns = 3
    client = make_mock_client([_report("low")] * max_turns)
    result = DebugLoop().run(
        "overflow_fault", [make_record()], client=client, max_turns=max_turns
    )
    assert result["converged"] is False
    assert result["turns"] == max_turns
    assert len(result["history"]) == max_turns
    assert client.messages.create.call_count == max_turns


def test_history_length_matches_turns():
    client = make_mock_client([_report("medium"), _report("high")])
    result = DebugLoop().run("overflow_fault", [make_record()], client=client, max_turns=3)
    assert len(result["history"]) == result["turns"]
    assert result["turns"] == 2


def test_converged_true_only_when_confidence_high():
    client = make_mock_client([_report("medium")] * 2)
    result = DebugLoop().run("overflow_fault", [make_record()], client=client, max_turns=2)
    assert result["converged"] is False

    client2 = make_mock_client([_report("high")])
    result2 = DebugLoop().run("overflow_fault", [make_record()], client=client2, max_turns=2)
    assert result2["converged"] is True


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_handles_api_error_gracefully():
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = anthropic.AnthropicError("timeout")
    result = DebugLoop().run("overflow_fault", [make_record()], client=client)
    assert REQUIRED_FIELDS <= set(result["report"].keys())
    assert result["converged"] is False
    assert result["turns"] >= 1
    assert len(result["history"]) == 1
