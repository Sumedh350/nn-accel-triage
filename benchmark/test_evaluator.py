"""pytest tests for benchmark/evaluator.py — zero real API calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.evaluator import evaluate, format_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_gt(tmp_path: Path, entries: dict) -> Path:
    p = tmp_path / "ground_truth.json"
    p.write_text(json.dumps(entries))
    return p


def _minimal_gt(tmp_path: Path) -> Path:
    return _write_gt(tmp_path, {
        "golden":            {"label": "no_fault",           "description": ""},
        "fault_acc_overflow":{"label": "accumulator_overflow","description": ""},
        "fault_wrong_sign":  {"label": "sign_extension_error","description": ""},
        "fault_off_by_one":  {"label": "loop_boundary_error", "description": ""},
        "fault_reset":       {"label": "reset_polarity_error","description": ""},
    })


def _feat(variant: str, cluster_label: str) -> dict:
    return {"variant": variant, "cluster_label": cluster_label}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_perfect_predictions(tmp_path: Path) -> None:
    gt_path = _minimal_gt(tmp_path)
    features = [
        _feat("golden",            "clean_pass"),
        _feat("fault_acc_overflow","overflow_fault"),
        _feat("fault_wrong_sign",  "sign_error"),
        _feat("fault_off_by_one",  "off_by_one"),
        _feat("fault_reset",       "reset_fault"),
    ]
    reports = {
        "overflow_fault": {"confidence": "high"},
        "sign_error":     {"confidence": "high"},
        "off_by_one":     {"confidence": "high"},
        "reset_fault":    {"confidence": "high"},
    }
    m = evaluate(features, reports, gt_path)

    assert m["overall_accuracy"] == pytest.approx(1.0)
    assert m["total_records"] == 5
    for label, vals in m["per_label"].items():
        assert vals["precision"] == pytest.approx(1.0), label
        assert vals["recall"]    == pytest.approx(1.0), label
        assert vals["f1"]        == pytest.approx(1.0), label


def test_all_wrong_predictions(tmp_path: Path) -> None:
    gt_path = _minimal_gt(tmp_path)
    # Map every record to a deliberately wrong cluster_label.
    # golden       → overflow_fault  (predicted: accumulator_overflow, true: no_fault)
    # acc_overflow → sign_error      (predicted: sign_extension_error, true: accumulator_overflow)
    # wrong_sign   → off_by_one      (predicted: loop_boundary_error,  true: sign_extension_error)
    # off_by_one   → reset_fault     (predicted: reset_polarity_error, true: loop_boundary_error)
    # fault_reset  → clean_pass      (predicted: no_fault,             true: reset_polarity_error)
    features = [
        _feat("golden",            "overflow_fault"),
        _feat("fault_acc_overflow","sign_error"),
        _feat("fault_wrong_sign",  "off_by_one"),
        _feat("fault_off_by_one",  "reset_fault"),
        _feat("fault_reset",       "clean_pass"),
    ]
    reports: dict = {}
    m = evaluate(features, reports, gt_path)

    assert m["overall_accuracy"] == pytest.approx(0.0)
    assert m["total_records"] == 5
    for label, vals in m["per_label"].items():
        assert vals["precision"] == pytest.approx(0.0), label
        assert vals["recall"]    == pytest.approx(0.0), label
        assert vals["f1"]        == pytest.approx(0.0), label


def test_mixed_predictions(tmp_path: Path) -> None:
    gt_path = _minimal_gt(tmp_path)
    # 3 records for no_fault (golden): 2 correct, 1 wrong
    # 2 records for accumulator_overflow: 1 correct, 1 wrong
    features = [
        _feat("golden",            "clean_pass"),    # correct
        _feat("golden",            "clean_pass"),    # correct
        _feat("golden",            "overflow_fault"),# wrong (predicted acc_overflow, true no_fault)
        _feat("fault_acc_overflow","overflow_fault"),# correct
        _feat("fault_acc_overflow","clean_pass"),    # wrong (predicted no_fault, true acc_overflow)
        # remaining labels: 0 support → precision=recall=f1=0.0 by default
        _feat("fault_wrong_sign",  "sign_error"),    # correct
        _feat("fault_off_by_one",  "off_by_one"),    # correct
        _feat("fault_reset",       "reset_fault"),   # correct
    ]
    reports: dict = {}
    m = evaluate(features, reports, gt_path)

    # 6 correct out of 8 (golden→overflow_fault and fault_acc_overflow→clean_pass are wrong)
    assert m["total_records"] == 8
    assert m["overall_accuracy"] == pytest.approx(6 / 8)

    # no_fault: TP=2, FP=1 (acc_overflow mispredicted as no_fault), FN=1 (golden mispredicted)
    nf = m["per_label"]["no_fault"]
    assert nf["precision"] == pytest.approx(2 / 3)
    assert nf["recall"]    == pytest.approx(2 / 3)
    assert nf["f1"]        == pytest.approx(2 / 3)
    assert nf["support"]   == 3

    # accumulator_overflow: TP=1, FP=1, FN=1
    ao = m["per_label"]["accumulator_overflow"]
    assert ao["precision"] == pytest.approx(0.5)
    assert ao["recall"]    == pytest.approx(0.5)
    assert ao["f1"]        == pytest.approx(0.5)
    assert ao["support"]   == 2

    # others: 1 TP each, 0 FP/FN
    for label in ("sign_extension_error", "loop_boundary_error", "reset_polarity_error"):
        assert m["per_label"][label]["f1"] == pytest.approx(1.0), label


def test_format_report_nonempty(tmp_path: Path) -> None:
    gt_path = _minimal_gt(tmp_path)
    features = [_feat("golden", "clean_pass")]
    reports = {"clean_pass": {"confidence": "medium"}}
    m = evaluate(features, reports, gt_path)
    text = format_report(m)
    assert isinstance(text, str)
    assert len(text) > 0
    assert "accuracy" in text.lower()


def test_missing_ground_truth_label(tmp_path: Path) -> None:
    # Ground truth has only 2 variants; one record uses an unknown variant.
    gt_path = _write_gt(tmp_path, {
        "golden":            {"label": "no_fault",           "description": ""},
        "fault_acc_overflow":{"label": "accumulator_overflow","description": ""},
    })
    features = [
        _feat("golden",            "clean_pass"),     # known → scored
        _feat("unknown_variant",   "overflow_fault"), # unknown → skipped
        _feat("fault_acc_overflow","overflow_fault"), # known → scored
    ]
    reports: dict = {}
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = evaluate(features, reports, gt_path)
    # skipped record emits a warning
    assert any("unknown_variant" in str(w.message) for w in caught)
    # only 2 records scored
    assert m["total_records"] == 2
    assert m["overall_accuracy"] == pytest.approx(1.0)
