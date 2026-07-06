"""Tests for triage/rag_store.py — no real API calls."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from rag_store import RAGStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def make_report(**overrides) -> dict:
    base = {
        "likely_cause": "Accumulator width too narrow.",
        "confidence": "high",
        "recommended_debug_steps": ["Check ACC_W", "Run overflow vectors"],
        "affected_configs": ["mac_array/N=4/INT8/fault_acc_overflow"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Empty store
# ---------------------------------------------------------------------------

def test_empty_store_returns_empty():
    store = RAGStore()
    assert store.query(make_feat()) == []


def test_empty_store_len_is_zero():
    assert len(RAGStore()) == 0


# ---------------------------------------------------------------------------
# add and query
# ---------------------------------------------------------------------------

def test_add_and_query_returns_similar():
    store = RAGStore()
    record = make_feat()
    report = make_report()
    store.add(record, report)

    results = store.query(make_feat())
    assert len(results) == 1
    assert results[0]["report"]["likely_cause"] == report["likely_cause"]


def test_add_increases_len():
    store = RAGStore()
    store.add(make_feat(), make_report())
    assert len(store) == 1
    store.add(make_feat(variant="fault_reset"), make_report())
    assert len(store) == 2


def test_top_k_respected():
    store = RAGStore()
    variants = ["v1", "v2", "v3", "v4", "v5"]
    for v in variants:
        store.add(make_feat(variant=v, run_id=v), make_report())

    results = store.query(make_feat(), top_k=2)
    assert len(results) == 2


def test_top_k_larger_than_store_returns_all():
    store = RAGStore()
    store.add(make_feat(), make_report())
    store.add(make_feat(variant="v2"), make_report())

    results = store.query(make_feat(), top_k=10)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Result dict shape
# ---------------------------------------------------------------------------

def test_matched_fields_populated():
    store = RAGStore()
    store.add(make_feat(), make_report())
    result = store.query(make_feat())[0]
    assert "matched_fields" in result
    assert isinstance(result["matched_fields"], list)


def test_similarity_scores_present():
    store = RAGStore()
    store.add(make_feat(), make_report())
    result = store.query(make_feat())[0]
    assert "similarity" in result
    assert isinstance(result["similarity"], float)


def test_result_contains_record_and_report_keys():
    store = RAGStore()
    rec = make_feat()
    rep = make_report()
    store.add(rec, rep)
    result = store.query(make_feat())[0]
    assert "record" in result
    assert "report" in result


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------

def test_identical_record_scores_highest():
    store = RAGStore()
    target = make_feat(cluster_label="reset_fault", has_reset_symptom=True,
                       has_overflow_symptom=False, error_magnitude_bucket="large")
    other = make_feat(cluster_label="overflow_fault", has_reset_symptom=False,
                      has_overflow_symptom=True, error_magnitude_bucket="large")
    store.add(target, make_report(likely_cause="Reset fault."))
    store.add(other, make_report(likely_cause="Overflow fault."))

    results = store.query(make_feat(cluster_label="reset_fault", has_reset_symptom=True,
                                    has_overflow_symptom=False, error_magnitude_bucket="large"),
                          top_k=2)
    assert results[0]["report"]["likely_cause"] == "Reset fault."


def test_unseen_record_type_returns_best_available():
    store = RAGStore()
    # Store only contains overflow records
    for i in range(3):
        store.add(make_feat(run_id=str(i)), make_report())

    # Query with a reset record — store has no reset record but must return something
    results = store.query(
        make_feat(cluster_label="reset_fault", has_reset_symptom=True,
                  has_overflow_symptom=False, error_magnitude_bucket="medium"),
        top_k=1,
    )
    assert len(results) == 1


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_round_trip():
    store = RAGStore()
    records = [
        make_feat(variant="v1", run_id="r1"),
        make_feat(variant="v2", run_id="r2"),
        make_feat(variant="v3", run_id="r3",
                  cluster_label="reset_fault", has_reset_symptom=True),
    ]
    reports = [make_report(likely_cause=f"Cause {i}") for i in range(3)]
    for rec, rep in zip(records, reports):
        store.add(rec, rep)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    try:
        store.save(path)
        loaded = RAGStore.load(path)
        assert len(loaded) == 3

        # The most similar record to v1 should still be top
        results = loaded.query(make_feat(variant="v1", run_id="r1"))
        causes = [r["report"]["likely_cause"] for r in results]
        assert "Cause 0" in causes
    finally:
        path.unlink(missing_ok=True)


def test_save_produces_valid_json():
    store = RAGStore()
    store.add(make_feat(), make_report())

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    try:
        store.save(path)
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert "record" in data[0]
        assert "report" in data[0]
    finally:
        path.unlink(missing_ok=True)


def test_load_empty_file():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        f.write("[]")
        path = Path(f.name)

    try:
        store = RAGStore.load(path)
        assert len(store) == 0
        assert store.query(make_feat()) == []
    finally:
        path.unlink(missing_ok=True)
