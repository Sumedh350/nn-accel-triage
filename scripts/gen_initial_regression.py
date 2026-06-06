"""
Bootstrap script — populates regression_db.jsonl with initial entries.

Uses software fault models (Python) that replicate each RTL fault's incorrect
arithmetic, avoiding the need for a separate Verilator compilation per variant.
The CocoTB testbench will append live simulation results in a later session.

Run from the project root:
    python scripts/gen_initial_regression.py
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── path setup ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "tb" / "reference_model"))

from mac_ref import mac_ref, _truncate_to_int48  # type: ignore[import]
from quant_ref import quant_ref  # type: ignore[import]
from test_generator import TestGenerator  # type: ignore[import]

_DB_PATH = _ROOT / "regression_db.jsonl"
_tg = TestGenerator()

# Seeds verified to produce int16 overflow for fault_acc_overflow (N=4 INT8).
_FAULT_SEEDS = [1, 9, 13]
# General seeds for golden-only configs.
_GOLDEN_SEEDS = [42, 7, 99]


# ── software fault models for mac_array ─────────────────────────────────────

def _mac_fault_acc_overflow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fault: ACC_W=16 for INT8 — step-by-step 16-bit accumulation with wrap."""
    n = a.shape[0]
    acc = np.zeros((n, n), dtype=np.int16)
    for k in range(n):
        prod = a[:, k : k + 1].astype(np.int16) * b[k : k + 1, :].astype(np.int16)
        acc = (acc.astype(np.int32) + prod.astype(np.int32)).astype(np.int16)
    return acc


def _mac_fault_wrong_sign(
    a: np.ndarray, b: np.ndarray, data_type: int
) -> np.ndarray:
    """Fault: A_reg unsigned — zero-extends A instead of sign-extending."""
    uint_dtype = np.uint8 if data_type == 0 else np.uint16
    a_unsigned = a.view(uint_dtype)  # same bits, unsigned interpretation
    c = a_unsigned.astype(np.int64) @ b.astype(np.int64)
    if data_type == 0:
        return c.astype(np.int32)
    return _truncate_to_int48(c)


def _mac_fault_off_by_one(
    a: np.ndarray, b: np.ndarray, data_type: int
) -> np.ndarray:
    """Fault: loop exits after N-2 steps — last outer product never accumulated."""
    n = a.shape[0]
    c = np.zeros((n, n), dtype=np.int64)
    for k in range(max(0, n - 1)):  # N-1 iterations instead of N
        c += a[:, k : k + 1].astype(np.int64) @ b[k : k + 1, :].astype(np.int64)
    if data_type == 0:
        return c.astype(np.int32)
    return _truncate_to_int48(c)


def _mac_fault_reset(a: np.ndarray, b: np.ndarray, data_type: int) -> np.ndarray:
    """Fault: rst_n polarity inverted — DUT held in reset, outputs all zeros."""
    dtype = np.int32 if data_type == 0 else np.int64
    return np.zeros(a.shape, dtype=dtype)


# ── comparison helper ────────────────────────────────────────────────────────

def _compare(expected: np.ndarray, actual: np.ndarray) -> dict | None:
    """Return mismatch_details dict on failure, None on pass."""
    if np.array_equal(expected, actual):
        return None
    diff = expected.astype(np.int64) - actual.astype(np.int64)
    mismatches = np.argwhere(expected != actual)
    first = mismatches[0]
    return {
        "total_elements": int(expected.size),
        "mismatch_count": int(len(mismatches)),
        "max_abs_error": int(np.abs(diff).max()),
        "first_mismatch": {
            "row": int(first[0]),
            "col": int(first[1]),
            "expected": int(expected[first[0], first[1]]),
            "actual": int(actual[first[0], first[1]]),
        },
    }


# ── record builder ────────────────────────────────────────────────────────────

def _record(
    dut: str,
    variant: str,
    config: dict,
    test_name: str,
    mismatch: dict | None,
) -> dict:
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dut": dut,
        "variant": variant,
        "config": config,
        "test_name": test_name,
        "status": "fail" if mismatch else "pass",
        "mismatch_details": mismatch,
    }


# ── mac_array runs ────────────────────────────────────────────────────────────

def run_mac_array(db: list[dict]) -> None:
    # N=4 INT8: golden + all 4 fault variants
    cfg_int8 = {"n": 4, "data_type": 0, "acc_w": 32}
    for seed in _FAULT_SEEDS:
        vectors = _tg.gen_mac_vectors(n=4, data_type=0, count=1, seed=seed)
        a, b = vectors[0]
        golden = mac_ref(a, b, data_type=0)
        name = f"random_n4_int8_seed{seed}"

        # golden
        db.append(_record("mac_array", "golden", cfg_int8, name, _compare(golden, golden)))

        # fault_acc_overflow
        faulty = _mac_fault_acc_overflow(a, b)
        db.append(_record("mac_array", "fault_acc_overflow", cfg_int8, name,
                           _compare(golden, faulty)))

        # fault_wrong_sign
        faulty = _mac_fault_wrong_sign(a, b, data_type=0)
        db.append(_record("mac_array", "fault_wrong_sign", cfg_int8, name,
                           _compare(golden, faulty)))

        # fault_off_by_one
        faulty = _mac_fault_off_by_one(a, b, data_type=0)
        db.append(_record("mac_array", "fault_off_by_one", cfg_int8, name,
                           _compare(golden, faulty)))

        # fault_reset
        faulty = _mac_fault_reset(a, b, data_type=0)
        db.append(_record("mac_array", "fault_reset", cfg_int8, name,
                           _compare(golden, faulty)))

    # N=4 INT16: golden only
    cfg_int16 = {"n": 4, "data_type": 1, "acc_w": 48}
    for seed in _GOLDEN_SEEDS:
        vectors = _tg.gen_mac_vectors(n=4, data_type=1, count=1, seed=seed)
        a, b = vectors[0]
        golden = mac_ref(a, b, data_type=1)
        name = f"random_n4_int16_seed{seed}"
        db.append(_record("mac_array", "golden", cfg_int16, name, _compare(golden, golden)))

    # N=1 INT8: golden only
    cfg_n1 = {"n": 1, "data_type": 0, "acc_w": 32}
    for seed in _GOLDEN_SEEDS:
        vectors = _tg.gen_mac_vectors(n=1, data_type=0, count=1, seed=seed)
        a, b = vectors[0]
        golden = mac_ref(a, b, data_type=0)
        name = f"random_n1_int8_seed{seed}"
        db.append(_record("mac_array", "golden", cfg_n1, name, _compare(golden, golden)))


# ── quant_unit runs ───────────────────────────────────────────────────────────

def run_quant_unit(db: list[dict]) -> None:
    # N=4 ACC_W=32
    cfg32 = {"n": 4, "acc_w": 32}
    for seed in _GOLDEN_SEEDS:
        vectors = _tg.gen_quant_vectors(n=4, acc_w=32, count=1, seed=seed)
        acc, scale, shift, zero_pt = vectors[0]
        golden = quant_ref(acc, scale, shift, zero_pt)
        name = f"random_n4_aw32_seed{seed}"
        db.append(_record("quant_unit", "golden", cfg32, name, _compare(golden, golden)))

    # N=4 ACC_W=48
    cfg48 = {"n": 4, "acc_w": 48}
    for seed in _GOLDEN_SEEDS:
        vectors = _tg.gen_quant_vectors(n=4, acc_w=48, count=1, seed=seed)
        acc, scale, shift, zero_pt = vectors[0]
        golden = quant_ref(acc, scale, shift, zero_pt)
        name = f"random_n4_aw48_seed{seed}"
        db.append(_record("quant_unit", "golden", cfg48, name, _compare(golden, golden)))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    db: list[dict] = []
    run_mac_array(db)
    run_quant_unit(db)

    with _DB_PATH.open("a") as f:
        for record in db:
            f.write(json.dumps(record) + "\n")

    golden = [r for r in db if r["variant"] == "golden"]
    faults = [r for r in db if r["variant"] != "golden"]
    print(f"Appended {len(db)} records to {_DB_PATH.name}")
    print(f"  golden: {len(golden)} (all pass={all(r['status']=='pass' for r in golden)})")
    print(f"  faults: {len(faults)} (all fail={all(r['status']=='fail' for r in faults)})")


if __name__ == "__main__":
    main()
