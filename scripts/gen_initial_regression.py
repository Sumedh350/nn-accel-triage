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
# Seeds for constrained quant vectors (small acc values so faults are observable).
_QUANT_FAULT_SEEDS = [100, 101, 102]


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


def _mac_fault_acc_overflow_w(a: np.ndarray, b: np.ndarray, acc_w: int) -> np.ndarray:
    """Fault: ACC_W reduced to acc_w bits — per-step signed wrap at acc_w bits."""
    n = a.shape[0]
    mask = (1 << acc_w) - 1
    sign_bit = 1 << (acc_w - 1)

    def _trunc(v: int) -> int:
        v = int(v) & mask
        return v - (1 << acc_w) if v >= sign_bit else v

    acc = np.zeros((n, n), dtype=np.int64)
    for k in range(n):
        prod = a[:, k : k + 1].astype(np.int64) @ b[k : k + 1, :].astype(np.int64)
        raw = acc + prod
        acc = np.vectorize(_trunc)(raw)
    return acc.astype(np.int32)


def _mac_fault_loop_over(
    a: np.ndarray, b: np.ndarray, data_type: int
) -> np.ndarray:
    """Fault: COMPUTE loop i <= N — row 0 receives one extra outer-product addition."""
    n = a.shape[0]
    c = np.zeros((n, n), dtype=np.int64)
    for k in range(n):
        c += a[:, k : k + 1].astype(np.int64) @ b[k : k + 1, :].astype(np.int64)
    # i=N wraps to i=0 in the OOB spatial loop; add the last-step product to row 0
    last_k = n - 1
    extra = (
        a[:1, last_k : last_k + 1].astype(np.int64)
        @ b[last_k : last_k + 1, :].astype(np.int64)
    )
    c[0, :] += extra[0, :]
    if data_type == 0:
        return c.astype(np.int32)
    return _truncate_to_int48(c)


def _mac_fault_subtract(
    a: np.ndarray, b: np.ndarray, data_type: int
) -> np.ndarray:
    """Fault: accumulator subtracts partial products instead of adding."""
    n = a.shape[0]
    c = np.zeros((n, n), dtype=np.int64)
    for k in range(n):
        c -= a[:, k : k + 1].astype(np.int64) @ b[k : k + 1, :].astype(np.int64)
    if data_type == 0:
        return c.astype(np.int32)
    return _truncate_to_int48(c)


def _mac_fault_b_unsigned(
    a: np.ndarray, b: np.ndarray, data_type: int
) -> np.ndarray:
    """Fault: B_reg unsigned — zero-extends B values instead of sign-extending."""
    uint_dtype = np.uint8 if data_type == 0 else np.uint16
    b_unsigned = b.view(uint_dtype)
    c = a.astype(np.int64) @ b_unsigned.astype(np.int64)
    if data_type == 0:
        return c.astype(np.int32)
    return _truncate_to_int48(c)


# ── software fault models for quant_unit ─────────────────────────────────────

def _quant_fault_reset(
    acc: np.ndarray, scale: np.ndarray, shift: np.ndarray, zero_pt: np.ndarray
) -> np.ndarray:
    """Fault: reset polarity inverted — unit always in reset, outputs all zeros."""
    return np.zeros(acc.shape, dtype=np.int8)


def _quant_fault_shift_fixed(
    acc: np.ndarray, scale: np.ndarray, shift: np.ndarray, zero_pt: np.ndarray
) -> np.ndarray:
    """Fault: shift by fixed 1 instead of per-channel shift[i]."""
    n = acc.shape[0]
    result = np.empty((n, n), dtype=np.int8)
    for i in range(n):
        s = int(scale[i])
        zp = int(zero_pt[i])
        result[i] = [max(-128, min(127, (int(v) * s >> 1) + zp)) for v in acc[i]]
    return result


def _quant_fault_no_clamp(
    acc: np.ndarray, scale: np.ndarray, shift: np.ndarray, zero_pt: np.ndarray
) -> np.ndarray:
    """Fault: saturation removed — result wraps to 8-bit signed (no clamp)."""
    n = acc.shape[0]
    result = np.empty((n, n), dtype=np.int8)
    for i in range(n):
        s = int(scale[i])
        sh = int(shift[i])
        zp = int(zero_pt[i])
        for j in range(n):
            biased = (int(acc[i][j]) * s >> sh) + zp
            # Truncate lower 8 bits, interpret as signed (RTL: $signed(q_bias[7:0]))
            truncated = biased & 0xFF
            result[i][j] = truncated - 256 if truncated >= 128 else truncated
    return result


def _quant_fault_wrong_sign_zp(
    acc: np.ndarray, scale: np.ndarray, shift: np.ndarray, zero_pt: np.ndarray
) -> np.ndarray:
    """Fault: zero_pt added as unsigned — negative zero-points corrupted."""
    n = acc.shape[0]
    result = np.empty((n, n), dtype=np.int8)
    for i in range(n):
        s = int(scale[i])
        sh = int(shift[i])
        # Treat zero_pt[i] as unsigned uint8 (PROD_W'(zero_pt[i]) in RTL)
        zp = int(zero_pt[i]) & 0xFF
        result[i] = [max(-128, min(127, (int(v) * s >> sh) + zp)) for v in acc[i]]
    return result


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

        # fault_acc_w24
        faulty = _mac_fault_acc_overflow_w(a, b, acc_w=24)
        db.append(_record("mac_array", "fault_acc_w24", cfg_int8, name,
                           _compare(golden, faulty)))

        # fault_acc_w20
        faulty = _mac_fault_acc_overflow_w(a, b, acc_w=20)
        db.append(_record("mac_array", "fault_acc_w20", cfg_int8, name,
                           _compare(golden, faulty)))

        # fault_loop_over
        faulty = _mac_fault_loop_over(a, b, data_type=0)
        db.append(_record("mac_array", "fault_loop_over", cfg_int8, name,
                           _compare(golden, faulty)))

        # fault_subtract
        faulty = _mac_fault_subtract(a, b, data_type=0)
        db.append(_record("mac_array", "fault_subtract", cfg_int8, name,
                           _compare(golden, faulty)))

        # fault_b_unsigned
        faulty = _mac_fault_b_unsigned(a, b, data_type=0)
        db.append(_record("mac_array", "fault_b_unsigned", cfg_int8, name,
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

def _gen_constrained_quant_vector(n: int, seed: int) -> tuple:
    """Generate quant vectors that exercise clamp, shift, and zero_pt faults.

    acc in ±150, scale in [1, 100], fixed shift=6, zero_pt in ±20.  These
    parameters ensure some elements exceed INT8 range (making no_clamp
    observable) while others stay in the interior (making shift_fixed and
    wrong_sign_zp observable).
    """
    rng = np.random.default_rng(seed)
    acc = rng.integers(-150, 151, size=(n, n), dtype=np.int32)
    scale = rng.integers(1, 101, size=(n,), dtype=np.uint16)
    shift = np.full(n, 6, dtype=np.uint8)
    zero_pt = rng.integers(-20, 21, size=(n,), dtype=np.int8)
    return acc, scale, shift, zero_pt


def run_quant_faults(db: list[dict]) -> None:
    """N=4 ACC_W=32: golden + 4 fault variants, using constrained quant seeds."""
    cfg32 = {"n": 4, "acc_w": 32}
    for seed in _QUANT_FAULT_SEEDS:
        acc, scale, shift, zero_pt = _gen_constrained_quant_vector(4, seed)
        golden = quant_ref(acc, scale, shift, zero_pt)
        name = f"constrained_n4_aw32_qfault_seed{seed}"

        db.append(_record("quant_unit", "golden", cfg32, name, _compare(golden, golden)))

        faulty = _quant_fault_reset(acc, scale, shift, zero_pt)
        db.append(_record("quant_unit", "fault_quant_reset", cfg32, name,
                           _compare(golden, faulty)))

        faulty = _quant_fault_shift_fixed(acc, scale, shift, zero_pt)
        db.append(_record("quant_unit", "fault_quant_shift_fixed", cfg32, name,
                           _compare(golden, faulty)))

        faulty = _quant_fault_no_clamp(acc, scale, shift, zero_pt)
        db.append(_record("quant_unit", "fault_quant_no_clamp", cfg32, name,
                           _compare(golden, faulty)))

        faulty = _quant_fault_wrong_sign_zp(acc, scale, shift, zero_pt)
        db.append(_record("quant_unit", "fault_quant_wrong_sign_zp", cfg32, name,
                           _compare(golden, faulty)))


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
    run_quant_faults(db)

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
